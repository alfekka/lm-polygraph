import numpy as np
import torch

from collections import defaultdict
from typing import List, Set, Dict, Tuple, Optional
from tqdm import tqdm

from lm_polygraph.utils.dataset import Dataset
from lm_polygraph.utils.model import WhiteboxModel, BlackboxModel, Model
from lm_polygraph.utils.processor import Processor
from lm_polygraph.generation_metrics.generation_metric import GenerationMetric
from lm_polygraph.ue_metrics.ue_metric import (
    UEMetric,
    get_random_scores,
    normalize_metric,
)
from lm_polygraph.estimators.estimator import Estimator
from lm_polygraph.stat_calculators.stat_calculator import StatCalculator
from lm_polygraph.utils.builder_enviroment_stat_calculator import (
    BuilderEnvironmentStatCalculator,
)
from lm_polygraph.utils.factory_stat_calculator import (
    FactoryStatCalculator,
    StatCalculatorContainer,
)
from lm_polygraph.utils.common import flatten_results
from lm_polygraph.utils.uq_pipeline import UQPipeline

import logging

log = logging.getLogger("lm_polygraph")


def _check_unique_names(xs):
    names = set()
    for x in xs:
        if str(x) in names:
            raise Exception(f"Got multiple __str__ values for {x}")
        names.add(str(x))


def _delete_nans(ue, metric):
    new_ue, new_metric = [], []
    for i in range(len(metric)):
        if not np.isnan(metric[i]) and not np.isnan(ue[i]):
            if not isinstance(ue[i], complex):
                new_ue.append(ue[i])
            else:
                new_ue.append(ue[i].real)
            new_metric.append(metric[i])

    return np.array(new_ue), np.array(new_metric)


def order_calculators(
    stats: List[str],
    stat_calculators: Dict[str, StatCalculator],
    stat_dependencies: Dict[str, List[str]],
) -> Tuple[List[str], Set[str]]:
    ordered: List[str] = []
    have_stats: Set[str] = set()
    while len(stats) > 0:
        stat = stats[0]
        if stat in have_stats:
            stats = stats[1:]
            continue
        dependent = False
        if stat not in stat_dependencies.keys():
            raise Exception(
                f"Cant find stat calculator for: {stat}. Maybe you forgot to register it in "
                + "lm_polygraph.utils.register_stat_calculators.register_stat_calculators()?"
            )
        for d in stat_dependencies[stat]:
            if d not in have_stats:
                stats = [d] + stats
                if stats.count(d) > 40:
                    raise Exception(f"Found possibly cyclic dependencies: {d}")
                dependent = True
        if not dependent:
            stats = stats[1:]
            ordered.append(stat)
            for new_stat in stat_calculators[stat].meta_info()[0]:
                have_stats.add(new_stat)

    return ordered, have_stats


class UEManager(UQPipeline):
    """
    Manager to conduct uncertainty estimation experiments by using several uncertainty methods, ground-truth
    uncertainty values and correlation metrics at once. Used for running benchmarks.

    Examples:

    ```python
    >>> from lm_polygraph import WhiteboxModel
    >>> from lm_polygraph.utils.dataset import Dataset
    >>> from lm_polygraph.estimators import *
    >>> from lm_polygraph.ue_metrics import *
    >>> from lm_polygraph.generation_metrics import *
    >>> model = WhiteboxModel.from_pretrained(
    ...     'bigscience/bloomz-560m',
    ...     device='cuda:0',
    ... )
    >>> dataset = Dataset.load(
    ...     '../workdir/data/triviaqa.csv',
    ...     'question', 'answer',
    ...     batch_size=4,
    ... )
    >>> ue_methods = [MaximumSequenceProbability(), SemanticEntropy()]
    >>> ue_metrics = [RiskCoverageCurveAUC()]
    >>> ground_truth = [RougeMetric('rougeL'), BartScoreSeqMetric('rh')]
    >>> man = UEManager(dataset, model, ue_methods, ground_truth, ue_metrics, processors=[])
    >>> results = man()
    >>> results.save("./manager.man")
    ```
    """

    def __init__(
        self,
        data: Dataset,
        model: Model,
        estimators: List[Estimator],
        builder_env_stat_calc: BuilderEnvironmentStatCalculator,
        available_stat_calculators: List[StatCalculatorContainer],
        generation_metrics: List[GenerationMetric],
        ue_metrics: List[UEMetric],
        processors: List[Processor],
        ignore_exceptions: bool = True,
        verbose: bool = True,
        max_new_tokens: int = 100,
    ):
        """
        Parameters:
            data (Dataset): Dataset to run benchmark on.
            model (Model): Model to run benchmark on. Can be either lm_polygraph.WhiteboxModel or
                lm_polygraph.BlackboxModel
            estimators (List[Estimator]): List of estimators to evaluate at benchmark.
            generation_metrics (List[GenerationMetrics]): List of methods to use to calculate ground-truth uncertainty.
            ue_metrics (List[UEMetric]): List of methods to measure correlation between ground-truth uncertainties from
                `generation_metrics` and uncertainty estimators in `estimators`.
            processors (List[Processor]): List of processors to apply after each batch.
            train_data (Optional[Dataset]): Dataset to train density-based estimators on. Can be set to None, if
                no density-based method is used. Default: None.
            ignore_exceptions (bool): If true, exceptions on a new batch will be printed to stderr and
                the batch will be skipped. Useful to skip CUDA OOM errors on large datasets. Default: True.
            deberta_batch_size (int): Batch size for DeBERTa model used in some estimators. Default: 10.
            deberta_device (Optional[str]): The device to run deberta on. If None, will use 'cuda:0' if available,
                'cpu' otherwise. Default: None.
            language (str): Language to test in claim-level benchmark, one of 'en', 'zh', 'ar', 'ru'. Default: 'en'.
            verbose (bool): If set, will print useful info during batch processing. Default: True.
            max_new_tokens (int): Maximum new tokens to use in generation. Default: 100.
        """

        self.model: Model = model
        self.data: Dataset = data
        self.estimators: List[Estimator] = estimators
        self.generation_metrics: List[GenerationMetric] = generation_metrics
        self.ue_metrics: List[UEMetric] = ue_metrics
        _check_unique_names(generation_metrics)
        _check_unique_names(estimators)
        _check_unique_names(ue_metrics)

        self.gen_metrics: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self.estimations: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self.metrics: Dict[Tuple[str, str, str, str], float] = {}
        self.total_bad_estimators: Dict[Estimator, float] = {}
        self.stats: Dict[str, List] = defaultdict(list)

        self.processors = processors
        self.ignore_exceptions = ignore_exceptions
        self.verbose = verbose
        self.max_new_tokens = max_new_tokens

        self.stat_calculator_descr = available_stat_calculators
        self.factory_stat_calc = FactoryStatCalculator(builder_env_stat_calc)

        self.init()

    def init(self):
        log.info("=" * 100)
        log.info("Initializing stat calculators...")

        self.stat_calculators_dict = dict()
        for sc in self.stat_calculator_descr:
            for stat in sc.stats:
                self.stat_calculators_dict[stat] = sc
            
        #stat_calculators_dict = {sc.name: sc for sc in self.stat_calculator_descr}
        stat_dependencies_dict = dict()
        for sc in self.stat_calculator_descr:
            for stat in sc.stats:
                stat_dependencies_dict[stat] = sc.dependencies

        greedy = ["greedy_texts"]
        if not isinstance(self.model, BlackboxModel):
            greedy += ["greedy_tokens"]

        stats = (
            [s for e in self.estimators for s in e.stats_dependencies]
            + [s for m in self.generation_metrics for s in m.stats_dependencies]
            + greedy
        )

        stats, have_stats = order_calculators(
            stats,
            self.stat_calculators_dict,
            stat_dependencies_dict,
        )

        self.stats_names = stats
        stats = [
            s
            for s in stats
            if not (str(s).startswith("ensemble_"))
            and not (
                (
                    str(s).startswith("blackbox_")
                    and s[len("blackbox_") :] in have_stats
                )  # remove blackbox_X from stats only if X is already in stats to remove duplicated run of stat calculator
            )
        ]  # below in calculate() we copy X in blackbox_X

        self.stat_calculators = self.factory_stat_calc(
            [self.stat_calculators_dict[c] for c in stats]
        )

        if self.verbose:
            log.info(f"Stat calculators: {str(self.stat_calculators)}")

        log.info("Done intitializing stat calculators...")

    def __call__(self) -> Dict[Tuple[str, str, str, str], float]:
        """
        Runs benchmark and reports metrics results. Saves all useful calculated statistics for further usage.
        The run includes:
        * Calculating uncertainty estimations for each `estimator` for all input texts in the dataset
        * Calculating ground-truth uncertainties for each `generation_metrics` for all input texts in the dataset.
        * Calculating correlation measure for each `ue_metrics`, between each pair of
          (uncertainty estimation, ground-truth uncertainty) which comes from the same level
          (both 'sequence' or both 'token').
        * Saving uncertainty estimations, ground-truth uncertainties and ue_metrics values for further usage.

        Returns:
            [Tuple[str, str, str, str], float]: dictionary with metrics results. Dictionary keys consist of
                - uncertainty estimation level: 'sequence' or 'token',
                - estimator name,
                - generation metrics name,
                - `ue_metrics` name which was used to calculate quality.
        """
        iterable_data = tqdm(self.data) if self.verbose else self.data

        def fn_on_batch_callback(
            batch_i, target_texts, batch_stats, batch_estimations, bad_estimators
        ):
            for bad_estimator in bad_estimators:
                key = (bad_estimator.level, str(bad_estimator))
                self.estimations.pop(key, None)
                self.estimators.remove(bad_estimator)
                self.total_bad_estimators[bad_estimator] = batch_i

            batch_gen_metrics: Dict[Tuple[str, str], List[float]] = defaultdict(list)
            for generation_metric in self.generation_metrics:
                m = generation_metric(batch_stats, target_texts=target_texts)
                if not isinstance(m, list):
                    m = m.tolist()
                if generation_metric.level == "claim":
                    m = flatten_results(m, generation_metric)
                self.gen_metrics[generation_metric.level, str(generation_metric)] += m
                batch_gen_metrics[generation_metric.level, str(generation_metric)] += m

            for key in ["greedy_texts", "greedy_tokens"]:
                if key in batch_stats.keys():
                    self.stats[key] += batch_stats[key]
            for processor in self.processors:
                processor.on_batch(batch_stats, batch_gen_metrics, batch_estimations)

        super().__call__(iterable_data, fn_on_batch_callback)

        for (e_level, e_name), estimator_values in self.estimations.items():
            for (gen_level, gen_name), generation_metric in self.gen_metrics.items():
                for ue_metric in self.ue_metrics:
                    if gen_level != e_level:
                        continue
                    if len(estimator_values) != len(generation_metric):
                        raise Exception(
                            f"Got different number of metrics for {e_name} and {gen_name}: "
                            f"{len(estimator_values)} and {len(generation_metric)}"
                        )
                    # TODO: Report how many nans!
                    # This is important to know for a user
                    ue, metric = _delete_nans(estimator_values, generation_metric)
                    if len(ue) == 0:
                        self.metrics[e_level, e_name, gen_name, str(ue_metric)] = np.nan
                    else:
                        oracle_score = ue_metric(-metric, metric)
                        random_score = get_random_scores(ue_metric, metric)
                        ue_metric_val = ue_metric(ue, metric)
                        self.metrics[e_level, e_name, gen_name, str(ue_metric)] = (
                            ue_metric_val
                        )
                        self.metrics[
                            e_level, e_name, gen_name, str(ue_metric) + "_normalized"
                        ] = normalize_metric(ue_metric_val, oracle_score, random_score)

        for processor in self.processors:
            processor.on_eval(self.metrics, self.total_bad_estimators)

        return self.metrics

    def save(self, save_path: str):
        """
        Saves the run results in the provided path. Will raise exception, if no results are calculated yet.
        To load the saved manager, see UEManager.load().

        Parameters:
            save_path (str): Path to file to save benchmark results to.
        """
        if len(self.metrics) == 0:
            raise Exception("Nothing to save. Consider calling manager() first.")
        torch.save(
            {
                "metrics": self.metrics,
                "gen_metrics": self.gen_metrics,
                "estimations": self.estimations,
                "stats": self.stats,
            },
            save_path,
        )

    @staticmethod
    def load(load_path: str) -> "UEManager":
        """
        Loads UEManager from the specified path. To save the calculated manager results, see UEManager.save().

        Parameters:
            load_path (str): Path to file with saved benchmark results to load.
        """
        res_dict = torch.load(load_path)
        man = UEManager(None, None, [], [], [], [])
        man.metrics = res_dict.get("metrics", None)
        man.gen_metrics = res_dict.get("gen_metrics", None)
        man.estimations = res_dict.get("estimations", None)
        man.stats = res_dict.get("stats", None)
        return man
