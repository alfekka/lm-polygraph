from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import tqdm
import torch
import pickle
import numpy as np
import pandas as pd
import base64
import pdb

model_path = "Vikhrmodels/Vikhr-7B-instruct_0.2"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

# Since transformers 4.35.0, the GPT-Q/AWQ model can be loaded using AutoModelForCausalLM.
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=True
).eval()
names = ["Джалил Мамедзаде", "Душан Попович", "Лиза Штормит", "Воронов Абрам Соломонович", "Урумов Тамерлан Михайлович",
"Абдухаким Каримович Гафуров",  "Василий Васильевич Бурхановский", "Сергей Алексеевич Христианович",
"Олег Михайлович Белоцерковский", "Всеволод Феликсович Гантмахер", "Николай Николаевич Салащенко",
"Борис Семенович Милютин", "Пелагея Федоровна Шайн", "Зинаида Гиппиус", "Елизавета Петровна Глинка",
"Екатерина Алексеевна Фурцева", "Глеб Максимилианович Кржижановский", "Александр Ильич Ахиезер",
"Александра Михайловна Коллонтай", "Вера Иванова Засулич", "Екатерина Семеновна Сванидзе",
"Валентина Степановна Гризодубова", "Игорь Евгеньевич Тамм", "Шафаревич Игорь Ростиславович",
"Елена Петровна Блаватская", "Ирина Констатиновна Роднина", "Сергей Тимофеевич Аксаков",
"Вера Михайловна Инбер", "Вера Николаевна Фигнер", "Анна Комнина", "Надежда Александровна Тэффи",
"Андрей Михайлович Райгородский", "Андрей Митрофанович Журавский", "Тигран Оганесович Худавердян",
"Геннадий Владимирович Короткевич", "Андрей Андреевич Бреслав", "Александр Александрович Фридман",
"Глеб Евгеньевич Лозино-Лозинский", "Ольга Александровна Ладыженская",  "Борис Викторович Раушенбах",
"Самойлов Вадим Рудольфович", "Диана Арбенина", "Алена Швец", "Нина Карлссон",
"Илья Игоревич Лагутенко", "Александр Николаевич Башлачев", "Александр Борисович Пушной",
"Юрий Юрьевич Музыченко", "Владимир Евгеньевич Кристовский", "Игорь Владимирович Рыбаков",
"Рябушинский Владимир Павлович", "Габриэлян Тамара Овнановна", "Давид Ян", "Виктор Викторович Кантор",
"Михаил Васильевич Нестеров", "Василий Андреевич Архипов", "Александр Иванович Герцен",
"Валерия Ильинична Новодворская", "Анатолий Васильевич Луначарский", "Кузьма Сергеевич Петров-Водкин",
"Владимир Андреевич Фаворский", "Михаил Александрович Врубель", "Сергей Сергеевич Ольденбург",
"Виссарион Григорьевич Белинский", "Василий Владимирович Бартольд", "Людмила Георгиевна Карачкина",
"Виктор Амазаспович Амбарцумян", "Яков Борисович Зельдович", "Андрей Петрович Ершов",
"Андрей Анатольевич Зализняк"]


df = pd.DataFrame({"question":[f"Расскажи биографию {name}" for name in names]})
list_of_inputs = df['question'].tolist()

all_input_texts = []
all_input_ids = []
all_generated_tokens = []
all_cut_alternatives = []
all_alternatives = []
all_output_text = []





for q in tqdm.tqdm(list_of_inputs):    
    input_ids = tokenizer(q, return_tensors="pt").input_ids
    
    outputs = model.generate(
        input_ids.to("cuda"),
        output_scores=True,
        return_dict_in_generate=True,
        max_new_tokens=256,  # не ну какое слово будет длиннее 10 токенов?
        num_beams=1,
        do_sample=False,
        #num_return_sequences=10,
        # logits_processor=LogitsProcessorList([processor])
    )

    transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True).cpu().numpy()
    input_length = 1 if model.config.is_encoder_decoder else input_ids.shape[1]
    generated_tokens = outputs.sequences[:, input_length:]
    output_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    # scores = outputs.scores
    # scores = np.array([t.cpu().numpy() for t in scores])
    # probs = np.exp(transition_scores)

    logits = torch.stack([s for s in outputs.scores], dim=1)
    logits = logits.log_softmax(-1)

    length=len(generated_tokens[0])
    n_alternatives = 10
    alternatives = []
    cut_alternatives = []
    cut_sequences = generated_tokens
    for j in range(length):
        #print(' Position:', j)
        l = logits[0, j, :].cpu().numpy()
        best_tokens = np.argpartition(l, -n_alternatives)[-n_alternatives:]

        alternatives.append([])
        cut_alternatives.append([])

        for t in best_tokens:
            cut_alternatives[-1].append((t.item(), l[t].item()))
        cut_alternatives[-1].sort(key=lambda x: x[0] == cut_sequences[0,j], reverse=True)

        for token, lprob in cut_alternatives[-1]:
            alternatives[-1].append([tokenizer.decode([token]), np.exp(lprob)])

    all_input_texts.append(q)
    all_input_ids.append(input_ids)
    all_generated_tokens.append(generated_tokens)
    all_cut_alternatives.append(cut_alternatives)
    all_alternatives.append(alternatives)
    all_output_text.append(output_text)



data_to_save = {
    "all_input_texts": all_input_texts,
    "all_input_ids": [tensor.cpu().numpy().tolist() for tensor in all_input_ids],
    "all_generated_tokens": [tensor.cpu().numpy().tolist() for tensor in all_generated_tokens],
    "all_cut_alternatives": all_cut_alternatives,
    "all_alternatives": all_alternatives,
    "all_output_text": all_output_text
}

dffin = pd.DataFrame(data_to_save)
dffin.to_csv('bio_1.csv', sep='\t')

data_to_save1 = {
    "all_input_texts": all_input_texts,
    "all_input_ids": [tensor.cpu().numpy().tolist() for tensor in all_input_ids],
    "all_generated_tokens": [tensor.cpu().numpy().tolist() for tensor in all_generated_tokens],
}
dffin1 = pd.DataFrame(data_to_save1)
dffin1.to_csv('bio_2.csv', sep='\t')


with open("russian_vikhr7b_output.json", 'w') as fout:
    json.dump(data_to_save, fout)
