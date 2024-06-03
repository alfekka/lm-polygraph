CLAIM_EXTRACTION_PROMPTS = {"en" : """Please breakdown the sentence into independent claims.

Example:
Sentence: \"He was born in London and raised by his mother and father until 11 years old.\"
Claims:
- He was born in London.
- He was raised by his mother and father.
- He was raised by his mother and father until 11 years old.

Sentence: \"{sent}\"
Claims:"""
,
"ar": """فكك النص التالي الى ادعاءات منفصلة.

Example:
Sentence: \"طارق ذياب هو لاعب كرة قدم تونسي سابق.\"
Claims:
- طارق ذياب لاعب كرة قدم.
- طارق ذياب تونسي.
- طارق ذياب لاعب سابق.

الجملة: \"{sent}\"
الادعاءات:"""                          
,
"ru": """Пожалуйста разбей предложение на независимые утверждения.

Example:
Sentence: \"Он родился в Лондоне и воспитывался матерью и отцом до 11 лет.\"
Claims:
- Он родился в Лондоне.
- Он воспитывался матерью и отцом.
- Он воспитывался матерью и отцом до 11 лет.

Sentence: \"{sent}\"
Claims:"""
}

MATCHING_PROMPTS = { "en": (
    "Given the fact, identify the corresponding words "
    "in the original sentence that help derive this fact. "
    "Please list all words that are related to the fact, "
    "in the order they appear in the original sentence, "
    "each word separated by comma.\nFact: {claim}\n"
    "Sentence: {sent}\nWords from sentence that helps to "
    "derive the fact, separated by comma: "
),
"ar": (
"""بناءً على الحقيقة، حدد الكلمات المقابلة في الجملة الأصلية التي تساعد في استنتاج هذه الحقيقة. يرجى سرد جميع الكلمات المتعلقة بالحقيقة، بالترتيب الذي تظهر به في الجملة الأصلية، وكل كلمة مفصولة بفاصلة.
الحقيقة: {claim}
الجملة: {sent}
الكلمات من الجملة التي تساعد في استنتاج الحقيقة، مفصولة بفاصلة: """
),
"ru": (
    "Используя факт, определи соответствующие слова "
    "в исходном предложении, которые помогают получить этот факт "
    "Пожалуйста, перечисли все слова через запятую, "
    "имеющие отношение к данному не изменяя форм слов"
    "в том порядке, в котором они появляются в исходном предложении. "
    "Не используй кавычки при перечислении."
    "Не меняй форму слова"
    "\nFact: {claim}\n"
    "Sentence: {sent}\nСлова из предложения, которые помогают "
    "получить факт, через запятую без кавычек: "
)
                   }


OPENAI_FACT_CHECK_PROMPT = { "en": (
    "Determine if all provided information in the following claim"
    "is true according to the most recent sources of information."

),
"ar": (
    """
هل الادعاءات صحيحة وفقًا لأحدث مصادر المعلومات؟
أجب ب "نعم"، "لا" أو "لا يُعرف".
مثال:
السياق: أعطني سيرة ذاتية لألبرت أينشتاين.
الادعاء: وُلد في 14 مارس.
الجواب: نعم
السياق: أعطني سيرة ذاتية لألبرت أينشتاين.
الادعاء: وُلد في المملكة المتحدة.
الجواب: لا
الادعاء: {claim}
السياق: {input}
الجواب:
"""
),
"ru": (    
   "Question: {input}\n"
    "Определи, соответствует ли вся предоставленная информация в следующем"
    "утверждении действительности согласно самым последним источникам информации. \n"
    "Think step by step on how to summarize the claim within the provided <sketchpad>. \n"
    "Then, return a <summary> based on the <sketchpad>."
    "\n\n"
    "Claim: {claim}\n"
    "Answer: "

)
}
OPENAI_FACT_CHECK_SUMMARIZE_PROMPT = { "en": (
    """Summarize this reply into one word, whether the claim is true: "True", "False" or "Not known"."""
),
"ar": (
    """
هل الادعاءات صحيحة وفقًا لأحدث مصادر المعلومات؟
أجب ب "نعم"، "لا" أو "لا يُعرف".
مثال:
السياق: أعطني سيرة ذاتية لألبرت أينشتاين.
الادعاء: وُلد في 14 مارس.
الجواب: نعم
السياق: أعطني سيرة ذاتية لألبرت أينشتاين.
الادعاء: وُلد في المملكة المتحدة.
الجواب: لا
الادعاء: {claim}
السياق: {input}
الجواب:
"""
),
"ru": ( 
    """Question: {input}

Claim: {claim}

Is the following claim true?

Reply: {reply}

Summarize this reply into one word, whether the claim is true: "True", "False" or "Not known".
"""
    
)
}
