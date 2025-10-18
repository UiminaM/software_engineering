from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

ner = pipeline("ner", model=model, tokenizer=tokenizer)
example = "Arkady Volozh founded Yandex in Moscow, Russia, in 1997, building it into one of the leading technology companies in Eastern Europe."

ner_results = ner(example)
print(ner_results)