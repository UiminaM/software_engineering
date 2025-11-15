from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

def load_ner_model():
    model_name = "dslim/bert-base-NER"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    return ner_pipeline

def normalize_scores(entities: list):
    for ent in entities:
        ent["score"] = float(ent["score"])
    return entities

ner = load_ner_model()
