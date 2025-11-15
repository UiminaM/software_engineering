from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from text_model import ner, normalize_scores
from typing import List

app = FastAPI(
    title="NER API",
    description="API для распознавания именованных сущностей с помощью BERT",
)

class TextInput(BaseModel):
    text: str

class Entity(BaseModel):
    entity_group: str
    word: str
    score: float
    start: int
    end: int

class EntitiesResponse(BaseModel):
    entities: List[Entity]


@app.get("/", response_class=HTMLResponse)
def root():
    return """
<pre>
Welcome to NER API!

Метод    Путь       Описание
GET     /status     Проверка состояния модели
POST    /ner        Извлечение всех сущностей
POST    /ner/per    Извлечение только персональных имен (PER)
POST    /ner/org    Извлечение только организаций (ORG)
POST    /ner/loc    Извлечение только локаций (LOC)
</pre>
"""

@app.get("/status")
def model_status():
    """Проверка состояния модели."""
    try:
        test_output = ner("Test")
        model_ready = True if test_output is not None else False
    except Exception:
        model_ready = False
    return {
        "status": "Model loaded and ready" if model_ready else "Model not ready",
        "model": "dslim/bert-base-NER"
    }


@app.post("/ner")
def get_all_entities(input_data: TextInput, response_model=EntitiesResponse):
    """Извлекает все сущности из текста."""
    result = ner(input_data.text)
    return {"entities": normalize_scores(result)}


@app.post("/ner/per")
def get_person_entities(input_data: TextInput, response_model=EntitiesResponse):
    """Извлекает только персональные имена (PER)."""
    result = ner(input_data.text)
    filtered = [r for r in result if r["entity_group"] == "PER"]
    return {"PER_entities": normalize_scores(filtered)}


@app.post("/ner/org")
def get_organization_entities(input_data: TextInput, response_model=EntitiesResponse):
    """Извлекает только организации (ORG)."""
    result = ner(input_data.text)
    filtered = [r for r in result if r["entity_group"] == "ORG"]
    return {"ORG_entities": normalize_scores(filtered)}


@app.post("/ner/loc")
def get_location_entities(input_data: TextInput, response_model=EntitiesResponse):
    """Извлекает только локации (LOC)."""
    result = ner(input_data.text)
    filtered = [r for r in result if r["entity_group"] == "LOC"]
    return {"LOC_entities": normalize_scores(filtered)}
