import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)
TEST_TEXT = "Arkady Volozh founded Yandex in Moscow, Russia."

def test_status():
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model" in data
    assert data["status"] == "Model loaded and ready"

def test_all_entities():
    response = client.post("/ner", json={"text": TEST_TEXT})
    assert response.status_code == 200
    data = response.json()
    assert "entities" in data
    assert isinstance(data["entities"], list)
    assert len(data["entities"]) > 0

    for ent in data["entities"]:
        assert "entity_group" in ent
        assert "word" in ent
        assert "score" in ent
        assert isinstance(ent["score"], float)
        assert isinstance(ent["word"], str)

def test_per_entities():
    response = client.post("/ner/per", json={"text": TEST_TEXT})
    assert response.status_code == 200
    data = response.json()
    assert "PER_entities" in data
    assert isinstance(data["PER_entities"], list)
    for ent in data["PER_entities"]:
        assert ent["entity_group"] == "PER"

def test_org_entities():
    response = client.post("/ner/org", json={"text": TEST_TEXT})
    assert response.status_code == 200
    data = response.json()
    assert "ORG_entities" in data
    for ent in data["ORG_entities"]:
        assert ent["entity_group"] == "ORG"

def test_loc_entities():
    response = client.post("/ner/loc", json={"text": TEST_TEXT})
    assert response.status_code == 200
    data = response.json()
    assert "LOC_entities" in data
    for ent in data["LOC_entities"]:
        assert ent["entity_group"] == "LOC"

def test_invalid_json():
    response = client.post("/ner", json={"wrong_key": "some text"})
    assert response.status_code == 422

    response = client.post("/ner", json={"text": 12345})
    assert response.status_code == 422