# NER API

API-сервис для извлечения именованных сущностей из текста с помощью модели **dslim/bert-base-NER** (HuggingFace).

Поддерживаются категории: **PER**, **ORG**, **LOC**.

---

## Запуск проекта

### 1. Клонирование репозитория
```bash
git clone https://github.com/UiminaM/software_engineering.git
cd software_engineering
git checkout series_2
```

### 2. Установка зависимостей
```bash
cd api_model
pip install -r requirements.txt
```
### 3. Запуск сервера
```bash
uvicorn api:app --reload
```

## Endpoints

| Метод | Путь       | Описание                                |
|-------|------------|----------------------------------------|
| GET   | /status    | Проверка состояния модели              |
| POST  | /ner       | Извлечение всех сущностей              |
| POST  | /ner/per   | Извлечение только персональных имен (PER) |
| POST  | /ner/org   | Извлечение только организаций (ORG)   |
| POST  | /ner/loc   | Извлечение только локаций (LOC)       |

## Документация

[Интерактивная документация FastAPI](http://127.0.0.1:8000/docs)  
[ReDoc](http://127.0.0.1:8000/redoc)