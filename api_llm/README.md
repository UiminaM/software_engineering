# LLM API

## Запуск проекта
```bash
git clone https://github.com/UiminaM/software_engineering.git
cd software_engineering
git checkout series_2

cd api_llm
uvicorn api:app --reload
```

## Endpoints

| Метод | Путь              | Описание |
|-------|-------------------|----------|
| GET   | `/status`         | Проверка работоспособности и доступности модели |
| POST  | `/chat`           | Общение с моделью в обычном диалоговом формате |
| POST  | `/chat/crossword` | Получение одного слова — ответа для кроссворда |
| POST  | `/chat/summarize` | Краткое резюме текста в одном предложении |


## Документация

[Интерактивная документация FastAPI](http://127.0.0.1:8000/docs)  
[ReDoc](http://127.0.0.1:8000/redoc)