from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from llm import LLM

app = FastAPI(title="Local LLM API")
llm = LLM()


class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str


@app.get("/", response_class=HTMLResponse)
def root():
    return """
<pre>
Welcome to Local LLM API!

Метод    Путь            Описание
GET     /status          Проверка состояния модели
POST    /chat            Обычный чат
POST    /chat/crossword  Ответы в кроссворде (одно слово)
POST    /chat/summarize  Резюме текста в одном предложении
</pre>
"""

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    content = llm.chat(request.message)
    return ChatResponse(response=content)

@app.post("/chat/crossword", response_model=ChatResponse)
def chat_crossword(request: ChatRequest):
    content = llm.chat_crossword(request.message)
    return ChatResponse(response=content)

@app.post("/chat/summarize", response_model=ChatResponse)
def chat_summarize(request: ChatRequest):
    content = llm.chat_summarize(request.message)
    return ChatResponse(response=content)

@app.get("/status")
def status():
    if llm.status():
        return {"status": "Model loaded and ready", "model": "gemma3:1b"}
    else:
        return {"status": "Model not ready", "model": "gemma3:1b"}