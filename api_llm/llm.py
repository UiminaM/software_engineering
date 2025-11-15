import ollama

class LLM:
    def __init__(self, model_name="gemma3:1b"):
        self.model_name = model_name

    def chat_custom(self, user_message: str, system_message: str, temperature: float = 0.7, max_tokens: int = 200) -> str:
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )
        return response["message"]["content"].strip()

    def chat(self, user_message: str) -> str:
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        return response["message"]["content"].strip()

    def chat_crossword(self, user_message: str) -> str:
        system_message = """
        Ты — эксперт по кроссвордам и загадкам.
        Отвечай только одним словом, без пояснений, без пунктуации, без кавычек.
        Если не уверен, выбери наиболее вероятный ответ.
        """
        return self.chat_custom(user_message, system_message, temperature=0.3, max_tokens=10)

    def chat_summarize(self, user_message: str) -> str:
        system_message = """
        Ты — помощник, который умеет кратко и точно передавать суть текста.
        Сделай краткое резюме или вывод по содержанию, максимум в одном предложении.
        Не добавляй вступления, пояснения или комментарии.
        """
        return self.chat_custom(user_message, system_message, temperature=0.4, max_tokens=60)

    def status(self) -> bool:
            try:
                response = ollama.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": "Test"}]
                )
                return True
            except Exception:
                return False