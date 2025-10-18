import ollama

response = ollama.chat(
    model="gemma3:1b",
    messages=[
        {"role": "system", "content": "Ты опытный шеф-повар"},
        {"role": "user", "content": "Поделись рецептом блинов"}
    ]
)

print(response["message"]["content"])
