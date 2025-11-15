from langchain_ollama import OllamaLLM
from langchain_classic.agents import initialize_agent, Tool
from langchain_community.utilities import SQLDatabase
from langchain_classic.memory import ConversationBufferMemory
from dotenv import load_dotenv
from sqlalchemy import text
import requests
import re
import os

load_dotenv()
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT")
PG_DB = os.getenv("PG_DB")
ORS_API_KEY = os.getenv("ORS_API_KEY")

db_uri = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
db = SQLDatabase.from_uri(db_uri)

ORS_URL = "https://api.openrouteservice.org/v2/directions/foot-walking"
llm = OllamaLLM(model="qwen3:1.7b")

def get_data(query: str, **kwargs):
    with db._engine.connect() as conn:
        result = conn.execute(text(query), kwargs)
        rows = result.fetchall()
    return rows

def get_products(input=None):
    rows = get_data("SELECT product, description FROM products;")
    if not rows:
        return "Список товаров пуст."
    return rows
    
def geocode(address: str):
    address = re.sub(r"^address\s*=\s*", "", address.strip())
    address = address.strip("'").strip('"')
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": f"Казань, {address}",
        "format": "jsonv2",
        "limit": 1
    }
    headers = {"User-Agent": "shop-agent/1.0"}

    r = requests.get(url, params=params, headers=headers)
    r.raise_for_status()
    data = r.json()

    if not data:
        raise ValueError(f"Адрес не найден: {address}")

    lat = float(data[0]["lat"])
    lon = float(data[0]["lon"])
    return lat, lon


def get_distance(lat1, lon1, lat2, lon2):
    headers = {"Authorization": ORS_API_KEY, "Content-Type": "application/json"}
    body = {"coordinates": [[lon1, lat1], [lon2, lat2]]}
    r = requests.post(ORS_URL, headers=headers, json=body)
    r.raise_for_status()
    data = r.json()
    distance_m = data["routes"][0]["summary"].get("distance", 0.0)
    return round(distance_m / 1000, 2)


def get_nearest_shops(address: str):
    address = str(address).strip().strip('"').strip("'")
    lat_user, lon_user = geocode(address)
    rows = get_data("SELECT address, schedule, lat, lon FROM stores;")

    distances = []
    for addr, schedule, lat, lon in rows:
        try:
            dist = get_distance(lat_user, lon_user, lat, lon)
            distances.append((addr, schedule, dist))
        except Exception as e:
            print(f"Ошибка при расчете маршрута к {address}: {e}")
    nearest = sorted(distances, key=lambda x: x[2])
    return nearest


def get_prices(product_name: str):
    product_name = str(product_name).strip().strip('"').strip("'")
    rows = get_data("""
        SELECT s.address, i.quantity, i.price
        FROM inventory i
        JOIN stores s ON i.store_id = s.id
        JOIN products pr ON i.product_id = pr.id
        WHERE lower(pr.product) = lower(:product_name)
    """, product_name=product_name) 

    if not rows:
        return f"Товар '{product_name}' не найден в базе или нет данных о ценах."

    prices = [(r[0], float(r[1]), float(r[2])) for r in rows]
    result = f"Цены и наличие товара '{product_name}':\n"
    for addr, amount, price in prices:
        result += f"{addr} — {amount} шт — {price} руб\n"
    return result

tools = [
    Tool(name="Get Products", func=get_products, description="Возвращает финальный список товаров из базы данных."),
    Tool(name="Get Nearest Shops", func=get_nearest_shops, description="Возвращает список ближайших магазинов к определенному адресу. Пример входных данных: ул.Татарстан 7"),
    Tool(name="Get Prices", func=get_prices, description="Возвращает список цен и количество указанного товара по всем магазинам. Вход: название товара (первая буква заглавная, следущие строчные), например 'Молоко'. Выходные данные (адрес, количество, цена)")
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True
)

print(agent.run("Покажи магазин где самая маленькая стоимость сыра"))