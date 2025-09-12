import os
import json
from decimal import Decimal
from datetime import date
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from dotenv import load_dotenv
import mysql.connector
import chromadb
from chromadb.utils import embedding_functions

# --- Setup and Initialization ---

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(
    title="Coffee Shop AI Microservice",
    description="Handles AI-powered querying, vector DB operations, and direct SQL analysis.",
)


@app.get("/health/database", status_code=200)
def health_database():
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"), user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"), database=os.getenv("DB_DATABASE")
        )
        if conn.is_connected():
            conn.close()
            return {"database": "connected"}
        else:
            return {"database": "unreachable"}
    except mysql.connector.Error as err:
        return {"database": f"error: {err}"}


@app.get("/health/chromadb", status_code=200)
def health_chromadb():
    try:
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_or_create_collection(name="coffee_shop_receipts")
        return {"chromadb": "connected"}
    except Exception as e:
        return {"chromadb": f"error: {e}"}


@app.get("/health/gemini_api", status_code=200)
def health_gemini_api():
    try:
        test_model = genai.GenerativeModel('gemini-1.5-flash')
        test_chat = test_model.start_chat()
        test_response = test_chat.send_message("Hello")
        if test_response.text:
            return {"gemini_api": "connected"}
        else:
            return {"gemini_api": "unreachable"}
    except Exception as e:
        return {"gemini_api": f"error: {e}"}


@app.get("/health", status_code=200)
def health_check():
    health_status = {
        "database": health_database()["database"],
        "chromadb": health_chromadb()["chromadb"],
        "gemini_api": health_gemini_api()["gemini_api"]
    }
    return health_status


# ChromaDB for semantic search fallback
chroma_client = chromadb.PersistentClient(path="./chroma_db")
google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=os.getenv("GEMINI_API_KEY"))
collection = chroma_client.get_or_create_collection(name="coffee_shop_receipts", embedding_function=google_ef)


# --- Database Connection & Utility ---

def get_db_connection():
    try:
        return mysql.connector.connect(
            host=os.getenv("DB_HOST"), user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"), database=os.getenv("DB_DATABASE")
        )
    except mysql.connector.Error as err:
        print(f"Error connecting to MySQL: {err}")
        return None


def default_json_serializer(obj):
    """Handle non-serializable types like Decimal and date."""
    if isinstance(obj, Decimal): return float(obj)
    if isinstance(obj, date): return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def execute_query_and_fetch(query: str, params=None):
    """A robust utility to execute queries and fetch results."""
    print(f"Executing SQL: {query}" + (f" with params: {params}" if params else ""))
    conn = get_db_connection()
    if not conn: return "Error: Could not connect to the database."
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query, params or ())
        results = cursor.fetchall()
        return json.dumps(results, default=default_json_serializer)
    except mysql.connector.Error as err:
        print(f"SQL Error: {err}")
        return f"Error executing query: {err}"
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


# --- Agent Tools ---

def get_last_selling_product():
    """Finds the most recently sold product based on receipt creation time."""
    query = """
            SELECT li.item_name, r.created_at
            FROM receipts r
                     JOIN line_items li ON r.receipt_number = li.receipt_number
            ORDER BY r.created_at DESC LIMIT 1; \
            """
    return execute_query_and_fetch(query)


def get_customer_last_visit(customer_name: str):
    """Finds the last visit date for a specific customer by their name."""
    query = "SELECT name, last_visit FROM customers WHERE name LIKE %s ORDER BY last_visit DESC LIMIT 1;"
    return execute_query_and_fetch(query, (f"%{customer_name}%",))


def get_sales_on_date(target_date: str):
    """Calculates the total sales for a specific date (YYYY-MM-DD)."""
    query = "SELECT SUM(total_money) as total_sales FROM receipts WHERE DATE(created_at) = %s;"
    return execute_query_and_fetch(query, (target_date,))


def get_best_sales_day():
    """Finds the date with the highest total sales."""
    query = """
            SELECT DATE (created_at) as sales_day, SUM (total_money) as daily_total
            FROM receipts
            GROUP BY sales_day
            ORDER BY daily_total DESC LIMIT 1; \
            """
    return execute_query_and_fetch(query)


def get_most_expensive_item():
    """Finds the item with the highest price listed in any transaction."""
    query = "SELECT item_name, price FROM line_items ORDER BY price DESC LIMIT 1;"
    return execute_query_and_fetch(query)


def get_top_products(limit: int = 5):
    """Retrieves the top N selling products by quantity."""
    limit = int(limit)
    query = "SELECT item_name, SUM(quantity) as total_quantity FROM line_items GROUP BY item_name ORDER BY total_quantity DESC LIMIT %s;"
    return execute_query_and_fetch(query, (limit,))

def get_top_points_customers(limit: int = 5):
    """Retrieves the top N customers by total points."""
    limit = int(limit)
    query = "SELECT name, total_spent, total_points FROM customers WHERE name IS NOT NULL AND total_points > 0 ORDER BY total_points DESC LIMIT %s;"
    return execute_query_and_fetch(query, (limit,))


def get_top_customers(limit: int = 5):
    """Retrieves the top N customers by total spending."""
    limit = int(limit)
    query = "SELECT name, total_spent, total_poins FROM customers WHERE name IS NOT NULL AND total_spent > 0 ORDER BY total_spent DESC LIMIT %s;"
    return execute_query_and_fetch(query, (limit,))


# --- Agent Configuration ---

# "Clerk" Agent for /chat endpoint
clerk_tools = {
    "get_last_selling_product": get_last_selling_product,
    "get_customer_last_visit": get_customer_last_visit,
    "get_sales_on_date": get_sales_on_date,
    "get_best_sales_day": get_best_sales_day,
    "get_most_expensive_item": get_most_expensive_item,
    "get_top_products": get_top_products,
    "get_top_customers": get_top_customers,
    "get_top_points_customers": get_top_points_customers
}
clerk_model = genai.GenerativeModel('gemini-1.5-flash', tools=list(clerk_tools.values()))

# "Strategist" Agent for /analyze endpoint
DB_SCHEMA_PROMPT = """
You are a data analyst for a coffee shop. You have access to a SQL database with the following schema to answer questions:
- `customers` (id, name, email, phone_number, total_visits, total_spent, total_points, last_visit)
- `receipts` (receipt_number, created_at, total_money, customer_id)
- `line_items` (id, receipt_number, item_name, quantity, price)
Your only tool is a SQL executor. Write and execute queries to gather data before answering the user's question.
"""
strategist_tools = {"execute_sql_query_tool": execute_query_and_fetch}
strategist_model = genai.GenerativeModel(
    'gemini-1.5-flash',
    tools=list(strategist_tools.values()),
    system_instruction=DB_SCHEMA_PROMPT
)


# --- Pydantic Models ---
class QueryRequest(BaseModel): question: str


class QueryResponse(BaseModel): answer: str


class EmbedRequest(BaseModel): documents: List[str]


# --- API Endpoints ---
@app.post("/embed-and-store", status_code=201)
def embed_and_store_documents(request: EmbedRequest):
    if not request.documents: raise HTTPException(status_code=400, detail="No documents provided.")
    try:
        doc_ids = [f"doc_{i}" for i in range(len(request.documents))]
        collection.add(documents=request.documents, ids=doc_ids)
        return {"message": f"Successfully embedded and stored {len(request.documents)} documents."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process documents: {e}")


def run_agent(model, tools_map, question):
    """Generic function to run a tool-using agent conversation."""
    chat = model.start_chat(enable_automatic_function_calling=True)
    response = chat.send_message(question)
    return response.text


@app.post("/query/chat", response_model=QueryResponse)
def query_chat_agent(request: QueryRequest):
    """Handles specific, factual questions using the 'Clerk' agent."""
    try:
        answer = run_agent(clerk_model, clerk_tools, request.question)
        return QueryResponse(answer=answer)
    except Exception as e:
        print(f"Clerk agent failed: {e}")
        raise HTTPException(status_code=500, detail="The AI agent encountered an error.")


@app.post("/query/analyze", response_model=QueryResponse)
def query_analysis_agent(request: QueryRequest):
    """Handles complex, analytical questions using the 'Strategist' agent."""
    try:
        answer = run_agent(strategist_model, strategist_tools, request.question)
        return QueryResponse(answer=answer)
    except Exception as e:
        print(f"Strategist agent failed: {e}")
        raise HTTPException(status_code=500, detail="The AI analyst encountered an error.")
