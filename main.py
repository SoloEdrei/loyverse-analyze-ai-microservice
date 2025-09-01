import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import google.generativeai as genai
from dotenv import load_dotenv
import mysql.connector
import chromadb
from chromadb.utils import embedding_functions

# --- Setup and Initialization ---

load_dotenv()

# Configure the Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize FastAPI app
app = FastAPI(
    title="Coffee Shop AI Microservice",
    description="Handles AI-powered querying, vector DB operations, and direct SQL analysis.",
)

# Connect to ChromaDB (local persistent vector database)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Use a Google Generative AI embedding function for ChromaDB
google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=os.getenv("GEMINI_API_KEY"))

# Get or create a collection in ChromaDB
collection = chroma_client.get_or_create_collection(
    name="coffee_shop_receipts",
    embedding_function=google_ef
)


# --- Database Connection ---
def get_db_connection():
    """Establishes and returns a connection to the MySQL database."""
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_DATABASE")
        )
        return conn
    except mysql.connector.Error as err:
        print(f"Error connecting to MySQL: {err}")
        return None


# --- Pydantic Models for API validation ---

class EmbedRequest(BaseModel):
    documents: List[str]


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Coffee Shop AI Microservice is running."}

@app.get("/health/db")
def check_db_health():
    conn = get_db_connection()
    if conn:
        conn.close()
        return {"status": "Database connection is healthy."}
    else:
        raise HTTPException(status_code=500, detail="Failed to connect to the database.")

@app.get("/health/vector-db")
def check_vector_db_health():
    try:
        # Attempt to list collections to check if the connection is healthy
        collections = chroma_client.list_collections()
        return {"status": "Vector DB connection is healthy.", "collections": [col.name for col in collections]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to connect to the vector DB: {e}")

@app.get("/health/gemini")
def check_gemini_health():
    try:
        # Attempt to generate a simple response to check if the API is healthy
        response = model.generate_content("Hello, Gemini!")
        if response and response.text:
            return {"status": "Gemini API connection is healthy."}
        else:
            raise HTTPException(status_code=500, detail="Gemini API did not return a valid response.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to connect to the Gemini API: {e}")

@app.get("/health")
def check_service_health():
    db_health = check_db_health()
    vector_db_health = check_vector_db_health()
    gemini_health = check_gemini_health()
    return {
        "database": db_health,
        "vector_db": vector_db_health,
        "gemini_api": gemini_health
    }

@app.post("/embed-and-store", status_code=201)
def embed_and_store_documents(request: EmbedRequest):
    """
    Receives text documents, generates embeddings, and stores them in the vector DB.
    """
    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided.")

    try:
        # ChromaDB requires unique IDs for each document
        doc_ids = [f"doc_{i}_{j}" for i, doc in enumerate(request.documents) for j, chunk in
                   enumerate([doc])]  # Simple ID generation

        collection.add(
            documents=request.documents,
            ids=doc_ids
        )
        return {"message": f"Successfully embedded and stored {len(request.documents)} documents."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process documents: {e}")


@app.post("/query/chat", response_model=QueryResponse)
def query_chat(request: QueryRequest):
    """
    Handles direct business questions by first attempting a precise SQL query,
    then falling back to a vector search for semantic questions.
    """
    question = request.question.lower()

    # Simple routing based on keywords. More sophisticated NLP could be used here.
    if "sold item" in question:
        query = """
                SELECT item_name, SUM(quantity) as total_quantity
                FROM line_items
                GROUP BY item_name
                ORDER BY total_quantity DESC LIMIT 1; \
                """
        result = execute_sql_query(query)
        if result:
            return QueryResponse(
                answer=f"The most sold item is '{result[0][0]}' with a total of {result[0][1]} units sold.")
        else:
            return QueryResponse(answer="I couldn't determine the most sold item from the database.")

    if "best sales day" in question:
        query = """
                SELECT DATE (created_at) as sales_day, SUM (total_money) as daily_total
                FROM receipts
                GROUP BY sales_day
                ORDER BY daily_total DESC
                    LIMIT 1; \
                """
        result = execute_sql_query(query)
        if result:
            return QueryResponse(
                answer=f"Our best sales day was {result[0][0]} with a total of ${result[0][1]:.2f} in sales.")
        else:
            return QueryResponse(answer="I couldn't determine the best sales day from the database.")

    if "best customer" in question:
        query = """
                SELECT name, total_spent, total_points
                FROM customers
                ORDER BY total_spent DESC LIMIT 1; \
                """
        result = execute_sql_query(query)
        if result and result[0][1] and result[0][1] > 0:
            return QueryResponse(
                answer=f"Our best customer is {result[0][0]}, who has spent a total of ${result[0][1]:.2f}.")
        else:
            return QueryResponse(
                answer="I couldn't determine the best customer from the database, or there is no spending data available.")

    # If no specific SQL query matches, use RAG with Vector DB
    return perform_rag_query(question, "Answer the following business question based on the provided receipt data.")


@app.post("/query/analyze", response_model=QueryResponse)
def query_analysis(request: QueryRequest):
    """
    Handles complex analytical questions by always using RAG to provide
    rich context to the Gemini model for forecasting and strategy.
    """
    return perform_rag_query(request.question,
                             "You are a business strategist for a coffee shop. Based on the following sales data, provide a detailed analysis and actionable strategies to answer the user's question.")


# --- Helper Functions ---

def execute_sql_query(query: str):
    """Executes a given SQL query and returns the results."""
    conn = get_db_connection()
    if not conn:
        return None
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except mysql.connector.Error as err:
        print(f"SQL Query failed: {err}")
        return None
    finally:
        cursor.close()
        conn.close()


def perform_rag_query(question: str, system_prompt: str) -> QueryResponse:
    """
    Performs Retrieval-Augmented Generation (RAG).
    1. Queries the vector DB for relevant context.
    2. Sends the context and question to Gemini.
    """
    try:
        # 1. Retrieve relevant documents from ChromaDB
        results = collection.query(
            query_texts=[question],
            n_results=5  # Get the top 5 most relevant receipt summaries
        )

        context = "\n".join(results['documents'][0])

        # 2. Generate a response using Gemini with the retrieved context
        prompt = f"""
        {system_prompt}

        --- Context from Sales Data ---
        {context}
        --- End of Context ---

        Question: {question}

        Answer:
        """

        response = model.generate_content(prompt)
        return QueryResponse(answer=response.text)
    except Exception as e:
        print(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get a response from the AI model.")

