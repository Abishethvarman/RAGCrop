import os
import json
import requests
import chromadb
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Query
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./vector_db")
collection = chroma_client.get_collection(name="rag_knowledge_base")

# FastAPI app
app = FastAPI()

def retrieve_context(query, top_k=3):
    """Retrieves relevant information from ChromaDB."""
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    retrieved_texts = [meta["info"] for meta in results["metadatas"][0]] if results["metadatas"] else []
    return "\n\n".join(retrieved_texts)

def generate_optimized_prompt(user_input, model_prediction, query, context):
    """Creates an optimized prompt based on user input and retrieved knowledge."""
    return f"""
You are a domain-specific assistant for agriculture. DO NOT use your own knowledge or assumptions.

Only answer based on the provided context. If the context does not contain the answer, respond with:
"I'm sorry, but I couldn't find relevant information in the database."

Be accurate, relevant, and clear. Focus on helpful responses backed by context.

---

### User Input:
{user_input}

### ML Model Prediction:
{model_prediction}

### Retrieved Context:
{context}

---

### User Query:
{query}
"""

@app.post("/chat")
async def chat_with_rag(
    feature_1: str = Query(None),
    feature_2: str = Query(None),
    feature_3: str = Query(None),
    feature_4: str = Query(None),
    feature_5: str = Query(None),
    feature_6: str = Query(None),
    feature_7: str = Query(None),
    crop_name: str = Query(None),
    model_prediction: str = Query(None),
    user_query: str = Query(...)
):
    # Determine user input type
    user_input = f"Crop: {crop_name}" if crop_name else "Environmental Conditions:\n" + "\n".join(
        [f"{i+1}. {feat}" for i, feat in enumerate([
            feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7
        ]) if feat]
    )

    # Retrieve context
    context = retrieve_context(user_query)

    if not context:
        return {"response": "I'm sorry, but I couldn't find relevant information in the database."}

    # Generate prompt
    structured_prompt = generate_optimized_prompt(user_input, model_prediction, user_query, context)

    # Use Ollama locally (LLaMA 3.1 8B) - requires Ollama installed and model pulled
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3:8b",  # Ensure llama3:8b is pulled using `ollama pull llama3:8b`
            "prompt": structured_prompt,
            "temperature": 0,
            "system": "You are an agriculture assistant. Only use the context retrieved from the knowledge base. Do not use your own knowledge.",
            "stream": False
        }
    )

    if response.status_code == 200:
        result = response.json()
        return {"response": result.get("response", "").strip()}
    else:
        return {"error": response.text}
