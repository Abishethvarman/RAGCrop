import os
import json
import requests
import chromadb
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Query
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("❌ API key not found in .env file!")

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./vector_db")
collection = chroma_client.get_collection(name="rag_knowledge_base")

# FastAPI app
app = FastAPI()

# OpenAI headers
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def retrieve_context(query, top_k=3):
    """Retrieves relevant information from ChromaDB."""
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    retrieved_texts = [meta["info"] for meta in results["metadatas"][0]] if results["metadatas"] else []
    return "\n\n".join(retrieved_texts)

def generate_optimized_prompt(user_input, model_prediction, query, context):
    """Creates an optimized prompt based on user input and retrieved knowledge."""
    return f"""
You are an agriculture-specific assistant. Do NOT use your own knowledge or assumptions.
Only answer based on the provided context. If the context does not contain the answer, reply:
"I'm sorry, but I couldn't find relevant information in the database."

Be clear and accurate based only on the context.

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
    """Handles user queries using retrieved knowledge and OpenAI GPT model."""

    # Determine user input
    user_input = f"Crop: {crop_name}" if crop_name else "Environmental Conditions:\n" + "\n".join(
        [f"{i+1}. {feat}" for i, feat in enumerate([
            feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7
        ]) if feat]
    )

    # Retrieve context from ChromaDB
    context = retrieve_context(user_query)

    if not context:
        return {"response": "I'm sorry, but I couldn't find relevant information in the database."}

    # Generate structured prompt
    structured_prompt = generate_optimized_prompt(user_input, model_prediction, user_query, context)

    # Send request to OpenAI
    data = {
        "model": "gpt-4",  # You can switch to "gpt-3.5-turbo" if needed
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "You are an agriculture assistant. Only respond based on the retrieved context from the vector database. Do not use your own knowledge."},
            {"role": "user", "content": structured_prompt}
        ]
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        return {"response": response.json()["choices"][0]["message"]["content"].strip()}
    else:
        return {"error": response.text}
