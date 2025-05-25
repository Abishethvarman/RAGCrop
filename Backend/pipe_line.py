import os
import json
import requests
import chromadb
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Query
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError("❌ API key not found in .env file!")

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./vector_db")
collection = chroma_client.get_collection(name="rag_knowledge_base")

# FastAPI app
app = FastAPI()

# OpenRouter headers
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
    You are an AI assistant. Your task is to provide responses strictly based on retrieved knowledge.

    ### User Input:  
    Provided Information:  
    {user_input}

    Machine Learning Model's Prediction:  
    {model_prediction}

    ### Retrieved Knowledge:  
    {context}

    ---

    ### Response Instructions:
    - If the user has provided 7 environmental features, return the best crop recommendations.  
    - If the user has provided a crop name, return its suitable growing conditions, environmental requirements, and suggestions.  
    - If relevant information is unavailable, say:  
      "I'm sorry, but I couldn't find relevant information in the database."

    ---

    ### User's Query:  
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
    """Handles user queries using retrieved knowledge and OpenRouter's GPT-4o."""
    
    # Determine user input type
    user_input = ""
    if crop_name:
        user_input = f"Crop: {crop_name}"
    else:
        features = [f"{i+1}. {feat}" for i, feat in enumerate([feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7]) if feat]
        user_input = f"Environmental Conditions:\n" + "\n".join(features)

    # Retrieve knowledge from ChromaDB
    context = retrieve_context(user_query)

    if not context:
        return {"response": "I'm sorry, but I couldn't find relevant information in the database."}

    # Generate structured prompt
    structured_prompt = generate_optimized_prompt(user_input, model_prediction, user_query, context)

    # Send request to OpenRouter API
    data = {
        "model": "openai/gpt-4o",
        "messages": [{"role": "user", "content": structured_prompt}]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        return {"response": response.json()["choices"][0]["message"]["content"]}
    else:
        return {"error": response.text}