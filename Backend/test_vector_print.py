from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load ChromaDB
chroma_db_path = "./grownius_vdb"
vector_store = Chroma(persist_directory=chroma_db_path, embedding_function=embedding_model)

# Retrieve and print 10 stored vectors
def print_vectors():
    docs = vector_store.get(include=['documents'], ids=None)  # Retrieve all stored documents
    stored_docs = docs['documents'][:10]  # Get the first 10 documents
    
    for i, doc in enumerate(stored_docs):
        print(f"Vector {i+1}: {doc}")

if __name__ == "__main__":
    print_vectors()