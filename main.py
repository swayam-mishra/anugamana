from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb

# ---------------- CONFIG ---------------- #
# Must match what you used in indexer.py
CHROMA_DIR = "chroma_gita"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "gita_verses"

# ---------------- INITIALIZATION ---------------- #
app = FastAPI()

print("Loading model... (this might take a second)")
model = SentenceTransformer(MODEL_NAME)

print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(name=COLLECTION_NAME)

# ---------------- DATA MODELS ---------------- #
class SearchRequest(BaseModel):
    query: str
    limit: int = 3  # Default to returning top 3 results

# ---------------- API ENDPOINTS ---------------- #

@app.get("/")
def home():
    return {"message": "Anugamana API is running correctly."}

@app.post("/search")
def search_verses(request: SearchRequest):
    try:
        # 1. Convert user query to vector
        query_embedding = model.encode(request.query).tolist()

        # 2. Search in ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=request.limit
        )

        # 3. Format the results nicely
        formatted_results = []
        
        # Chroma returns lists of lists (because you can query multiple things at once)
        # We only queried one thing, so we take index 0
        if results['metadatas'] and results['documents']:
            metadatas = results['metadatas'][0]
            documents = results['documents'][0]
            distances = results['distances'][0] if results['distances'] else []

            for i in range(len(documents)):
                formatted_results.append({
                    "text": documents[i],
                    "metadata": metadatas[i],
                    "score": distances[i] if len(distances) > i else None
                })

        return {"results": formatted_results}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))