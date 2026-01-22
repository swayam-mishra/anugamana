import pickle
import json
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder 
import chromadb
import numpy as np

# ---------------- CONFIG ---------------- #
CHROMA_DIR = "chroma_gita"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION_NAME = "gita_verses"
BM25_FILE = "bm25_index.pkl"
BM25_IDS_FILE = "bm25_ids.pkl"
DATA_FILE = "gita_full.json" # Used for BM25 filtering lookup

# ---------------- INITIALIZATION ---------------- #
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Embedding Model...")
embedder = SentenceTransformer(EMBEDDING_MODEL)

print("Loading Re-ranking Model...")
reranker = CrossEncoder(RERANK_MODEL)

print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(name=COLLECTION_NAME)

print("Loading BM25 Index...")
try:
    with open(BM25_FILE, "rb") as f:
        bm25 = pickle.load(f)
    with open(BM25_IDS_FILE, "rb") as f:
        bm25_ids = pickle.load(f)
    print("BM25 Loaded Successfully.")
except FileNotFoundError:
    print("Error: BM25 index files not found.")
    bm25 = None
    bm25_ids = []

print("Building Chapter Map for Filtering...")
id_to_chapter = {}
try:
    with open(DATA_FILE, encoding="utf-8") as f:
        data = json.load(f)
        for v in data:
            # Reconstruct ID exactly as indexer.py does
            vid = v.get("verse_id", str(v.get("chapter")) + "-" + str(v.get("verse")))
            id_to_chapter[vid] = v.get("chapter")
    print(f"Mapped {len(id_to_chapter)} verses for filtering.")
except Exception as e:
    print(f"Warning: Could not load {DATA_FILE} for filtering: {e}")

# ---------------- DATA MODELS ---------------- #
class SearchRequest(BaseModel):
    query: str
    limit: int = 5
    chapter: Optional[int] = None # New optional filter

# ---------------- HELPER FUNCTIONS ---------------- #

def perform_hybrid_search(query: str, chapter_filter: Optional[int] = None, initial_k: int = 20, k: int = 60):
    """
    Hybrid Search with Metadata Filtering.
    """
    
    # --- 1. Vector Search (with Chroma Filter) ---
    query_embedding = embedder.encode(query).tolist()
    
    # Prepare Chroma where clause if filter exists
    where_clause = {"chapter": chapter_filter} if chapter_filter else None

    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=initial_k,
        where=where_clause # <--- Chroma handles this natively
    )
    
    vector_ids = vector_results['ids'][0] if vector_results['ids'] else []
    vector_ranks = {vid: i for i, vid in enumerate(vector_ids)}

    # --- 2. BM25 Search (with Manual Filter) ---
    bm25_ranks = {}
    
    if bm25:
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # Get more candidates initially because some might be filtered out
        top_n = initial_k * 3 
        top_indices = np.argsort(bm25_scores)[::-1][:top_n]
        
        rank = 0
        for idx in top_indices:
            if bm25_scores[idx] <= 0:
                continue
                
            vid = bm25_ids[idx]
            
            # --- MANUAL FILTER CHECK ---
            if chapter_filter is not None:
                verse_chapter = id_to_chapter.get(vid)
                if verse_chapter != chapter_filter:
                    continue # Skip if not in requested chapter
            
            bm25_ranks[vid] = rank
            rank += 1
            if rank >= initial_k: # Stop once we have enough valid BM25 results
                break

    # --- 3. Reciprocal Rank Fusion (RRF) ---
    combined_scores = {}
    all_found_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())
    
    for vid in all_found_ids:
        score = 0.0
        if vid in vector_ranks:
            score += 1 / (k + vector_ranks[vid] + 1)
        if vid in bm25_ranks:
            score += 1 / (k + bm25_ranks[vid] + 1)
        combined_scores[vid] = score

    sorted_ids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    return [vid for vid, score in sorted_ids[:initial_k]]

# ---------------- API ENDPOINTS ---------------- #

@app.get("/")
def home():
    return {"message": "Anugamana API is ready with Filtering & Re-ranking."}

@app.post("/search")
def search_verses(request: SearchRequest):
    try:
        # 1. Hybrid Search with Filter
        candidate_ids = perform_hybrid_search(
            query=request.query, 
            chapter_filter=request.chapter, 
            initial_k=20
        )
        
        if not candidate_ids:
            return {"results": []}

        # 2. Fetch Text for Re-ranking
        results_data = collection.get(ids=candidate_ids, include=["documents", "metadatas"])
        
        fetched_map = {}
        for i, vid in enumerate(results_data['ids']):
            fetched_map[vid] = {
                "text": results_data['documents'][i],
                "metadata": results_data['metadatas'][i]
            }

        # 3. Cross-Encoder Re-ranking
        pairs = []
        valid_ids = []
        
        for vid in candidate_ids:
            if vid in fetched_map:
                pairs.append([request.query, fetched_map[vid]["text"]])
                valid_ids.append(vid)

        if pairs:
            cross_scores = reranker.predict(pairs)
            
            scored_results = []
            for i, score in enumerate(cross_scores):
                scored_results.append({
                    "id": valid_ids[i],
                    "score": float(score),
                    "data": fetched_map[valid_ids[i]]
                })
            
            scored_results.sort(key=lambda x: x["score"], reverse=True)
            
            formatted_results = []
            for item in scored_results[:request.limit]:
                formatted_results.append({
                    "text": item["data"]["text"],
                    "metadata": item["data"]["metadata"],
                    "score": item["score"]
                })

            return {"results": formatted_results}
            
        return {"results": []}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))