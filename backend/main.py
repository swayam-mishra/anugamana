import pickle
import json
import os
import logging
from typing import Optional
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, CrossEncoder 
import chromadb
import numpy as np
from google import genai 

# Security: Rate Limiting Imports
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ---------------- CONFIGURATION ---------------- #
CHROMA_DIR = "chroma_gita"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION_NAME = "gita_verses"
BM25_FILE = "bm25_index.pkl"
BM25_IDS_FILE = "bm25_ids.pkl"
DATA_FILE = "gita_full.json"

# Security: Secure Logger Setup
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

try:
    if GEMINI_API_KEY:
        client = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ Gemini AI Connected (New SDK)")
    else:
        client = None
        print("⚠️ Warning: GEMINI_API_KEY not found.")
except Exception as e:
    print(f"⚠️ Client Init Error: {e}")
    client = None

# ---------------- INITIALIZATION ---------------- #
# Security: Initialize Rate Limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI()

# Security: Register Limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Security: Secure CORS
# Fetch allowed origins from env or default to localhost
origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173")
ALLOWED_ORIGINS = origins_env.split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Explicit list, no "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Embedding Model...")
embedder = SentenceTransformer(EMBEDDING_MODEL)

print("Loading Re-ranking Model...")
reranker = CrossEncoder(RERANK_MODEL)

print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(name=COLLECTION_NAME)

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

print("Building Chapter Map...")
id_to_chapter = {}
try:
    with open(DATA_FILE, encoding="utf-8") as f:
        data = json.load(f)
        for v in data:
            vid = v.get("verse_id", str(v.get("chapter")) + "-" + str(v.get("verse")))
            id_to_chapter[vid] = v.get("chapter")
except Exception as e:
    print(f"Warning: Could not load {DATA_FILE}: {e}")

# ---------------- DATA MODELS ---------------- #
class SearchRequest(BaseModel):
    # Security: Input Validation (Max length)
    query: str = Field(..., max_length=500)
    limit: int = 5
    chapter: Optional[int] = None

# ---------------- HELPER FUNCTIONS ---------------- #

def _calculate_hybrid_candidates(query: str, chapter_filter: Optional[int], initial_k: int, k: int = 60):
    # 1. Vector Search
    query_embedding = embedder.encode(query).tolist()
    where_clause = {"chapter": chapter_filter} if chapter_filter else None
    
    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=initial_k,
        where=where_clause
    )
    vector_ids = vector_results['ids'][0] if vector_results['ids'] else []
    vector_ranks = {vid: i for i, vid in enumerate(vector_ids)}

    # 2. BM25 Search
    bm25_ranks = {}
    if bm25:
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        top_n = initial_k * 3 
        top_indices = np.argsort(bm25_scores)[::-1][:top_n]
        
        rank = 0
        for idx in top_indices:
            if bm25_scores[idx] <= 0: continue
            vid = bm25_ids[idx]
            if chapter_filter is not None:
                if id_to_chapter.get(vid) != chapter_filter: continue
            
            bm25_ranks[vid] = rank
            rank += 1
            if rank >= initial_k: break

    # 3. RRF Fusion
    combined_scores = {}
    all_found_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())
    
    for vid in all_found_ids:
        score = 0.0
        if vid in vector_ranks: score += 1 / (k + vector_ranks[vid] + 1)
        if vid in bm25_ranks: score += 1 / (k + bm25_ranks[vid] + 1)
        combined_scores[vid] = score

    sorted_ids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    return [vid for vid, score in sorted_ids[:initial_k]]

async def generate_advice(query: str, verse_text: str):
    """
    Uses Gemini (New SDK) to generate personalized advice.
    """
    if not client:
        return None
        
    # Security: Mitigate Prompt Injection with delimiters
    prompt = f"""
    You are a wise spiritual guide. 
    The user asked the question enclosed in triple backticks:
    ```
    {query}
    ```

    The Bhagavad Gita says:
    "{verse_text}"
    
    Explain briefly how this verse answers their question and offer one actionable piece of advice.
    Keep it warm, empathetic, and under 100 words.
    """
    
    try:
        response = await run_in_threadpool(
            client.models.generate_content,
            model="gemini-2.5-flash", 
            contents=prompt
        )
        return response.text
    except Exception as e:
        logger.error(f"LLM Error: {e}")
        return None

# ---------------- API ENDPOINTS ---------------- #

@app.get("/")
def home():
    return {"message": "Anugamana API: Hybrid Search + Re-Ranking + RAG"}

@app.post("/search")
@limiter.limit("15/minute") # Security: Rate Limit applied
async def search_verses(request: Request, payload: SearchRequest):
    try:
        # 1. Hybrid Search
        candidate_ids = await run_in_threadpool(
            _calculate_hybrid_candidates, 
            payload.query, 
            payload.chapter, 
            20
        )
        
        if not candidate_ids:
            return {"results": []}

        # 2. Fetch Text
        results_data = collection.get(ids=candidate_ids, include=["documents", "metadatas"])
        
        fetched_map = {}
        for i, vid in enumerate(results_data['ids']):
            fetched_map[vid] = {
                "text": results_data['documents'][i],
                "metadata": results_data['metadatas'][i]
            }

        # 3. Prepare for Re-ranking
        pairs = []
        valid_ids = []
        for vid in candidate_ids:
            if vid in fetched_map:
                pairs.append([payload.query, fetched_map[vid]["text"]])
                valid_ids.append(vid)

        if pairs:
            # 4. Re-ranking
            cross_scores = await run_in_threadpool(reranker.predict, pairs)
            
            scored_results = []
            for i, score in enumerate(cross_scores):
                scored_results.append({
                    "id": valid_ids[i],
                    "score": float(score),
                    "data": fetched_map[valid_ids[i]]
                })
            
            scored_results.sort(key=lambda x: x["score"], reverse=True)
            
            # 5. Format & RAG
            final_results = []
            top_results = scored_results[:payload.limit]
            
            rag_advice = None
            if top_results and payload.limit == 1:
                 top_verse = top_results[0]
                 rag_advice = await generate_advice(payload.query, top_verse["data"]["text"])

            for item in top_results:
                res = {
                    "text": item["data"]["text"],
                    "metadata": item["data"]["metadata"],
                    "score": item["score"]
                }
                if rag_advice and item == top_results[0]:
                    res["metadata"]["ai_advice"] = rag_advice
                    
                final_results.append(res)

            return {"results": final_results}
            
        return {"results": []}

    except Exception as e:
        # Security: Prevent Information Leakage
        logger.error(f"Internal Search Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the search.")