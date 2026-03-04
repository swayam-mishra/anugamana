import pickle
import json
import os
import shutil
import re
from typing import Optional, Any
from contextlib import asynccontextmanager
from dotenv import load_dotenv

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

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
# Configurable Base Directory via Environment Variable
BASE_DIR = os.getenv("DB_PATH", ".")

CHROMA_DIR = os.path.join(BASE_DIR, "chroma_gita")  # Dynamic Path
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION_NAME = "gita_verses"
BM25_FILE = os.path.join(BASE_DIR, "bm25_index.pkl")  # Dynamic Path
BM25_IDS_FILE = os.path.join(BASE_DIR, "bm25_ids.pkl")  # Dynamic Path
DATA_FILE = "gita_full.json"

# Security: Structured JSON Logger Setup
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.PrintLoggerFactory(),
)
logger = structlog.get_logger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

try:
    if GEMINI_API_KEY:
        client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("gemini_connected")
    else:
        client = None
        logger.warning("gemini_api_key_missing")
except Exception as e:
    logger.error("gemini_client_init_error", error=str(e))
    client = None

# ---------------- GLOBAL STATE ---------------- #
# Initialize as None. They will be populated in lifespan.
embedder: Optional[SentenceTransformer] = None
reranker: Optional[CrossEncoder] = None
chroma_client: Optional[chromadb.PersistentClient] = None
collection: Optional[Any] = None
bm25: Optional[Any] = None
bm25_ids: list = []
id_to_chapter: dict = {}

# ---------------- LIFESPAN MANAGER ---------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager handles startup and shutdown events.
    Models and DB connections are loaded here to prevent import-time blocking/crashes.
    """
    global embedder, reranker, chroma_client, collection, bm25, bm25_ids, id_to_chapter

    logger.info("startup_begin")

    # --- PERSISTENCE CHECK ---
    # If we are using a custom DB_PATH (like /data) and it's empty,
    # copy the pre-baked DB from the Docker image to the new location.
    if BASE_DIR != "." and not os.path.exists(CHROMA_DIR):
        logger.info("hydrating_persistent_storage", base_dir=BASE_DIR)
        try:
            # Copy ChromaDB
            shutil.copytree("chroma_gita", CHROMA_DIR)
            # Copy BM25 Indices
            if os.path.exists("bm25_index.pkl"):
                shutil.copy("bm25_index.pkl", BM25_FILE)
            if os.path.exists("bm25_ids.pkl"):
                shutil.copy("bm25_ids.pkl", BM25_IDS_FILE)
            logger.info("database_migrated")
        except Exception as e:
            logger.error("database_migration_failed", error=str(e))
            # Fallback to local files if migration fails
    # -------------------------

    # 1. Load Embedding Model
    try:
        logger.info("loading_embedding_model", model=EMBEDDING_MODEL)
        embedder = SentenceTransformer(EMBEDDING_MODEL)
    except Exception as e:
        logger.error("embedding_model_load_failed", error=str(e))

    # 2. Load Re-ranking Model
    try:
        logger.info("loading_reranking_model", model=RERANK_MODEL)
        reranker = CrossEncoder(RERANK_MODEL)
    except Exception as e:
        logger.error("reranker_load_failed", error=str(e))

    # 3. Connect to ChromaDB
    try:
        logger.info("connecting_chromadb", path=CHROMA_DIR)
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        logger.error("chromadb_connection_failed", error=str(e))

    # 4. Load BM25 Index
    logger.info("loading_bm25_index")
    try:
        with open(BM25_FILE, "rb") as f:
            bm25 = pickle.load(f)
        with open(BM25_IDS_FILE, "rb") as f:
            bm25_ids = pickle.load(f)
        logger.info("bm25_loaded")
    except FileNotFoundError:
        logger.warning("bm25_not_found", detail="Keyword search disabled")
        bm25 = None
        bm25_ids = []

    # 5. Build Chapter Map
    logger.info("building_chapter_map")
    try:
        with open(DATA_FILE, encoding="utf-8") as f:
            data = json.load(f)
            for v in data:
                vid = v.get("verse_id", str(v.get("chapter")) + "-" + str(v.get("verse")))
                id_to_chapter[vid] = v.get("chapter")
    except Exception as e:
        logger.warning("chapter_map_load_failed", file=DATA_FILE, error=str(e))

    logger.info("startup_complete")
    
    yield  # Control is yielded to the application
    
    # Shutdown logic (if any cleanup is needed)
    logger.info("shutdown")

# ---------------- INITIALIZATION ---------------- #
# Security: Initialize Rate Limiter
limiter = Limiter(key_func=get_remote_address)

# Pass lifespan to FastAPI
app = FastAPI(lifespan=lifespan)

# Security: Register Limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Security: Secure CORS
origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173")
ALLOWED_ORIGINS = origins_env.split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- DATA MODELS ---------------- #
class SearchRequest(BaseModel):
    # Security: Input Validation (Max length and Range Bounds)
    query: str = Field(..., max_length=500)
    limit: int = Field(default=5, ge=1, le=20) # Max 20 results to prevent OOM
    chapter: Optional[int] = Field(default=None, ge=1, le=18) # Only 18 chapters exist

# ---------------- HELPER FUNCTIONS ---------------- #

def _calculate_hybrid_candidates(query: str, chapter_filter: Optional[int], initial_k: int, k: int = 60):
    if not embedder or not collection:
        raise RuntimeError("Search models are not initialized.")

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
        # Improved Tokenization: Strip punctuation
        tokenized_query = re.findall(r'\b\w+\b', query.lower())
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

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def generate_advice(query: str, verse_text: str):
    """
    Uses Gemini (New SDK) to generate personalized advice.
    Retries up to 3 times with exponential backoff on failure.
    """
    if not client:
        return None

    # Security: Use System Instructions for persona (prevents prompt injection)
    system_instruction = (
        "You are Lord Krishna, a wise and compassionate spiritual guide from the Bhagavad Gita. "
        "You speak with warmth and empathy. You always ground your advice in the verse provided. "
        "Keep your response under 100 words."
    )

    user_prompt = (
        f"The user asked the following question:\n"
        f"```\n{query}\n```\n\n"
        f"The Bhagavad Gita says:\n"
        f"\"{verse_text}\"\n\n"
        f"Explain briefly how this verse answers their question and offer one actionable piece of advice."
    )

    try:
        response = await run_in_threadpool(
            client.models.generate_content,
            model="gemini-2.5-flash",
            contents=user_prompt,
            config={"system_instruction": system_instruction},
        )
        return response.text
    except Exception as e:
        logger.error("llm_error", error=str(e))
        raise  # Re-raise so tenacity can retry

# ---------------- API ENDPOINTS ---------------- #

@app.get("/")
def home():
    status = "Online" if embedder and collection else "Maintenance Mode (Models Loading)"
    return {"message": "Anugamana API: Hybrid Search + Re-Ranking + RAG", "status": status}

@app.post("/search")
@limiter.limit("15/minute") # Security: Rate Limit applied
async def search_verses(request: Request, payload: SearchRequest):
    # Check if models are ready
    if not embedder or not collection or not reranker:
        raise HTTPException(status_code=503, detail="Search services are initializing. Please try again in a few seconds.")

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

        # 2. Fetch Text (Optimized: Non-blocking DB call)
        results_data = await run_in_threadpool(
            collection.get,
            ids=candidate_ids, 
            include=["documents", "metadatas"]
        )
        
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
                 try:
                     rag_advice = await generate_advice(payload.query, top_verse["data"]["text"])
                 except Exception:
                     logger.warning("rag_advice_failed_after_retries")
                     rag_advice = None

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

    except HTTPException:
        raise
    except Exception as e:
        # Security: Prevent Information Leakage
        logger.error("internal_search_error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the search.")