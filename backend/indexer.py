import json
import pickle
import os
import re
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

# ---------------- CONFIG ---------------- #

DATA_FILE = "gita_full.json"
EMOTION_FILE = "verse_emotions.json"
CHROMA_DIR = "chroma_gita"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
COLLECTION_NAME = "gita_verses"
BM25_FILE = "bm25_index.pkl"
BM25_IDS_FILE = "bm25_ids.pkl"

# ---------------- LOAD DATA ---------------- #

print(f"Loading {DATA_FILE}...")
try:
    with open(DATA_FILE, encoding="utf-8") as f:
        verses = json.load(f)
    print(f"Loaded {len(verses)} verses")
except FileNotFoundError:
    print(f"Error: Could not find {DATA_FILE}. Make sure it is in the same folder.")
    exit()

print(f"Loading {EMOTION_FILE}...")
try:
    with open(EMOTION_FILE, encoding="utf-8") as f:
        emotion_map = json.load(f)
    print(f"Loaded emotions for {len(emotion_map)} verses")
except FileNotFoundError:
    print(f"Warning: {EMOTION_FILE} not found. Indexing without emotion tags.")
    emotion_map = {}

# ---------------- MODEL ---------------- #

print(f"Loading AI Model ({MODEL_NAME})...")
model = SentenceTransformer(MODEL_NAME)

# ---------------- CHROMA SETUP ---------------- #

print("Initializing Database...")
client = chromadb.PersistentClient(path=CHROMA_DIR)

# Delete old collection if it exists
try:
    client.delete_collection(name=COLLECTION_NAME)
    print("Deleted old collection to start fresh.")
except:
    pass

# Create the collection
collection = client.create_collection(name=COLLECTION_NAME)

# ---------------- PREPARE DATA ---------------- #

documents = []
metadatas = []
ids = []

print("Preparing verses...")

for v in verses:
    verse_id = v.get("verse_id", str(v.get("chapter")) + "-" + str(v.get("verse")))
    
    # Get emotion tag
    emotions = emotion_map.get(verse_id, "")

    # Inject emotions into text
    # We use this same text for both Vector embedding and BM25 keywords
    text = (
        f"Context: {emotions}\n"
        f"Translation: {v.get('translation', '')}\n"
        f"Purport: {v.get('purport', '')}\n"
        f"Sanskrit: {v.get('sanskrit', '')}\n"
        f"Synonyms: {v.get('synonyms', '')}"
    ).strip()
    
    documents.append(text)
    
    # Store metadata
    metadatas.append({
        "verse_id": verse_id,
        "emotions": emotions,
        "chapter": v.get("chapter"),
        "verse": v.get("verse"),
        "sanskrit": v.get("sanskrit", ""),
        "transliteration": v.get("transliteration", ""),
        "synonyms": v.get("synonyms", ""),
        "translation": v.get("translation", ""),
        "purport": v.get("purport", "")[:2000]
    })
    
    ids.append(verse_id)

# ---------------- BM25 INDEXING ---------------- #

print("Building BM25 Index...")
# Improved Tokenization: Strip punctuation and keep only alphanumeric words
tokenized_corpus = [re.findall(r'\b\w+\b', doc.lower()) for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

# Save BM25 index and corresponding IDs
with open(BM25_FILE, "wb") as f:
    pickle.dump(bm25, f)
    
with open(BM25_IDS_FILE, "wb") as f:
    pickle.dump(ids, f)
    
print(f"Saved BM25 index to {BM25_FILE} and IDs to {BM25_IDS_FILE}")

# ---------------- VECTOR INDEXING ---------------- #

print(f"Generating embeddings and indexing {len(documents)} verses...")

# Add in batches
batch_size = 50
total_batches = (len(documents) + batch_size - 1) // batch_size

for i in range(total_batches):
    start = i * batch_size
    end = min((i + 1) * batch_size, len(documents))
    
    batch_docs = documents[start:end]
    batch_ids = ids[start:end]
    batch_meta = metadatas[start:end]
    
    batch_embeddings = model.encode(batch_docs).tolist()
    
    collection.add(
        documents=batch_docs,
        embeddings=batch_embeddings,
        metadatas=batch_meta,
        ids=batch_ids
    )
    print(f"Indexed batch {i+1}/{total_batches}")

print("Success! Database updated with Hybrid Search assets.")