import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# ---------------- CONFIG ---------------- #

DATA_FILE = "gita_full.json"
CHROMA_DIR = "chroma_gita"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "gita_verses"

# ---------------- LOAD DATA ---------------- #

print(f"Loading {DATA_FILE}...")
try:
    with open(DATA_FILE, encoding="utf-8") as f:
        verses = json.load(f)
    print(f"Loaded {len(verses)} verses")
except FileNotFoundError:
    print(f"Error: Could not find {DATA_FILE}. Make sure it is in the same folder.")
    exit()

# ---------------- MODEL ---------------- #

print("Loading AI Model...")
model = SentenceTransformer(MODEL_NAME)

# ---------------- CHROMA SETUP ---------------- #

print("Initializing Database...")
client = chromadb.PersistentClient(path=CHROMA_DIR)

# Delete the old collection to ensure we don't have mixed data
try:
    client.delete_collection(name=COLLECTION_NAME)
    print("Deleted old collection to start fresh.")
except:
    pass

collection = client.create_collection(name=COLLECTION_NAME)

# ---------------- INDEXING ---------------- #

documents = []
metadatas = []
ids = []

print("Processing verses...")

for v in verses:
    # We search against both Translation AND Purport for better accuracy
    text = f"{v.get('translation', '')}\n\n{v.get('purport', '')}".strip()
    
    documents.append(text)
    
    # THIS IS THE FIX: We now store ALL the data you want to display
    metadatas.append({
        "verse_id": v.get("verse_id", str(v.get("chapter")) + "-" + str(v.get("verse"))),
        "chapter": v.get("chapter"),
        "verse": v.get("verse"),
        "sanskrit": v.get("sanskrit", ""),
        "transliteration": v.get("transliteration", ""),
        "synonyms": v.get("synonyms", ""),
        "translation": v.get("translation", ""),
        "purport": v.get("purport", "")[:2000] # Limit purport size to prevent DB errors on huge texts
    })
    
    ids.append(v.get("verse_id", f"{v['chapter']}_{v['verse']}"))

print("Generating embeddings & storing in Chroma (this may take a moment)...")

# Add in batches to be safe
batch_size = 50
total_batches = (len(documents) + batch_size - 1) // batch_size

for i in range(total_batches):
    start = i * batch_size
    end = min((i + 1) * batch_size, len(documents))
    
    collection.add(
        documents=documents[start:end],
        metadatas=metadatas[start:end],
        ids=ids[start:end]
    )
    print(f"Indexed batch {i+1}/{total_batches}")

print("Success! Database updated with full metadata.")