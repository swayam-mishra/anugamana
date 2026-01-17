import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# ---------------- CONFIG ---------------- #

DATA_FILE = "gita_full.json"
CHROMA_DIR = "chroma_gita"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------- LOAD DATA ---------------- #

with open(DATA_FILE, encoding="utf-8") as f:
    verses = json.load(f)

print(f"Loaded {len(verses)} verses")

# ---------------- MODEL ---------------- #

model = SentenceTransformer(MODEL_NAME)

# ---------------- CHROMA SETUP ---------------- #

# client = chromadb.Client(
#     Settings(
#         persist_directory=CHROMA_DIR,
#         anonymized_telemetry=False
#     )
# )

client = chromadb.PersistentClient(path=CHROMA_DIR)

collection = client.get_or_create_collection(
    name="gita_verses"
)
# ---------------- INDEXING ---------------- #

documents = []
metadatas = []
ids = []

for v in verses:
    text = f"{v['translation']}\n\n{v['purport']}".strip()

    documents.append(text)
    metadatas.append({
        "verse_id": v["verse_id"],
        "chapter": v["chapter"],
        "verse": v["verse"]
    })
    ids.append(v["verse_id"])

print("Generating embeddings & storing in Chroma...")

collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

print("Embeddings created and stored successfully")
