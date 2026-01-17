import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "chroma_gita"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

client = chromadb.Client(
    Settings(
        persist_directory=CHROMA_DIR,
        anonymized_telemetry=False
    )
)

collection = client.get_collection("gita_verses")
model = SentenceTransformer(MODEL_NAME)

query = "I feel afraid and confused about my future"

query_embedding = model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

for i in range(3):
    print("\n--- RESULT", i + 1, "---")
    print("Verse:", results["metadatas"][0][i]["verse_id"])
    print(results["documents"][0][i][:300], "...")
