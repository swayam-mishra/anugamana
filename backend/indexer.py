import json
import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("anugamana")

print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Loading Gita data...")
with open("gita_full.json", "r", encoding="utf-8") as f:
    verses = json.load(f)

print("Indexing verses to Pinecone...")
batch_size = 100
vectors = []

for i, verse in enumerate(verses):
    # Combine relevant text for embedding (matching your previous strategy)
    text_to_embed = f"Chapter {verse['chapter']}, Verse {verse['verse']}: {verse['translation']} {verse.get('purport', '')}"
    embedding = model.encode(text_to_embed).tolist()
    
    # Pinecone requires string IDs
    vector_id = f"c{verse['chapter']}v{verse['verse']}"
    
    metadata = {
        "chapter": verse['chapter'],
        "verse": verse['verse'],
        "text": verse.get('sanskrit', ''),
        "translation": verse.get('translation', ''),
        "meaning": verse.get('purport', '')
    }
    
    vectors.append({
        "id": vector_id,
        "values": embedding,
        "metadata": metadata
    })
    
    # Upsert in batches of 100 to respect network limits
    if len(vectors) >= batch_size:
        index.upsert(vectors=vectors)
        vectors = []
        print(f"Upserted batch up to verse {i+1}")

# Upsert any remaining vectors
if vectors:
    index.upsert(vectors=vectors)
    print("Final batch upserted.")

print("Indexing complete! You can verify the vector count in the Pinecone dashboard.")