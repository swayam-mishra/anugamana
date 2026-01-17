# Anugamana (The Path)

**Anugamana** is an AI-powered spiritual companion designed to bridge the gap between modern emotional dilemmas and ancient wisdom. 

Unlike traditional keyword searches, Anugamana uses **Semantic Search** and **Vector Embeddings**. This allows users to input their state of mind in natural language (e.g., *"I feel overwhelmed by pressure"*), and the system retrieves the most relevant verses from the **Bhagavad Gita**.

## Features

- **Semantic Search:** Understands the *meaning* behind queries, not just keywords.
- **Vector Database:** Uses ChromaDB to store and retrieve high-dimensional text embeddings.
- **FastAPI Backend:** A lightweight, high-performance API serving the data.
- **Data Source:** Scraped from Vedabase (Chapters 1-18).

## Tech Stack

- **Language:** Python 3.11+
- **Framework:** FastAPI
- **AI/NLP:** Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Database:** ChromaDB
- **Frontend:** React + Tailwind CSS (In Progress)

## Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/anugamana.git](https://github.com/YOUR_USERNAME/anugamana.git)
cd anugamana
### 2. Backend Setup

Create a virtual environment and install dependencies:

Bash

```
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate

# Install libraries
pip install -r requirements.txt
```

### 3. Build the Database

The semantic search relies on a vector index. You must generate this locally first.

Bash

```
# 1. (Optional) Scrape data if gita_full.json is missing
python scraper.py

# 2. Generate Embeddings and Index
python indexer.py
```

_Note: This creates a `chroma_gita/` folder locally._

### 4. Run the API

Start the local server:

Bash

```
uvicorn main:app --reload
```

The API will be available at http://127.0.0.1:8000.

Documentation is available at http://127.0.0.1:8000/docs.

## Usage Example

**Endpoint:** `POST /search`

**Request:**

JSON

```
{
  "query": "I am confused about my duty",
  "limit": 3
}
```

Response:

Returns a JSON list of the 3 most relevant verses, including translation, purport, and a relevance score.

---

_Project created as part of the Anugamana initiative._
