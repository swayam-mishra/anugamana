# Anugamana (The Path)

**Anugamana** is an AI-powered spiritual companion designed to bridge the gap between modern emotional dilemmas and ancient wisdom.

Anugamana goes beyond simple keyword matching by utilizing a **Hybrid Search** architecture (Vector + BM25) and **Re-Ranking** to find the precise wisdom you need. It then employs **Generative AI (Google Gemini)** to interpret that wisdom and offer personalized, actionable advice for your specific situation.

## Features

- **Hybrid Search Engine:** Combines the semantic understanding of Vector Search (ChromaDB) with the precision of Keyword Search (BM25) using Reciprocal Rank Fusion (RRF).
- **AI-Powered Insight:** Uses **Gemini 2.5 Flash** to analyze the retrieved verses and generate empathetic, context-aware advice (RAG).
- **Smart Re-Ranking:** Utilizes a Cross-Encoder model to deeply analyze and re-rank search results for maximum relevance.
- **Interactive Web UI:** A clean, spiritual interface designed for reflection and clarity.
- **High-Performance Backend:** Built with FastAPI for speed and efficiency.

## Tech Stack

### Backend 
- **Language:** Python 3.11+
- **Framework:** FastAPI
- **LLM/AI:** Google Gemini 2.5 Flash (`google-genai`)
- **Embeddings:** Sentence-Transformers (`all-mpnet-base-v2`)
- **Re-Ranking:** Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)
- **Search:** ChromaDB (Vector) + BM25 (Keyword)
- **Scraping:** BeautifulSoup4
### Frontend
- **Framework:** React (Vite)
- **Styling:** Tailwind CSS v4
- **Icons:** Lucide React
- **Animations:** Motion
- **Networking:** Axios

## Installation & Setup

### 1. Clone the Repository

```bash
git clone [https://github.com/YOUR_USERNAME/anugamana.git](https://github.com/YOUR_USERNAME/anugamana.git)
cd anugamana
````
### 2. Backend Setup

Open a terminal and navigate to the backend folder:

``` Bash
cd backend

# Create virtual environment
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Environment Configuration:**

You need a Google Gemini API key for the generative features.

1. Create a `.env` file (or set it in your environment).
2. Add your key:

``` Bash
export GEMINI_API_KEY="your_actual_api_key_here"
```

_(Windows: `set GEMINI_API_KEY=your_actual_api_key_here`)_

**Initialize the Knowledge Base:** You must generate the vector index and BM25 index locally before the app can work.

``` Bash
# 1. (Optional) Scrape data if gita_full.json is missing
python scraper.py

# 2. Generate Embeddings and Indexes (Creates /chroma_gita and .pkl files)
# Note: This downloads models and may take a moment.
python indexer.py
```

**Start the API Server:**
``` Bash
uvicorn main:app --reload
```

### 3. Frontend Setup

Open a **new** terminal window (keep the backend running) and navigate to the frontend folder:

``` Bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

## Usage

1. Ensure both the **Backend** and **Frontend** are running.
2. Open your browser to the frontend URL.
3. Type your dilemma or question into the input field (e.g., _"I feel overwhelmed by pressure"_).
4. **Anugamana will:**
    - Perform a hybrid search to find the most relevant verses.
    - Re-rank them for precision.
    - Use Gemini AI to explain _how_ that verse applies to your specific problem.
## API Documentation

The backend provides automatic interactive documentation. Once the server is running, visit:

- **Swagger UI:** `http://127.0.0.1:8000/docs`
- **ReDoc:** `http://127.0.0.1:8000/redoc`

---

_Project created as part of the Almost Perfect initiative._