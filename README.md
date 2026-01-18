# Anugamana (The Path)

**Anugamana** is an AI-powered spiritual companion designed to bridge the gap between modern emotional dilemmas and ancient wisdom.

Unlike traditional keyword searches, Anugamana uses **Semantic Search** and **Vector Embeddings**. This allows users to input their state of mind in natural language (e.g., *"I feel overwhelmed by pressure"*), and the system retrieves the most relevant verses from the **Bhagavad Gita** through a beautiful, purpose-built web interface.

## Features

- **Semantic Search:** Understands the *meaning* behind queries, not just keywords.
- **Interactive Web UI:** A clean, spiritual interface designed for reflection and clarity.
- **Vector Database:** Uses ChromaDB to store and retrieve high-dimensional text embeddings.
- **FastAPI Backend:** A lightweight, high-performance API serving the wisdom.
- **Data Source:** Curated wisdom from the Bhagavad Gita (Chapters 1-18).

## Tech Stack

### Backend 
- **Language:** Python 3.11+
- **Framework:** FastAPI
- **AI/NLP:** Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Database:** ChromaDB

### Frontend
- **Framework:** React (Vite)
- **Styling:** Tailwind CSS (Custom Spiritual Theme)
- **Networking:** Axios
- **Animation:** Framer Motion

## Installation & Setup

### 1. Clone the Repository

```bash
git clone [https://github.com/YOUR_USERNAME/anugamana.git](https://github.com/YOUR_USERNAME/anugamana.git)
cd anugamana
````
### 2. Backend Setup (The Brain)

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

Initialize the Brain:
You must generate the vector index locally before the app can work.
``` Bash
# 1. (Optional) Scrape data if gita_full.json is missing
python scraper.py

# 2. Generate Embeddings and Index (Creates /chroma_gita folder)
python indexer.py
```
**Start the API Server:**
``` Bash
uvicorn main:app --reload
```

_The Backend is now running at_

---
### 3. Frontend Setup (

Open a **new** terminal window (keep the backend running) and navigate to the frontend folder:

``` Bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

_The Web Interface is now running_
## Usage

1. Ensure both the **Backend** (port 8000) and **Frontend** (port 5173) are running.
2. Open your browser to `http://localhost:XXXX`.
3. Type your dilemma or question into the input field (e.g., _"I am confused about my duty"_).
4. Anugamana will analyze your intent and display the most relevant verse with its translation and purport.

## API Documentation

If you want to use the API directly or integrate it into other apps (like Mobile), full documentation is available at:

---

_Project created as part of the Anugamana initiative._