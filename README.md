# Social Media Marketing Intelligence Assistant — RAG Pipeline

## Setup
```bash
pip install -r requirements.txt
```

## Build the vector store (run once, locally, before deploying)
```bash
python 05_create_chroma_store.py
```
This embeds all 50,000+ posts in batches and writes a persisted index to
`chroma_store/`. Commit that folder to your repo — the deployed app only
*loads* it, it never rebuilds it live.

## Run locally
```bash
export GROQ_API_KEY=your_key_here   # get one free at console.groq.com
streamlit run streamlit_app.py
```

## Deploy to Streamlit Cloud
1. Push this project (including `chroma_store/`, excluding `.env` and any
   real `secrets.toml`) to GitHub.
2. Deploy on Streamlit Community Cloud.
3. In **Manage app → Settings → Secrets**, paste:
   ```toml
   GROQ_API_KEY = "your_key_here"
   ```

## Pipeline
```
01_documents.py            raw CSV ingestion
02_preprocessing.py        cleaning, normalization (vectorized for 50k+ rows)
03_chunking.py              build Documents + split into chunks
04_vector_representation.py embedding model (all-MiniLM-L6-v2)
05_create_chroma_store.py  BUILD SCRIPT: batch-embed + persist Chroma store
06_retrieve_context.py     load persisted store, similarity search
07_prompting.py            prompt template + RAG chain, with [Source N] citations
streamlit_app.py           UI: loads the persisted store, never rebuilds live
```
