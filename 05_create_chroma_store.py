"""
05_create_chroma_store.py
--------------------------
Pipeline stage 5: VECTOR STORE (build + persist a Chroma database)

This is a BUILD SCRIPT, not app code. Run it once, locally:

    python 05_create_chroma_store.py

It embeds every chunk and writes a persisted Chroma database to
./chroma_store/. You then commit that folder to your GitHub repo (or
otherwise ship it with the deployment) so streamlit_app.py can simply
LOAD the store at startup instead of rebuilding it from scratch on every
cold start.

Why this matters at 50,000+ rows
---------------------------------
The original prototype built an in-memory FAISS index inside the
Streamlit app itself, on every session. That does not scale:
  - Embedding ~50k chunks live, inside a web request, is slow (minutes)
    and easily exceeds Streamlit Community Cloud's resource limits/free
    tier timeouts.
  - It repeats the same (expensive) work for every new session, since
    Streamlit Cloud's filesystem/process can be recycled.
  - It risks silent memory errors with no persisted fallback if the
    process is killed mid-build.

Building the store ONCE, offline, in batches, and persisting it to disk
fixes all three problems:
  - Batching (BATCH_SIZE below) keeps memory use flat regardless of
    dataset size -- we never hold all 50k+ embeddings in memory at once.
  - The deployed app just loads an already-built index (see
    06_retrieve_context.py), which takes seconds, not minutes.
  - Progress is printed per batch, so a build that's interrupted can be
    diagnosed instead of failing silently.

If you truly cannot commit a ~50-100 MB chroma_store/ folder to your repo
(e.g. GitHub size limits on a free/student account), set MAX_ROWS below
to cap the sample size -- but prefer the full dataset when possible.
"""

import importlib.util
import os

# Modern, maintained integration -- matches what 06_retrieve_context.py
# uses at query time. The old langchain_community.vectorstores.Chroma
# path is deprecated and, combined with older chromadb releases, can
# trip a recursive-typing hang in the `overrides` package on import.
from langchain_chroma import Chroma

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "chroma_store")

# IMPORTANT: this MUST match COLLECTION_NAME in 06_retrieve_context.py.
# Chroma silently creates a new, empty collection if you open a name
# that doesn't exist yet -- a mismatch here is what caused
# "The Chroma collection exists but contains zero documents."
COLLECTION_NAME = "marketpulse"

# Chunks are embedded and written to disk in batches of this size, so
# memory usage stays flat no matter how large the dataset gets.
BATCH_SIZE = 1000

# Optional cap for constrained environments. None = use the full dataset
# (recommended; the dataset has 50,000+ rows and the pipeline is built to
# handle that via batching above).
MAX_ROWS = None


def _import(module_filename, module_name):
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(BASE_DIR, module_filename)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_preprocessing = _import("02_preprocessing.py", "preprocessing_mod")
_chunking = _import("03_chunking.py", "chunking_mod")
_vectors = _import("04_vector_representation.py", "vectors_mod")

load_and_preprocess = _preprocessing.load_and_preprocess
build_documents = _chunking.build_documents
chunk_documents = _chunking.chunk_documents
get_embedding_function = _vectors.get_embedding_function


def build_chroma_store(
    persist_directory: str = PERSIST_DIRECTORY,
    batch_size: int = BATCH_SIZE,
    max_rows: int = MAX_ROWS,
) -> Chroma:
    """Build (or rebuild) a persisted Chroma store from the dataset, in batches."""
    df = load_and_preprocess()
    if max_rows:
        df = df.sample(n=min(max_rows, len(df)), random_state=42).reset_index(drop=True)

    documents = build_documents(df)
    chunks = chunk_documents(documents)
    print(f"Embedding {len(chunks):,} chunks from {len(documents):,} posts "
          f"in batches of {batch_size}...")

    embedding_function = get_embedding_function()
    vectorstore = None

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embedding_function,
                persist_directory=persist_directory,
                collection_name=COLLECTION_NAME,
            )
        else:
            vectorstore.add_documents(batch)
        done = min(i + batch_size, len(chunks))
        print(f"  ...{done:,}/{len(chunks):,} chunks embedded")

    # langchain_chroma persists automatically as documents are added --
    # there's no manual .persist() call anymore (that method was removed
    # after Chroma 0.4.x; calling it on the old langchain_community
    # wrapper is what produced the DeprecationWarning you saw).
    print(f"Done. Persisted Chroma store at: {persist_directory}")
    return vectorstore


def store_exists(persist_directory: str = PERSIST_DIRECTORY) -> bool:
    return os.path.isdir(persist_directory) and len(os.listdir(persist_directory)) > 0


if __name__ == "__main__":
    if store_exists():
        print(f"A Chroma store already exists at {PERSIST_DIRECTORY}.")
        answer = input("Rebuild it from scratch? [y/N]: ").strip().lower()
        if answer != "y":
            print("Skipping rebuild.")
            raise SystemExit(0)
    build_chroma_store()
