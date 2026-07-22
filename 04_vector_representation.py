"""
04_vector_representation.py
----------------------------
Pipeline stage 4: VECTOR REPRESENTATION (embeddings)

Defines the embedding model used to turn text chunks into vectors.

Model choice
------------
`all-MiniLM-L6-v2` (via sentence-transformers) is used because it:
- Runs on CPU at a good speed (~1,000-2,000 sentences/sec on a modern CPU
  in batches), which matters when embedding 50,000+ posts.
- Produces small 384-dimensional vectors, keeping the persisted vector
  store compact (roughly 80 MB for ~50k chunks, vs. 3-4x that for a
  768-dim model) -- important since the store needs to be committed
  alongside the repo for deployment (see 05_create_chroma_store.py).
- Is free and open-source, no API key or network call needed at query
  time, unlike hosted embedding APIs.
"""

from langchain_community.embeddings import SentenceTransformerEmbeddings

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def get_embedding_function() -> SentenceTransformerEmbeddings:
    """Return the embedding function used across the whole pipeline.

    Both the store-building step (05) and the query-time retrieval step
    (06) must use this SAME function -- mismatched embedding models
    between build time and query time silently produce meaningless
    similarity scores.
    """
    return SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)


if __name__ == "__main__":
    embedder = get_embedding_function()
    sample_vector = embedder.embed_query("Which beauty content drives the most engagement?")
    print(f"Embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"Vector dimension: {len(sample_vector)}")
    print(f"First 5 values: {sample_vector[:5]}")
