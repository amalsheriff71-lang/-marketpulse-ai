"""
06_retrieve_context.py
------------------------
Pipeline stage 6: CONTEXT RETRIEVAL

Loads the persisted Chroma store and retrieves the most relevant
social media content for a user's question.

V2 improvements:
- Keeps retrieval lightweight.
- Includes document metadata in the context sent to the LLM.
- Keeps source numbering consistent between the LLM and Streamlit UI.
"""

import importlib.util
import os

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _import(module_filename, module_name):
    spec = importlib.util.spec_from_file_location(
        module_name,
        os.path.join(BASE_DIR, module_filename),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_vectors = _import(
    "04_vector_representation.py",
    "vectors_mod",
)

_store = _import(
    "05_create_chroma_store.py",
    "store_mod",
)


get_embedding_function = _vectors.get_embedding_function

PERSIST_DIRECTORY = _store.PERSIST_DIRECTORY
COLLECTION_NAME = _store.COLLECTION_NAME
store_exists = _store.store_exists


def load_vectorstore(
    persist_directory: str = PERSIST_DIRECTORY,
) -> Chroma:
    """
    Load the persisted Chroma vector store.

    The embedding function must match the one used when
    the store was originally created.
    """

    if not store_exists(persist_directory):
        raise FileNotFoundError(
            f"No Chroma store found at '{persist_directory}'. "
            "Run `python 05_create_chroma_store.py` first."
        )

    return Chroma(
        persist_directory=persist_directory,
        embedding_function=get_embedding_function(),
        collection_name=COLLECTION_NAME,
    )


def retrieve_context(
    vectorstore: Chroma,
    query: str,
    k: int = 5,
) -> list[Document]:
    """
    Retrieve the top-k most relevant documents.
    """

    if not query or not query.strip():
        return []

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )

    return retriever.invoke(query)


def format_context(
    docs: list[Document],
) -> str:
    """
    Format retrieved documents for the LLM.

    Each source includes:
    - Platform
    - Category
    - Content ID
    - Original content

    Source numbering is shared with the UI.
    """

    if not docs:
        return "No relevant sources were retrieved."

    parts = []

    for i, doc in enumerate(docs, start=1):

        metadata = doc.metadata or {}

        platform = metadata.get(
            "platform",
            "Unknown",
        )

        category = metadata.get(
            "category",
            "Unknown",
        )

        content_id = metadata.get(
            "content_id",
            "Unknown",
        )

        source_block = (
            f"[Source {i}]\n"
            f"Platform: {platform}\n"
            f"Category: {category}\n"
            f"Content ID: {content_id}\n\n"
            f"Content:\n"
            f"{doc.page_content}"
        )

        parts.append(source_block)

    return "\n\n".join(parts)


if __name__ == "__main__":

    vectorstore = load_vectorstore()

    query = (
        "Which content themes drive the most "
        "engagement in the beauty category?"
    )

    docs = retrieve_context(
        vectorstore,
        query,
        k=5,
    )

    print(
        f"Query: {query}\n"
    )

    print(
        format_context(docs)
    )