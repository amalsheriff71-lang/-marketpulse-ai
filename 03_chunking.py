"""
03_chunking.py
---------------
Pipeline stage 3: CHUNKING

Turns each cleaned row into a LangChain Document (one social media post =
one document), then splits long documents into smaller chunks for
embedding.

Dataset size note
------------------
With 50,000+ rows, building Documents with `for _, row in df.iterrows()`
is a real bottleneck (iterrows() is notoriously slow because it boxes each
row into a Series). We use `itertuples()` instead, which is 5-10x faster
and comfortably handles 50k+ rows in a couple of seconds.

Each Document keeps a stable `content_id` (and other citation fields) in
its metadata. This is what lets the final answer cite *which* post it
pulled information from -- without an id, "cite your sources" is not
actually possible.
"""

import importlib.util
import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _import(module_filename, module_name):
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(BASE_DIR, module_filename)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_preprocessing = _import("02_preprocessing.py", "preprocessing_mod")
load_and_preprocess = _preprocessing.load_and_preprocess

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def build_documents(df) -> list[Document]:
    """Convert each cleaned row into a Document with citation-ready metadata.

    Uses itertuples() rather than iterrows() for speed at 50k+ rows.
    """
    documents = []
    for row in df.itertuples(index=False):
        content = (
            f"Platform: {row.platform}\n"
            f"Category: {row.content_category}\n"
            f"Description: {row.content_description}\n"
            f"Engagement: {row.likes} Likes, {row.comments_count} Comments, {row.shares} Shares, "
            f"{row.views} Views\n"
            f"Status: {row.is_sponsored}"
            + (f" ({row.sponsor_name})" if getattr(row, "sponsor_name", "") else "")
            + "\n"
            f"Followers: {row.follower_count}\n"
            f"Top Comment: {row.comments_text}"
        )

        metadata = {
            "content_id": getattr(row, "content_id", None),
            "platform": row.platform,
            "category": row.content_category,
            "likes": int(row.likes),
            "is_sponsored": row.is_sponsored,
            "sponsor_name": getattr(row, "sponsor_name", "") or "",
            "content_url": getattr(row, "content_url", "") or "",
            "post_date": str(getattr(row, "post_date", "")),
        }
        documents.append(Document(page_content=content, metadata=metadata))
    return documents


def chunk_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into smaller chunks, preserving metadata on each chunk."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def load_chunks() -> list[Document]:
    """Convenience: run the full documents+preprocessing+chunking pipeline."""
    df = load_and_preprocess()
    documents = build_documents(df)
    return chunk_documents(documents)


if __name__ == "__main__":
    df = load_and_preprocess()
    docs = build_documents(df)
    chunks = chunk_documents(docs)
    print(f"Posts (documents): {len(docs):,}")
    print(f"Chunks after split: {len(chunks):,}")
    print(f"\nSample document:\n{'-' * 40}\n{docs[0].page_content}")
    print(f"\nSample metadata:\n{docs[0].metadata}")
