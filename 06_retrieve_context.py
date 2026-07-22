"""
06_retrieve_context.py
------------------------
Pipeline stage 6: CONTEXT RETRIEVAL + ANALYTICS

MarketPulse AI

Responsibilities:
1. Load the persisted Chroma vector store.
2. Retrieve the most relevant documents.
3. Extract structured marketing metrics from retrieved documents.
4. Calculate engagement scores using Python.
5. Rank content objectively before sending data to the LLM.
6. Provide a structured analytical context for the LLM.
7. Keep source numbering consistent between LLM and Streamlit UI.

IMPORTANT:
The LLM should interpret the calculated evidence.
Python is responsible for deterministic calculations and rankings.
"""


import importlib.util
import os
import re
from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# DYNAMIC IMPORT HELPER
# ============================================================

def _import(
    module_filename: str,
    module_name: str,
):
    """
    Dynamically import a Python module from the same directory.
    """

    module_path = os.path.join(
        BASE_DIR,
        module_filename,
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        module_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load module: {module_filename}"
        )

    mod = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        mod
    )

    return mod


# ============================================================
# LOAD VECTOR / STORE MODULES
# ============================================================

_vectors = _import(
    "04_vector_representation.py",
    "vectors_mod",
)

_store = _import(
    "05_create_chroma_store.py",
    "store_mod",
)


# ============================================================
# IMPORT REQUIRED FUNCTIONS / CONSTANTS
# ============================================================

get_embedding_function = (
    _vectors.get_embedding_function
)

PERSIST_DIRECTORY = (
    _store.PERSIST_DIRECTORY
)

COLLECTION_NAME = (
    _store.COLLECTION_NAME
)

store_exists = (
    _store.store_exists
)


# ============================================================
# LOAD VECTOR STORE
# ============================================================

def load_vectorstore(
    persist_directory: str = PERSIST_DIRECTORY,
) -> Chroma:
    """
    Load the persisted Chroma vector store.

    The embedding function must match the embedding
    function used when the store was created.
    """

    if not store_exists(
        persist_directory
    ):
        raise FileNotFoundError(
            f"No Chroma store found at "
            f"'{persist_directory}'. "
            f"Run `python 05_create_chroma_store.py` first."
        )

    return Chroma(
        persist_directory=persist_directory,
        embedding_function=get_embedding_function(),
        collection_name=COLLECTION_NAME,
    )


# ============================================================
# RETRIEVE CONTEXT
# ============================================================

def retrieve_context(
    vectorstore: Chroma,
    query: str,
    k: int = 5,
) -> List[Document]:
    """
    Retrieve the top-k most relevant documents.
    """

    if not query or not query.strip():
        return []

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": k
        }
    )

    return retriever.invoke(
        query
    )


# ============================================================
# SAFE NUMBER PARSER
# ============================================================

def _to_number(
    value: Any,
) -> Optional[float]:
    """
    Convert a value into a number safely.

    Supports values such as:
    - 1551
    - "1551"
    - "1,551"
    - "1551 Likes"
    """

    if value is None:
        return None

    if isinstance(
        value,
        (int, float),
    ):
        return float(value)

    text = str(
        value
    ).replace(
        ",",
        "",
    )

    match = re.search(
        r"-?\d+(?:\.\d+)?",
        text,
    )

    if not match:
        return None

    try:
        return float(
            match.group(0)
        )
    except (
        ValueError,
        TypeError,
    ):
        return None


# ============================================================
# EXTRACT METRIC FROM TEXT
# ============================================================

def _extract_metric(
    text: str,
    metric_name: str,
) -> Optional[float]:
    """
    Extract a metric from document text.

    Example:
    Engagement: 1551 Likes, 199 Comments,
    310 Shares, 10106 Views
    """

    if not text:
        return None

    pattern = (
        rf"{metric_name}"
        rf"\s*[:=]?\s*"
        rf"([\d,]+(?:\.\d+)?)"
    )

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    return _to_number(
        match.group(1)
    )


# ============================================================
# EXTRACT METADATA
# ============================================================

def _get_metadata_value(
    metadata: Dict[str, Any],
    key: str,
    default: str = "Unknown",
) -> str:
    """
    Safely retrieve metadata.
    """

    value = metadata.get(
        key
    )

    if value is None:
        return default

    return str(
        value
    )


# ============================================================
# ANALYZE DOCUMENT
# ============================================================

def analyze_document(
    doc: Document,
    source_number: int,
) -> Dict[str, Any]:
    """
    Convert a retrieved Document into structured analytics.

    Engagement Score is calculated deterministically:

        Engagement Score =
            Likes
            + Comments
            + Shares

    Views are intentionally NOT included in the score.

    This prevents the LLM from deciding rankings itself.
    """

    metadata = (
        doc.metadata or {}
    )

    content = (
        doc.page_content or ""
    )

    platform = _get_metadata_value(
        metadata,
        "platform",
    )

    category = _get_metadata_value(
        metadata,
        "category",
    )

    content_id = _get_metadata_value(
        metadata,
        "content_id",
    )

    likes = _extract_metric(
        content,
        "Likes",
    )

    comments = _extract_metric(
        content,
        "Comments",
    )

    shares = _extract_metric(
        content,
        "Shares",
    )

    views = _extract_metric(
        content,
        "Views",
    )

    followers = _extract_metric(
        content,
        "Followers",
    )

    sponsored = "Unknown"

    sponsored_match = re.search(
        r"Status\s*:\s*(.*?)(?:Followers|Top Comment|$)",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if sponsored_match:
        sponsored = (
            sponsored_match.group(1)
            .strip()
            .replace(
                "\n",
                " ",
            )
        )

    # --------------------------------------------------------
    # CALCULATE ENGAGEMENT SCORE
    # --------------------------------------------------------

    metric_values = [
        value
        for value in [
            likes,
            comments,
            shares,
        ]
        if value is not None
    ]

    engagement_score = (
        sum(
            metric_values
        )
        if metric_values
        else None
    )

    return {
        "source_number": source_number,
        "source_label": (
            f"[Source {source_number}]"
        ),
        "platform": platform,
        "category": category,
        "content_id": content_id,
        "likes": likes,
        "comments": comments,
        "shares": shares,
        "views": views,
        "followers": followers,
        "sponsored_status": sponsored,
        "engagement_score": engagement_score,
        "original_content": content,
    }


# ============================================================
# ANALYZE RETRIEVED DOCUMENTS
# ============================================================

def analyze_retrieved_documents(
    docs: List[Document],
) -> List[Dict[str, Any]]:
    """
    Analyze all retrieved documents.

    Returns structured records containing:
    - Source
    - Platform
    - Category
    - Content ID
    - Likes
    - Comments
    - Shares
    - Views
    - Followers
    - Engagement Score
    """

    analyzed = []

    for index, doc in enumerate(
        docs,
        start=1,
    ):
        record = analyze_document(
            doc,
            index,
        )

        analyzed.append(
            record
        )

    return analyzed


# ============================================================
# RANK BY ENGAGEMENT
# ============================================================

def rank_by_engagement(
    analyzed_docs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Rank content by deterministic Engagement Score.

    Engagement Score =
        Likes + Comments + Shares

    Items without a calculable score
    are placed at the bottom.
    """

    return sorted(
        analyzed_docs,
        key=lambda item: (
            item.get(
                "engagement_score"
            )
            if item.get(
                "engagement_score"
            ) is not None
            else -1
        ),
        reverse=True,
    )


# ============================================================
# FIND TOP CONTENT
# ============================================================

def get_top_content(
    analyzed_docs: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Return the highest-ranked content
    based on Engagement Score.
    """

    ranked = rank_by_engagement(
        analyzed_docs
    )

    if not ranked:
        return None

    valid_items = [
        item
        for item in ranked
        if item.get(
            "engagement_score"
        ) is not None
    ]

    if not valid_items:
        return None

    return valid_items[0]


# ============================================================
# FORMAT ANALYTICS FOR LLM
# ============================================================

def format_analytics(
    analyzed_docs: List[Dict[str, Any]],
) -> str:
    """
    Format deterministic Python analytics
    into a compact context for the LLM.

    IMPORTANT:
    The LLM receives the calculated ranking
    and should interpret it rather than recalculate it.
    """

    if not analyzed_docs:
        return (
            "No structured analytics were available."
        )

    ranked = rank_by_engagement(
        analyzed_docs
    )

    parts = []

    parts.append(
        "PYTHON-CALCULATED ANALYTICS\n"
    )

    parts.append(
        "Engagement Score Formula:\n"
        "Likes + Comments + Shares\n"
    )

    parts.append(
        "Ranking is calculated by Python. "
        "The LLM must not recalculate or invent rankings.\n"
    )

    parts.append(
        "RANKED CONTENT:\n"
    )

    for rank, item in enumerate(
        ranked,
        start=1,
    ):

        engagement_score = (
            item.get(
                "engagement_score"
            )
        )

        score_text = (
            str(
                int(
                    engagement_score
                )
            )
            if engagement_score is not None
            else "Not Available"
        )

        likes = item.get(
            "likes"
        )

        comments = item.get(
            "comments"
        )

        shares = item.get(
            "shares"
        )

        views = item.get(
            "views"
        )

        followers = item.get(
            "followers"
        )

        parts.append(
            f"""
Rank: {rank}
Source: {item.get("source_label")}
Content ID: {item.get("content_id")}
Platform: {item.get("platform")}
Category: {item.get("category")}
Likes: {likes if likes is not None else "N/A"}
Comments: {comments if comments is not None else "N/A"}
Shares: {shares if shares is not None else "N/A"}
Views: {views if views is not None else "N/A"}
Followers: {followers if followers is not None else "N/A"}
Engagement Score: {score_text}
Sponsored Status: {item.get("sponsored_status")}
"""
        )

    top_content = get_top_content(
        analyzed_docs
    )

    if top_content:

        parts.append(
            "\nPYTHON TOP-PERFORMING CONTENT:\n"
        )

        parts.append(
            f"""
Content ID:
{top_content.get("content_id")}

Source:
{top_content.get("source_label")}

Platform:
{top_content.get("platform")}

Category:
{top_content.get("category")}

Engagement Score:
{top_content.get("engagement_score")}

Likes:
{top_content.get("likes")}

Comments:
{top_content.get("comments")}

Shares:
{top_content.get("shares")}

Views:
{top_content.get("views")}
"""
        )

    return "\n".join(
        parts
    )


# ============================================================
# FORMAT RAW CONTEXT
# ============================================================

def format_context(
    docs: List[Document],
) -> str:
    """
    Format retrieved documents for the LLM.

    Includes:
    1. Python-calculated analytics.
    2. Original retrieved source content.

    Source numbering is shared with the UI.
    """

    if not docs:
        return (
            "No relevant sources were retrieved."
        )

    analyzed_docs = (
        analyze_retrieved_documents(
            docs
        )
    )

    analytics_context = (
        format_analytics(
            analyzed_docs
        )
    )

    source_parts = []

    source_parts.append(
        analytics_context
    )

    source_parts.append(
        "\n\nORIGINAL RETRIEVED SOURCES\n"
    )

    for item in analyzed_docs:

        source_parts.append(
            f"""
{item.get("source_label")}

Platform:
{item.get("platform")}

Category:
{item.get("category")}

Content ID:
{item.get("content_id")}

Original Content:
{item.get("original_content")}
"""
        )

    return "\n".join(
        source_parts
    )


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    vectorstore = (
        load_vectorstore()
    )

    query = (
        "Which content performs best "
        "based on engagement?"
    )

    docs = retrieve_context(
        vectorstore,
        query,
        k=5,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MARKETPULSE AI"
    )

    print(
        "PYTHON ANALYTICS TEST"
    )

    print(
        "=" * 70
    )

    print(
        f"\nQuery:\n{query}\n"
    )

    analyzed = (
        analyze_retrieved_documents(
            docs
        )
    )

    ranked = (
        rank_by_engagement(
            analyzed
        )
    )

    print(
        "\nRANKED RESULTS:"
    )

    for rank, item in enumerate(
        ranked,
        start=1,
    ):

        print(
            f"""
{rank}.
Content ID: {item.get("content_id")}
Source: {item.get("source_label")}
Platform: {item.get("platform")}
Category: {item.get("category")}
Likes: {item.get("likes")}
Comments: {item.get("comments")}
Shares: {item.get("shares")}
Views: {item.get("views")}
Engagement Score: {item.get("engagement_score")}
"""
        )

    top = get_top_content(
        analyzed
    )

    print(
        "\nTOP PERFORMER:"
    )

    if top:

        print(
            f"""
Content ID:
{top.get("content_id")}

Engagement Score:
{top.get("engagement_score")}

Source:
{top.get("source_label")}
"""
        )

    else:

        print(
            "No calculable engagement score."
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "\nFULL LLM CONTEXT:"
    )

    print(
        format_context(
            docs
        )
    )