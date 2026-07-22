"""
06_retrieve_context.py
----------------------
Pipeline Stage 6: RETRIEVAL

MarketPulse AI
AI-powered Marketing Intelligence

Production V6

Responsibilities:
- Load the Chroma vector store safely.
- Retrieve relevant documents.
- Keep metadata available for analytics and UI.
- Work locally on Windows.
- Work on Streamlit Cloud.
- Use the modern langchain-chroma integration.
"""

import importlib.util
import os


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_DIR = os.path.join(
    BASE_DIR,
    "chroma_store",
)

COLLECTION_NAME = "marketpulse"


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_embeddings():
    """
    Load the same embedding model used
    when creating the Chroma database.
    """

    vector_module_path = os.path.join(
        BASE_DIR,
        "04_vector_representation.py",
    )

    spec = importlib.util.spec_from_file_location(
        "vector_representation_mod",
        vector_module_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            "Could not load 04_vector_representation.py"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    # Try common function names
    if hasattr(
        module,
        "get_embedding_function",
    ):
        return module.get_embedding_function()

    if hasattr(
        module,
        "get_embeddings",
    ):
        return module.get_embeddings()

    if hasattr(
        module,
        "load_embeddings",
    ):
        return module.load_embeddings()

    if hasattr(
        module,
        "create_embeddings",
    ):
        return module.create_embeddings()

    # Try common embedding constants
    model_name = getattr(
        module,
        "EMBEDDING_MODEL_NAME",
        None,
    )

    if model_name:

        try:

            from langchain_huggingface import (
                HuggingFaceEmbeddings,
            )

            return HuggingFaceEmbeddings(
                model_name=model_name
            )

        except Exception as error:

            raise RuntimeError(
                "Could not initialize HuggingFace embeddings.\n"
                f"Technical details: {error}"
            )

    raise AttributeError(
        "No compatible embedding loader was found "
        "in 04_vector_representation.py."
    )


# ============================================================
# LOAD VECTOR STORE
# ============================================================

def load_vectorstore():
    """
    Load the existing Chroma vector store.

    Uses the modern langchain-chroma package.
    """

    # --------------------------------------------------------
    # Validate Chroma directory
    # --------------------------------------------------------

    if not os.path.exists(
        CHROMA_DIR
    ):

        raise FileNotFoundError(
            "Chroma vector store directory was not found:\n"
            f"{CHROMA_DIR}"
        )

    # --------------------------------------------------------
    # Check directory contents
    # --------------------------------------------------------

    if not os.listdir(
        CHROMA_DIR
    ):

        raise RuntimeError(
            "The Chroma vector store directory is empty:\n"
            f"{CHROMA_DIR}"
        )

    # --------------------------------------------------------
    # Load embeddings
    # --------------------------------------------------------

    embeddings = load_embeddings()

    # --------------------------------------------------------
    # Import modern Chroma integration
    # --------------------------------------------------------

    try:

        from langchain_chroma import Chroma

    except ImportError as error:

        raise ImportError(
            "The langchain-chroma package is required.\n"
            "Install it with:\n"
            "pip install -U langchain-chroma"
        ) from error

    # --------------------------------------------------------
    # Try collection name
    # --------------------------------------------------------

    try:

        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )

        # ----------------------------------------------------
        # Validate collection
        # ----------------------------------------------------

        collection = (
            vectorstore._collection
        )

        count = collection.count()

        if count == 0:

            raise RuntimeError(
                "The Chroma collection exists but contains "
                "zero documents."
            )

        return vectorstore

    except Exception as error:

        raise RuntimeError(
            "Failed to load the Chroma vector store.\n\n"
            f"Directory:\n{CHROMA_DIR}\n\n"
            f"Collection:\n{COLLECTION_NAME}\n\n"
            f"Technical details:\n{error}"
        ) from error


# ============================================================
# RETRIEVE CONTEXT
# ============================================================

def retrieve_context(
    vectorstore,
    query,
    k=5,
):
    """
    Retrieve the most relevant documents.
    """

    if vectorstore is None:

        raise ValueError(
            "Vector store is not loaded."
        )

    if not query or not query.strip():

        return []

    try:

        docs = vectorstore.similarity_search(
            query.strip(),
            k=k,
        )

        return docs

    except Exception as error:

        raise RuntimeError(
            "Failed to retrieve documents from "
            "the Chroma vector store.\n"
            f"Technical details: {error}"
        ) from error


# ============================================================
# FORMAT CONTEXT
# ============================================================

def format_context(
    docs,
):
    """
    Format retrieved documents for display
    or legacy pipeline compatibility.
    """

    if not docs:

        return (
            "No relevant documents were found."
        )

    chunks = []

    for index, doc in enumerate(
        docs,
        start=1,
    ):

        # ----------------------------------------------------
        # Content
        # ----------------------------------------------------

        if hasattr(
            doc,
            "page_content",
        ):

            content = (
                doc.page_content
            )

        elif isinstance(
            doc,
            dict,
        ):

            content = (
                doc.get(
                    "page_content"
                )
                or doc.get(
                    "content"
                )
                or doc.get(
                    "text"
                )
                or ""
            )

        else:

            content = str(
                doc
            )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        metadata = {}

        if hasattr(
            doc,
            "metadata",
        ):

            metadata = doc.metadata

        elif isinstance(
            doc,
            dict,
        ):

            metadata = doc.get(
                "metadata",
                {},
            )

        if not isinstance(
            metadata,
            dict,
        ):

            metadata = {}

        platform = metadata.get(
            "platform",
            "",
        )

        category = metadata.get(
            "category",
            "",
        )

        content_id = (
            metadata.get(
                "content_id"
            )
            or metadata.get(
                "id"
            )
            or ""
        )

        chunks.append(
            f"[Source {index}]\n"
            f"Content ID: {content_id}\n"
            f"Platform: {platform}\n"
            f"Category: {category}\n"
            f"Content: {content}"
        )

    return "\n\n".join(
        chunks
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        "============================================================"
    )

    print(
        "MARKETPULSE AI - CHROMA RETRIEVAL TEST"
    )

    print(
        "============================================================\n"
    )

    try:

        print(
            "Loading Chroma vector store..."
        )

        vectorstore = load_vectorstore()

        print(
            "✅ Chroma vector store loaded successfully."
        )

        print(
            "\nTesting retrieval..."
        )

        test_query = (
            "Which content performs best "
            "based on engagement?"
        )

        docs = retrieve_context(
            vectorstore,
            test_query,
            k=5,
        )

        print(
            f"\nRetrieved documents: {len(docs)}"
        )

        print(
            "\n"
            "============================================================"
        )

        print(
            "RETRIEVED CONTEXT"
        )

        print(
            "============================================================\n"
        )

        print(
            format_context(
                docs
            )
        )

        print(
            "\n"
            "============================================================"
        )

        print(
            "TEST COMPLETED SUCCESSFULLY"
        )

        print(
            "============================================================"
        )

    except Exception as error:

        print(
            "\n❌ TEST FAILED\n"
        )

        print(
            str(error)
        )