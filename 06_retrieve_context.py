"""
06_retrieve_context.py
----------------------

Pipeline Stage 6: RETRIEVAL

MarketPulse AI
Production V8

Loads the Chroma vector store from a private Hugging Face Dataset,
detects the existing Chroma collection automatically,
and performs retrieval.
"""

import importlib.util
import os
import shutil
from pathlib import Path

import streamlit as st


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = Path(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# CONFIGURATION
# ============================================================

# Expected collection name.
# If this collection does not exist,
# the code will automatically use
# the first available collection.

COLLECTION_NAME = "marketpulse"


# Private Hugging Face Dataset
HF_DATASET = (
    "amal-sherif71/marketpulse-chroma"
)


# Local temporary location
# where the Hugging Face Chroma database
# will be copied.

LOCAL_CHROMA_DIR = (
    BASE_DIR / "chroma_store"
)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_embeddings():

    vector_module_path = (
        BASE_DIR
        / "04_vector_representation.py"
    )

    spec = (
        importlib.util
        .spec_from_file_location(
            "vector_representation_mod",
            vector_module_path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):

        raise ImportError(
            "Could not load "
            "04_vector_representation.py"
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )


    # --------------------------------------------------------
    # Try get_embedding_function()
    # --------------------------------------------------------

    if hasattr(
        module,
        "get_embedding_function",
    ):

        return (
            module
            .get_embedding_function()
        )


    # --------------------------------------------------------
    # Try get_embeddings()
    # --------------------------------------------------------

    if hasattr(
        module,
        "get_embeddings",
    ):

        return (
            module
            .get_embeddings()
        )


    # --------------------------------------------------------
    # Try load_embeddings()
    # --------------------------------------------------------

    if hasattr(
        module,
        "load_embeddings",
    ):

        return (
            module
            .load_embeddings()
        )


    # --------------------------------------------------------
    # Try create_embeddings()
    # --------------------------------------------------------

    if hasattr(
        module,
        "create_embeddings",
    ):

        return (
            module
            .create_embeddings()
        )


    # --------------------------------------------------------
    # Try EMBEDDING_MODEL_NAME
    # --------------------------------------------------------

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

            return (
                HuggingFaceEmbeddings(
                    model_name=model_name
                )
            )

        except Exception as error:

            raise RuntimeError(
                "Could not initialize "
                "HuggingFace embeddings.\n"
                f"Technical details: {error}"
            ) from error


    # --------------------------------------------------------
    # No embedding loader found
    # --------------------------------------------------------

    raise AttributeError(
        "No compatible embedding loader "
        "was found in "
        "04_vector_representation.py."
    )


# ============================================================
# DOWNLOAD CHROMA FROM HUGGING FACE
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def download_chroma_store():

    # --------------------------------------------------------
    # Check if database already exists locally
    # --------------------------------------------------------

    sqlite_file = (
        LOCAL_CHROMA_DIR
        / "chroma.sqlite3"
    )


    if sqlite_file.exists():

        return str(
            LOCAL_CHROMA_DIR
        )


    # --------------------------------------------------------
    # Get Hugging Face Token
    # --------------------------------------------------------

    hf_token = st.secrets.get(
        "HF_TOKEN",
        None,
    )


    if not hf_token:

        raise RuntimeError(
            "HF_TOKEN is missing from "
            "Streamlit Secrets."
        )


    # --------------------------------------------------------
    # Import Hugging Face Hub
    # --------------------------------------------------------

    try:

        from huggingface_hub import (
            snapshot_download,
        )

    except ImportError as error:

        raise ImportError(
            "huggingface_hub is required.\n"
            "Install it with:\n"
            "pip install -U huggingface_hub"
        ) from error


    # --------------------------------------------------------
    # Download Dataset
    # --------------------------------------------------------

    try:

        downloaded_path = (
            snapshot_download(
                repo_id=HF_DATASET,
                repo_type="dataset",
                token=hf_token,
            )
        )

    except Exception as error:

        raise RuntimeError(
            "Failed to download the Chroma "
            "database from Hugging Face.\n\n"
            f"Dataset: {HF_DATASET}\n\n"
            f"Technical details:\n{error}"
        ) from error


    # --------------------------------------------------------
    # Locate Chroma Store
    # --------------------------------------------------------

    downloaded_chroma = (
        Path(downloaded_path)
        / "chroma_store"
    )


    if not downloaded_chroma.exists():

        raise FileNotFoundError(
            "The 'chroma_store' folder was not "
            "found inside the Hugging Face dataset.\n\n"
            f"Downloaded path:\n"
            f"{downloaded_path}"
        )


    # --------------------------------------------------------
    # Copy Chroma Database Locally
    # --------------------------------------------------------

    try:

        if LOCAL_CHROMA_DIR.exists():

            shutil.rmtree(
                LOCAL_CHROMA_DIR
            )


        shutil.copytree(
            downloaded_chroma,
            LOCAL_CHROMA_DIR,
        )

    except Exception as error:

        raise RuntimeError(
            "Failed to copy the Chroma "
            "database locally.\n"
            f"Technical details: {error}"
        ) from error


    # --------------------------------------------------------
    # Validate Chroma Database
    # --------------------------------------------------------

    if not (
        LOCAL_CHROMA_DIR
        / "chroma.sqlite3"
    ).exists():

        raise RuntimeError(
            "chroma.sqlite3 was not found "
            "after downloading the database."
        )


    return str(
        LOCAL_CHROMA_DIR
    )


# ============================================================
# LOAD VECTOR STORE
# ============================================================

def load_vectorstore(
    *args,
    **kwargs,
):

    # --------------------------------------------------------
    # Download / locate Chroma
    # --------------------------------------------------------

    chroma_dir = (
        download_chroma_store()
    )


    # --------------------------------------------------------
    # Load Embeddings
    # --------------------------------------------------------

    embeddings = (
        load_embeddings()
    )


    # --------------------------------------------------------
    # Import Chroma
    # --------------------------------------------------------

    try:

        import chromadb

        from langchain_chroma import (
            Chroma,
        )

    except ImportError as error:

        raise ImportError(
            "Chroma dependencies are required.\n"
            "Please install chromadb and "
            "langchain-chroma."
        ) from error


    # --------------------------------------------------------
    # Connect to Existing Chroma Database
    # --------------------------------------------------------

    try:

        client = (
            chromadb.PersistentClient(
                path=chroma_dir
            )
        )


        # ----------------------------------------------------
        # Discover Existing Collections
        # ----------------------------------------------------

        collections = (
            client.list_collections()
        )


        if not collections:

            raise RuntimeError(
                "No Chroma collections were found "
                "inside the downloaded database."
            )


        # ----------------------------------------------------
        # Get Collection Names
        # ----------------------------------------------------

        collection_names = [
            collection.name
            for collection in collections
        ]


        # ----------------------------------------------------
        # Select Collection
        # ----------------------------------------------------

        if (
            COLLECTION_NAME
            in collection_names
        ):

            selected_collection = (
                COLLECTION_NAME
            )

        else:

            # If "marketpulse" is not found,
            # automatically use the first
            # existing collection.

            selected_collection = (
                collection_names[0]
            )


        # ----------------------------------------------------
        # Load LangChain Chroma
        # ----------------------------------------------------

        vectorstore = Chroma(
            collection_name=(
                selected_collection
            ),
            embedding_function=embeddings,
            persist_directory=chroma_dir,
        )


        # ----------------------------------------------------
        # Validate Document Count
        # ----------------------------------------------------

        collection = (
            vectorstore._collection
        )


        count = (
            collection.count()
        )


        if count == 0:

            raise RuntimeError(
                "The Chroma collection "
                f"'{selected_collection}' "
                "exists but contains zero documents.\n\n"
                f"Available collections: "
                f"{collection_names}"
            )


        # ----------------------------------------------------
        # Success
        # ----------------------------------------------------

        return vectorstore


    except Exception as error:

        raise RuntimeError(
            "Failed to load the Chroma "
            "vector store.\n\n"
            f"Dataset:\n"
            f"{HF_DATASET}\n\n"
            f"Requested collection:\n"
            f"{COLLECTION_NAME}\n\n"
            f"Technical details:\n"
            f"{error}"
        ) from error


# ============================================================
# RETRIEVE CONTEXT
# ============================================================

def retrieve_context(
    vectorstore,
    query,
    k=5,
):

    # --------------------------------------------------------
    # Validate Vector Store
    # --------------------------------------------------------

    if vectorstore is None:

        raise ValueError(
            "Vector store is not loaded."
        )


    # --------------------------------------------------------
    # Validate Query
    # --------------------------------------------------------

    if (
        not query
        or not query.strip()
    ):

        return []


    # --------------------------------------------------------
    # Similarity Search
    # --------------------------------------------------------

    try:

        docs = (
            vectorstore
            .similarity_search(
                query.strip(),
                k=k,
            )
        )


        return docs


    except Exception as error:

        raise RuntimeError(
            "Failed to retrieve documents "
            "from the Chroma vector store.\n"
            f"Technical details:\n"
            f"{error}"
        ) from error


# ============================================================
# FORMAT CONTEXT
# ============================================================

def format_context(
    docs,
):

    # --------------------------------------------------------
    # No Documents
    # --------------------------------------------------------

    if not docs:

        return (
            "No relevant documents were found."
        )


    chunks = []


    # --------------------------------------------------------
    # Process Documents
    # --------------------------------------------------------

    for index, doc in enumerate(
        docs,
        start=1,
    ):

        # ----------------------------------------------------
        # Get Content
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
        # Get Metadata
        # ----------------------------------------------------

        metadata = {}


        if hasattr(
            doc,
            "metadata",
        ):

            metadata = (
                doc.metadata
            )


        elif isinstance(
            doc,
            dict,
        ):

            metadata = (
                doc.get(
                    "metadata",
                    {},
                )
            )


        if not isinstance(
            metadata,
            dict,
        ):

            metadata = {}


        # ----------------------------------------------------
        # Metadata Fields
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Build Context
        # ----------------------------------------------------

        chunks.append(
            f"[Source {index}]\n"
            f"Content ID: {content_id}\n"
            f"Platform: {platform}\n"
            f"Category: {category}\n"
            f"Content: {content}"
        )


    # --------------------------------------------------------
    # Return Formatted Context
    # --------------------------------------------------------

    return "\n\n".join(
        chunks
    )