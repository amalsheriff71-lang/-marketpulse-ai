"""
01_documents.py
----------------
Pipeline stage 1: DOCUMENTS (raw data ingestion)

Loads the raw social media dataset from disk. This is the only stage that
touches the raw CSV — every later stage imports `load_raw_data()` from here
instead of re-reading the file, so the dataset is only parsed once per
process.

Dataset size note
------------------
This dataset has 50,000+ rows (52,214 at last count). `pd.read_csv` handles
that comfortably in memory (~30-40 MB), so no chunked reading is required
at ingestion time. The expensive part of the pipeline is embedding, which
is handled with batching in 05_create_chroma_store.py.
"""

import os
import pandas as pd

# Resolve the dataset path relative to this file so the script works the
# same whether it's run from the repo root, from Streamlit Cloud, or from
# a different working directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data", "social_media_dataset.csv")


def load_raw_data(csv_path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the raw social media dataset from CSV.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file. Defaults to data/social_media_dataset.csv
        relative to this script.

    Returns
    -------
    pd.DataFrame
        The raw, unmodified dataset.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'. Make sure "
            "data/social_media_dataset.csv is present in the project."
        )
    df = pd.read_csv(csv_path)
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print("\nMissing values per column:")
    print(df.isna().sum()[df.isna().sum() > 0])
    print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
