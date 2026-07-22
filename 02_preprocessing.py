"""
02_preprocessing.py
--------------------
Pipeline stage 2: PREPROCESSING (cleaning and normalization)

Cleans the raw dataset before it's turned into documents. Every operation
here is vectorized (no row-by-row Python loops), which matters at 50,000+
rows -- a `.apply()` with a Python function over 52k rows is fine, but a
`for _, row in df.iterrows(): ...` cleaning loop is 10-50x slower and does
not scale as the dataset grows.
"""

import pandas as pd

try:
    from importlib import import_module
    _doc_mod = import_module("01_documents")
    load_raw_data = _doc_mod.load_raw_data
except ImportError:
    # Fallback for environments where numeric-prefixed module names can't
    # be imported directly (e.g. some notebook contexts).
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location(
        "doc_mod", os.path.join(os.path.dirname(__file__), "01_documents.py")
    )
    _doc_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_doc_mod)
    load_raw_data = _doc_mod.load_raw_data


TEXT_FILL_COLUMNS = ["content_description", "comments_text", "hashtags", "sponsor_name"]
NUMERIC_FILL_COLUMNS = ["likes", "shares", "comments_count", "views", "follower_count"]


def normalize_sponsored(series: pd.Series) -> pd.Series:
    """Vectorized normalization of the is_sponsored column to 'Sponsored' /
    'Not Sponsored', robust to booleans, 0/1, and 'True'/'False' strings."""
    as_str = series.astype(str).str.strip().str.lower()
    is_sponsored = as_str.isin(["true", "1", "1.0", "yes"])
    return is_sponsored.map({True: "Sponsored", False: "Not Sponsored"})


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataset.

    Steps:
    - Drop exact duplicate rows (can happen with scraped/merged data).
    - Fill missing text fields with empty strings.
    - Fill missing numeric engagement fields with 0.
    - Normalize is_sponsored into a readable label.
    - Parse post_date into an actual datetime (invalid dates -> NaT, kept,
      not dropped, since date isn't required for retrieval quality).
    - Drop rows missing the fields essential to a usable document
      (platform, content_category, content_description all blank).

    Returns a new DataFrame; does not mutate the input.
    """
    df = df.drop_duplicates().copy()

    for col in TEXT_FILL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    for col in NUMERIC_FILL_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    if "is_sponsored" in df.columns:
        df["is_sponsored"] = normalize_sponsored(df["is_sponsored"])

    if "post_date" in df.columns:
        df["post_date"] = pd.to_datetime(
            df["post_date"], format="%m/%d/%y %I:%M %p", errors="coerce"
        )

    # Drop rows with no usable content at all
    essential = [c for c in ["platform", "content_category"] if c in df.columns]
    if essential:
        df = df.dropna(subset=essential)
        df = df[(df["content_description"].str.len() > 0) | (df["comments_text"].str.len() > 0)]

    return df.reset_index(drop=True)


def load_and_preprocess(csv_path: str = None) -> pd.DataFrame:
    df = load_raw_data(csv_path) if csv_path else load_raw_data()
    return preprocess(df)


if __name__ == "__main__":
    raw = load_raw_data()
    clean = preprocess(raw)
    print(f"Raw rows:        {len(raw):,}")
    print(f"Cleaned rows:    {len(clean):,}")
    print(f"Rows dropped:    {len(raw) - len(clean):,}")
    print(f"\nis_sponsored value counts:\n{clean['is_sponsored'].value_counts()}")
