"""
I/O utilities: load parquet and iterate over rows.

Provides the data loading layer for Step 1 of the agentic pipeline.
Parquet files are read into DataFrames, then yielded as dict rows for adaptation.
"""

import pandas as pd
from pathlib import Path


def load_parquet(path: str | Path) -> pd.DataFrame:
    """
    Load a parquet file into a pandas DataFrame.
    Handles path as string or Path.
    """
    # Normalize path to Path object for consistent handling
    path = Path(path) if isinstance(path, str) else path
    # Read parquet into DataFrame (columnar format, efficient for large datasets)
    return pd.read_parquet(path)


def iter_rows(df: pd.DataFrame, limit: int | None = None):
    """
    Yield dict rows from the DataFrame.
    Each row is a dict mapping column name -> value.

    Args:
        df: The DataFrame to iterate over.
        limit: If set, yield at most this many rows. None = all rows.

    Yields:
        dict: One row per yield, with column names as keys.
    """
    # iterrows() yields (index, Series) pairs; we use the Series as a row
    it = df.iterrows()
    count = 0
    for idx, row in it:
        # Stop early if we've hit the row limit (for preview/sampling)
        if limit is not None and count >= limit:
            break
        # Convert pandas Series to dict for adapter (column name -> value)
        yield row.to_dict()
        count += 1
