"""
I/O utilities: load parquet and iterate over rows.
"""

import pandas as pd
from pathlib import Path


def load_parquet(path: str | Path) -> pd.DataFrame:
    """
    Load a parquet file into a pandas DataFrame.
    Handles path as string or Path.
    """
    path = Path(path) if isinstance(path, str) else path
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
    it = df.iterrows()
    count = 0
    for idx, row in it:
        if limit is not None and count >= limit:
            break
        yield row.to_dict()
        count += 1
