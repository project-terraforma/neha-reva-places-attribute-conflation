"""
Row adapter: convert raw parquet row to canonical format for the agent.
"""

from .schema_config import SCHEMA, CANONICAL_ATTRS


def _safe_value(val):
    """
    Handle nulls and ensure lists stay as lists.
    Returns None for null/NaN, otherwise the value as-is.
    """
    if val is None:
        return None
    if hasattr(val, "__iter__") and not isinstance(val, (str, dict)):
        # Could be list, ndarray, etc. - convert to list for JSON
        try:
            return list(val)
        except (TypeError, ValueError):
            return val
    # Check for pandas NA/NaN
    try:
        import pandas as pd
        if pd.isna(val):
            return None
    except ImportError:
        pass
    return val


def adapt_row(raw_row: dict, schema: dict | None = None) -> dict:
    """
    Adapt a raw parquet row to the canonical format.

    Args:
        raw_row: Dict mapping parquet column names to values.
        schema: Optional override. Defaults to SCHEMA from schema_config.

    Returns:
        {
            "id": "...",
            "base_id": "...",
            "base": {"name": ..., "address": ..., "phones": ..., ...},
            "other": {"name": ..., "address": ..., "phones": ..., ...}
        }
        Missing columns produce None. Lists (phones, socials) are passed through.
    """
    schema = schema or SCHEMA

    def get(key: str):
        col = schema.get(key)
        if col is None:
            return None
        val = raw_row.get(col)
        return _safe_value(val)

    base = {}
    other = {}
    for attr in CANONICAL_ATTRS:
        base_key = f"base.{attr}"
        other_key = f"other.{attr}"
        base[attr] = get(base_key)
        other[attr] = get(other_key)

    return {
        "id": get("id") or str(raw_row.get("id", "")),
        "base_id": get("base_id") or str(raw_row.get("base_id", "")),
        "base": base,
        "other": other,
    }
