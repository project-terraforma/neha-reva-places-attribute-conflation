"""
# AGENTIC APPROACH (necessary): Parquet row → canonical {id, base_id, base, other}.

Transforms the flat parquet schema (base_names, names, base_phones, phones, etc.)
into a nested structure {id, base_id, base: {...}, other: {...}} that the
unified flow expects.
"""

from .schema_config import SCHEMA, CANONICAL_ATTRS


def _safe_value(val):
    """
    Handle nulls and ensure lists stay as lists.
    Returns None for null/NaN, otherwise the value as-is.
    Pandas/NumPy types (e.g. ndarray) are converted to JSON-serializable forms.
    """
    if val is None:
        return None
    # Iterables (lists, ndarrays) -> list for JSON serialization
    if hasattr(val, "__iter__") and not isinstance(val, (str, dict)):
        try:
            return list(val)
        except (TypeError, ValueError):
            return val
    # Pandas NA/NaN must be treated as None
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

    # Helper: look up parquet column by canonical key, return safe value
    def get(key: str):
        col = schema.get(key)
        if col is None:
            return None
        val = raw_row.get(col)
        return _safe_value(val)

    # Build base and other attribute dicts from canonical attribute list
    base = {}
    other = {}
    for attr in CANONICAL_ATTRS:
        base_key = f"base.{attr}"
        other_key = f"other.{attr}"
        base[attr] = get(base_key)
        other[attr] = get(other_key)

    # Assemble final canonical row: ids + base/other attribute dicts
    return {
        "id": get("id") or str(raw_row.get("id", "")),
        "base_id": get("base_id") or str(raw_row.get("base_id", "")),
        "base": base,
        "other": other,
    }
