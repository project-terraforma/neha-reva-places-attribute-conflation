"""
Schema config: mapping of canonical fields to actual parquet columns.

Maps the standard adapter output keys (base.*, other.*) to the column names
in the places attribute conflation parquet. This allows the adapter to
transform raw parquet rows into a uniform structure regardless of source schema.
"""

# SCHEMA: Maps our canonical field names (used in adapted output) to the actual
# parquet column names in project_a_samples.parquet.
# - "base" = original place record from one dataset (e.g., Microsoft)
# - "other" = conflated/merged record combining attributes from multiple sources
SCHEMA = {
    # Row identifiers
    "id": "id",                    # Conflated record ID
    "base_id": "base_id",          # Base (original) place record ID
    # Base record attributes (original source)
    "base.name": "base_names",
    "base.address": "base_addresses",
    "base.phones": "base_phones",
    "base.website": "base_websites",
    "base.category": "base_categories",
    "base.brand": "base_brand",
    "base.socials": "base_socials",
    "base.email": "base_emails",
    # Other/conflated record attributes (merged from multiple sources)
    "other.name": "names",
    "other.address": "addresses",
    "other.phones": "phones",
    "other.website": "websites",
    "other.category": "categories",
    "other.brand": "brand",
    "other.socials": "socials",
    "other.email": "emails",
}

# CANONICAL_ATTRS: Ordered list of attribute keys used when building base/other
# dicts. Order determines output structure for the unified flow.
CANONICAL_ATTRS = [
    "name",
    "address",
    "phones",
    "website",
    "category",
    "brand",
    "socials",
    "email",
]
