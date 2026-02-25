"""
Schema config: mapping of canonical fields to actual parquet columns.

Maps the standard adapter output keys (base.*, other.*) to the column names
in the places attribute conflation parquet.
"""

# Canonical field -> parquet column name
# "other" = conflated/merged record; "base" = original record
SCHEMA = {
    "id": "id",
    "base_id": "base_id",
    "base.name": "base_names",
    "other.name": "names",
    "base.address": "base_addresses",
    "other.address": "addresses",
    "base.phones": "base_phones",
    "other.phones": "phones",
    "base.website": "base_websites",
    "other.website": "websites",
    "base.category": "base_categories",
    "other.category": "categories",
    "base.brand": "base_brand",
    "other.brand": "brand",
    "base.socials": "base_socials",
    "other.socials": "socials",
    "base.email": "base_emails",
    "other.email": "emails",
}

# Canonical attribute keys for base/other dicts (in output order)
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
