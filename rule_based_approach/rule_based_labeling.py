#!/usr/bin/env python3
"""
Rule-based approach to the attribute conflation problem.

This script implements a rule-based method for deciding which attributes (base vs conflated)
are higher quality when multiple representations of the same place disagree. It reads
inspection/golden/golden_dataset.json, scores each attribute pair (phones,
websites, addresses, categories), and updates the label field in-place.

Label meanings: 0=base wins, 1=alt (conflated) wins, 2=abstain (no clear winner)

Usage:
  python rule_based_approach/rule_based_labeling.py

See rule_based_approach/README.md for a description of the rules used.
"""

import ast
import json
import math
import re
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).parent.parent
GOLDEN_JSON_PATH = PROJECT_ROOT / "inspection" / "golden" / "golden_dataset.json"


# ----------------------------
# Parsing helpers
# ----------------------------

def _to_obj(x):
    """Parse a cell that may be None, already a list/dict, a JSON string, or a Python literal string."""
    if x is None:
        return None
    if isinstance(x, (list, dict)):
        return x
    if isinstance(x, float) and (math.isnan(x) or str(x) == "nan"):
        return None
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() == "nan":
            return None
        # Try JSON first
        try:
            return json.loads(s)
        except Exception:
            pass
        # Try Python literal (duckdb/pandas often prints like this)
        try:
            return ast.literal_eval(s)
        except Exception:
            return s  # fallback: keep raw string
    return x


# ----------------------------
# Phone scoring
# ----------------------------

DIGITS_RE = re.compile(r"\d+")

def _normalize_phone(p):
    """Return (digits_only, has_plus)."""
    if not isinstance(p, str):
        p = str(p)
    p = p.strip()
    has_plus = p.startswith("+")
    digits = "".join(DIGITS_RE.findall(p))
    return digits, has_plus

def _phone_quality(phones_obj):
    """
    phones_obj expected like ["9049989600"] or ["+19049989600"].
    Return an integer quality score (higher is better).
    Strict rubric:
      +5 if looks like valid US (10 digits) or valid E.164-ish (11-15 digits with leading country code)
      +2  if has '+' and 11-15 digits
      -5  if too short (<10)
      -3  if starts with obvious leading zeros pattern for long strings (e.g. "0137..." as "10 digits" is suspicious)
    """
    if phones_obj is None:
        return 0
    if isinstance(phones_obj, str):
        # sometimes it's already a single phone string
        phones_list = [phones_obj]
    elif isinstance(phones_obj, list):
        phones_list = phones_obj
    else:
        phones_list = [str(phones_obj)]

    # pick best phone in list
    best = 0
    for p in phones_list:
        if p is None:
            continue
        digits, has_plus = _normalize_phone(str(p))
        if digits == "":
            continue
        q = 0
        n = len(digits)

        # validity-ish
        if 10 <= n <= 15:
            q += 5
        else:
            q -= 5

        # has area code
        if has_plus and 11 <= n <= 15:
            q += 2

        # US local
        if n == 10 and digits.startswith("0"):
            # strict: leading zero on a 10-digit number is suspicious (often a trunk prefix in other countries)
            q -= 3

        # ultra-suspicious very short
        if n < 9:
            q -= 5

        best = max(best, q)
    return best


def phone_winner(base_phones, alt_phones):
    """
    Return +1 if base wins, -1 if alt wins, 0 if tie/unclear.
    Strict: require a gap >= 2 in quality to award the point.
    """
    bq = _phone_quality(base_phones)
    aq = _phone_quality(alt_phones)
    if bq >= aq + 2:
        return 1, bq, aq
    if aq >= bq + 2:
        return -1, bq, aq
    return 0, bq, aq


# ----------------------------
# Website scoring
# ----------------------------

SHORTENER_DOMAINS = {
    "bit.ly", "t.co", "tinyurl.com", "goo.gl", "ow.ly", "buff.ly", "rebrand.ly", "is.gd", "cutt.ly"
}

def _normalize_url(u):
    """Return parsed URL with scheme; treat bare domains as http."""
    if not isinstance(u, str):
        u = str(u)
    u = u.strip()
    if u == "" or u.lower() == "nan":
        return None
    # Some values might be empty string in list like [""]
    if u == "":
        return None
    # Ensure scheme for urlparse
    if "://" not in u:
        u = "http://" + u
    try:
        return urlparse(u)
    except Exception:
        return None

def _website_quality(websites_obj):
    """
    websites_obj expected like ["https://..."].
    Strict rubric:
      +10 if valid host with a TLD
      +3  if https
      +2  if path is short/canonical (no long tracking query)
      -6  if host is known shortener (bit.ly etc.)
      -4  if missing/empty/invalid
      -2  if has heavy query params (utm_, gclid, fbclid etc.)
    """
    if websites_obj is None:
        return 0
    if isinstance(websites_obj, str):
        websites_list = [websites_obj]
    elif isinstance(websites_obj, list):
        websites_list = websites_obj
    else:
        websites_list = [str(websites_obj)]

    best = 0
    for u in websites_list:
        if u is None:
            continue
        pu = _normalize_url(u)
        if pu is None:
            continue

        host = (pu.netloc or "").lower()
        path = pu.path or ""
        query = pu.query or ""

        q = 0
        # valid-ish host with a dot
        if host and "." in host and not host.startswith(".") and not host.endswith("."):
            q += 10
        else:
            q -= 4

        # shorteners
        if host in SHORTENER_DOMAINS:
            q -= 6

        # https
        if pu.scheme.lower() == "https":
            q += 3

        # canonical-ish: short path and no tracking
        if len(path) <= 40:
            q += 2

        # penalize tracking/query
        if query:
            if re.search(r"(utm_|gclid|fbclid|ref=|source=)", query, flags=re.IGNORECASE):
                q -= 2

        best = max(best, q)
    return best

def website_winner(base_websites, alt_websites):
    """
    Return +1 if base wins, -1 if alt wins, 0 if tie/unclear.
    Strict: require gap >= 3 to award the point.
    """
    bq = _website_quality(base_websites)
    aq = _website_quality(alt_websites)
    if bq >= aq + 3:
        return 1, bq, aq
    if aq >= bq + 3:
        return -1, bq, aq
    return 0, bq, aq


# ----------------------------
# Address scoring
# ----------------------------

def _addr_completeness(addr):
    """
    addr expected like {"freeform":..., "locality":..., "region":..., "country":..., "postcode":...}
    Return completeness score.
    Strict rubric:
      +3 if freeform present and length >= 8
      +2 if locality present
      +2 if country present
      +2 if region present and non-empty
      +3 if postcode present and length >= 4
      +1 if postcode has dash (ZIP+4) OR postcode has space letters (some countries)
      +1 if freeform contains a number (street number)
    """
    if not isinstance(addr, dict):
        return 0
    freeform = (addr.get("freeform") or "").strip()
    locality = (addr.get("locality") or "").strip()
    region = (addr.get("region") or "").strip()
    country = (addr.get("country") or "").strip()
    postcode = (addr.get("postcode") or "").strip()

    s = 0
    if len(freeform) >= 8:
        s += 3
    if locality:
        s += 2
    if country:
        s += 2
    if region:
        s += 2
    if len(postcode) >= 4:
        s += 3
        if "-" in postcode or re.search(r"[A-Za-z].*\d|\d.*[A-Za-z]", postcode):
            s += 1
    if re.search(r"\d", freeform):
        s += 1
    return s

def _addresses_quality(addresses_obj):
    """
    addresses_obj expected like [ {..} ].
    Strict: take best address completeness among list entries.
    """
    if addresses_obj is None:
        return 0
    if isinstance(addresses_obj, dict):
        addr_list = [addresses_obj]
    elif isinstance(addresses_obj, list):
        addr_list = addresses_obj
    else:
        return 0

    best = 0
    for a in addr_list:
        best = max(best, _addr_completeness(a))
    return best

def address_winner(base_addresses, alt_addresses):
    """
    Return +1 if base wins, -1 if alt wins, 0 if tie/unclear.
    Strict: require gap >= 2 to award the point (addresses are richer; smaller gap is meaningful).
    """
    bq = _addresses_quality(base_addresses)
    aq = _addresses_quality(alt_addresses)
    if bq >= aq + 2:
        return 1, bq, aq
    if aq >= bq + 2:
        return -1, bq, aq
    return 0, bq, aq


# ----------------------------
# Category scoring
# ----------------------------

def _categories_quality(cat_obj):
    """
    cat_obj expected like {"primary":"...", "alternate":[...]} or {"primary":"hotel"}.
    Strict rubric:
      +4 if primary present and non-empty and not too generic
      +1 per alternate category up to 3
      -2 if primary is missing/empty
      -2 if primary looks generic (e.g., "business", "professional_services")
    """
    if cat_obj is None:
        return 0
    if isinstance(cat_obj, str):
        # rare; treat as primary
        primary = cat_obj.strip()
        alt = []
    elif isinstance(cat_obj, dict):
        primary = (cat_obj.get("primary") or "").strip()
        alt = cat_obj.get("alternate") or []
        if isinstance(alt, str):
            alt = [alt]
        if not isinstance(alt, list):
            alt = []
    else:
        return 0

    generic = {"business", "professional_services", "retail", "services"}
    s = 0
    if primary:
        s += 4
        if primary in generic:
            s -= 2
    else:
        s -= 2

    # a few alternates can indicate richer taxonomy (but keep strict)
    s += min(3, len([x for x in alt if isinstance(x, str) and x.strip() != ""]))
    return s

def category_winner(base_categories, alt_categories):
    """
    Return +1 if base wins, -1 if alt wins, 0 if tie/unclear.
    Strict: require gap >= 2 to award the point.
    """
    bq = _categories_quality(base_categories)
    aq = _categories_quality(alt_categories)
    if bq >= aq + 2:
        return 1, bq, aq
    if aq >= bq + 2:
        return -1, bq, aq
    return 0, bq, aq


# ----------------------------
# Row scoring + labeling
# ----------------------------

def score_row(row):
    """
    Compute base_score and alt_score based on winners for each attribute.
    Returns dict with scores + debug info.
    """
    # Parse nested fields
    base_phones = _to_obj(row.get("base_phones"))
    phones = _to_obj(row.get("phones"))
    base_websites = _to_obj(row.get("base_websites"))
    websites = _to_obj(row.get("websites"))
    base_addresses = _to_obj(row.get("base_addresses"))
    addresses = _to_obj(row.get("addresses"))
    base_categories = _to_obj(row.get("base_categories"))
    categories = _to_obj(row.get("categories"))

    base_score = 0
    alt_score = 0

    # Phones
    w, bq, aq = phone_winner(base_phones, phones)
    if w == 1:
        base_score += 1
    elif w == -1:
        alt_score += 1
    phone_dbg = (w, bq, aq)

    # Websites
    w, bq, aq = website_winner(base_websites, websites)
    if w == 1:
        base_score += 1
    elif w == -1:
        alt_score += 1
    web_dbg = (w, bq, aq)

    # Addresses
    w, bq, aq = address_winner(base_addresses, addresses)
    if w == 1:
        base_score += 1
    elif w == -1:
        alt_score += 1
    addr_dbg = (w, bq, aq)

    # Categories
    w, bq, aq = category_winner(base_categories, categories)
    if w == 1:
        base_score += 1
    elif w == -1:
        alt_score += 1
    cat_dbg = (w, bq, aq)

    # Margin-2 label
    if base_score >= alt_score + 2:
        label = 0
    elif alt_score >= base_score + 2:
        label = 1
    else:
        label = 2

    return {
        "base_score": base_score,
        "alt_score": alt_score,
        "auto_label": label,
        "dbg_phone_winner": phone_dbg[0],
        "dbg_phone_baseq": phone_dbg[1],
        "dbg_phone_altq": phone_dbg[2],
        "dbg_web_winner": web_dbg[0],
        "dbg_web_baseq": web_dbg[1],
        "dbg_web_altq": web_dbg[2],
        "dbg_addr_winner": addr_dbg[0],
        "dbg_addr_baseq": addr_dbg[1],
        "dbg_addr_altq": addr_dbg[2],
        "dbg_cat_winner": cat_dbg[0],
        "dbg_cat_baseq": cat_dbg[1],
        "dbg_cat_altq": cat_dbg[2],
    }


def _flatten_for_scoring(rec):
    """
    Flatten golden_dataset record (data.base, data.current) to flat keys
    expected by score_row (base_phones, phones, etc.).
    """
    data = rec.get("data") or {}
    base = data.get("base") or {}
    current = data.get("current") or {}
    return {
        "base_phones": base.get("phones"),
        "phones": current.get("phones"),
        "base_websites": base.get("websites"),
        "websites": current.get("websites"),
        "base_addresses": base.get("addresses"),
        "addresses": current.get("addresses"),
        "base_categories": base.get("categories"),
        "categories": current.get("categories"),
    }


def main():
    # [CHANGE: read JSON instead of CSV - undo to revert]
    with open(GOLDEN_JSON_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    # Apply scoring to each record and update scores + label
    for rec in records:
        # Flatten nested data.base / data.current to flat keys for score_row
        flat = _flatten_for_scoring(rec)
        result = score_row(flat)
        # [CHANGE: populate scores section with attribute scores - undo to remove]
        rec["scores"] = {
            "phones": {
                "winner": result["dbg_phone_winner"],
                "base_q": result["dbg_phone_baseq"],
                "alt_q": result["dbg_phone_altq"],
            },
            "websites": {
                "winner": result["dbg_web_winner"],
                "base_q": result["dbg_web_baseq"],
                "alt_q": result["dbg_web_altq"],
            },
            "addresses": {
                "winner": result["dbg_addr_winner"],
                "base_q": result["dbg_addr_baseq"],
                "alt_q": result["dbg_addr_altq"],
            },
            "categories": {
                "winner": result["dbg_cat_winner"],
                "base_q": result["dbg_cat_baseq"],
                "alt_q": result["dbg_cat_altq"],
            },
            "base_score": result["base_score"],
            "alt_score": result["alt_score"],
        }
        rec["label"] = result["auto_label"]
        # [END CHANGE]

    # [CHANGE: write JSON instead of CSV - undo to revert]
    with open(GOLDEN_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"Updated {GOLDEN_JSON_PATH} with scores ({len(records)} rows)")
    # [END CHANGE]

    print("Label meanings: 0=base, 1=alt, 2=abstain")
    counts = {}
    for rec in records:
        lbl = rec.get("label", -1)
        counts[lbl] = counts.get(lbl, 0) + 1
    for lbl, name in [(0, "base"), (1, "alt"), (2, "abstain")]:
        n = counts.get(lbl, 0)
        print(f"  {lbl} ({name}): {n} rows")


if __name__ == "__main__":
    main()