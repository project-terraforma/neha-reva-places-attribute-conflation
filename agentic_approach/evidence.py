"""
Evidence utilities for the unified agentic flow.

Provides fetch_url, extract_fields, classify_page, build_query, and related helpers
used by flow.py for row-by-row processing. The batch evidence-gathering pipeline
(collect all rows → compare afterwards) has been removed; only the unified flow remains.
"""

import json
import re
from typing import Any
from urllib.parse import urlparse

# Optional: requests for HTTP fetches
try:
    import requests
except ImportError:
    requests = None

# Optional: BeautifulSoup for HTML parsing (falls back to regex if missing)
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

from .validators import validate_phone_number, try_with_region, verify_website

# Page classification: how we categorize each fetched URL
CLASS_PARKED = "parked"        # Domain for sale, placeholder page
CLASS_IRRELEVANT = "irrelevant"  # Page doesn't match the business
CLASS_BLOCKED = "blocked"      # 403, captcha, rate limit
CLASS_RELEVANT = "relevant"   # Page appears to be the business


def _parse_json_val(val: Any) -> Any:
    """
    Parse value that might be a JSON string (e.g. '{"primary":"..."}' from parquet).
    Returns parsed dict/list, or the string as-is if not JSON.
    """
    if val is None:
        return None
    if isinstance(val, dict) or isinstance(val, list):
        return val
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none"):
        return None
    # Try to parse as JSON (Overture stores nested structures as JSON strings)
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
    return s


def _first_str(val: Any) -> str | None:
    """Get first string from list, dict (primary/raw), or single value."""
    if val is None:
        return None
    parsed = _parse_json_val(val)
    if isinstance(parsed, list) and len(parsed) > 0:
        s = str(parsed[0]).strip()
        return s if s else None
    if isinstance(parsed, dict):
        return (parsed.get("primary") or parsed.get("raw") or str(parsed)).strip() or None
    s = str(parsed).strip()
    return s if s else None


def _list_strs(val: Any) -> list[str]:
    """Get list of non-empty strings from value (e.g. multiple URLs, phone numbers)."""
    if val is None:
        return []
    parsed = _parse_json_val(val)
    if isinstance(parsed, list):
        return [str(x).strip() for x in parsed if str(x).strip()]
    if isinstance(parsed, dict):
        v = parsed.get("primary") or parsed.get("raw") or str(parsed)
        return [v.strip()] if v and str(v).strip() else []
    s = str(parsed).strip()
    return [s] if s else []


def _alternate_categories(val: Any) -> list[str]:
    """Get alternate category strings from a category value (dict with 'alternate' key)."""
    if val is None:
        return []
    parsed = _parse_json_val(val)
    if isinstance(parsed, dict):
        alt = parsed.get("alternate")
        if isinstance(alt, list):
            return [str(x).strip() for x in alt if str(x).strip()]
    return []


def _region_from_address(addr: Any) -> str:
    """
    Infer country/region code from address text for phone parsing.
    Used when validating numbers that lack country code (e.g. (555) 123-4567).
    """
    if addr is None:
        return "US"
    text = str(addr).upper()
    if " UNITED STATES" in text or " USA" in text or " U.S." in text:
        return "US"
    if " CANADA" in text or " CAN " in text or " ON " in text or " BC " in text:
        return "CA"
    if " UNITED KINGDOM" in text or " UK" in text or " GB " in text:
        return "GB"
    if " ZA" in text or " SOUTH AFRICA" in text:
        return "ZA"
    if " ES" in text or " SPAIN" in text:
        return "ES"
    if re.search(r"\b(AK|AL|AR|AZ|CA|CO|CT|DE|FL|GA|HI|IA|ID|IL|IN|KS|KY|LA|MA|MD|ME|MI|MN|MO|MS|MT|NC|ND|NE|NH|NJ|NM|NV|NY|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VA|VT|WA|WI|WV|WY)\b", text):
        return "US"
    return "US"


def build_query(row: dict) -> dict:
    """
    Build search/evidence query from adapted row.
    Extracts: place name, locality/region from address, website URLs to fetch.
    Website order: other (conflated) first, then base — we prefer the merged record's URLs.
    """
    base = row.get("base") or {}
    other = row.get("other") or {}
    name = _first_str(other.get("name") or base.get("name"))
    addr = other.get("address") or base.get("address")
    locality = ""
    region = ""
    # Parse address structure for locality (city) and region (state/province)
    if addr:
        parsed = _parse_json_val(addr)
        if isinstance(parsed, list) and parsed:
            item = parsed[0] if isinstance(parsed[0], dict) else {}
            locality = str(item.get("locality", "") or "").strip()
            region = str(item.get("region", "") or "").strip()
        elif isinstance(parsed, dict):
            locality = str(parsed.get("locality", "") or "").strip()
            region = str(parsed.get("region", "") or "").strip()

    # Collect website URLs: other first (conflated), then base; dedupe
    urls = []
    for w in [other.get("website"), base.get("website")]:
        for u in _list_strs(w):
            if u.startswith(("http://", "https://")) and u not in urls:
                urls.append(u)

    return {
        "name": name or "",
        "locality": locality,
        "region": region,
        "website_candidates": urls,
    }


def _normalize_url(url: str) -> str:
    """Ensure URL has scheme (https://) for requests."""
    s = url.strip()
    if s and not s.startswith(("http://", "https://")):
        return "https://" + s
    return s


def _is_parked(html: str, url: str) -> bool:
    """Heuristic: detect parked domain / for-sale page (not the actual business)."""
    if not html:
        return False
    lower = html.lower()
    parked_phrases = [
        "buy this domain",
        "domain for sale",
        "this domain is for sale",
        "parking",
        "domain parking",
        "this website is under construction",
        "coming soon",
        "domain expired",
    ]
    return any(p in lower for p in parked_phrases)


def _is_blocked(html: str, status: int) -> bool:
    """Detect blocked page: HTTP 403/429/503 or captcha/Cloudflare in body."""
    if status in (403, 429, 503):
        return True
    if not html:
        return False
    lower = html.lower()
    return any(
        p in lower
        for p in ["access denied", "captcha", "cloudflare", "blocked", "rate limit"]
    )


def _text_relevance(html: str, name: str, locality: str) -> float:
    """
    Score how relevant the page is to the business (0-1).
    Checks if business name and locality appear in visible text (not scripts/styles).
    Used to classify page as relevant vs irrelevant.
    """
    if not html or not name:
        return 0.0
    # Strip scripts/styles to get visible text only
    if BeautifulSoup:
        try:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True).lower()
        except Exception:
            text = html.lower()
    else:
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.I).lower()
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.I)

    # Fraction of name tokens found in text; bonus if locality appears
    name_parts = [p.lower() for p in re.split(r"\s+", name) if len(p) >= 2]
    if not name_parts:
        return 0.5  # No name to check
    matches = sum(1 for p in name_parts if p in text)
    score = matches / len(name_parts) if name_parts else 0.5
    if locality and locality.lower() in text:
        score = min(1.0, score + 0.2)
    return min(1.0, score)


def classify_page(html: str, status: int, url: str, query: dict) -> str:
    """
    Classify fetched page into one of: parked, irrelevant, blocked, relevant.
    Only "relevant" pages are used for field extraction.
    """
    if _is_blocked(html, status):
        return CLASS_BLOCKED
    if _is_parked(html, url):
        return CLASS_PARKED
    rel = _text_relevance(html, query.get("name", ""), query.get("locality", ""))
    if rel < 0.3:
        return CLASS_IRRELEVANT
    return CLASS_RELEVANT


def _extract_phones_from_text(text: str, region: str) -> list[tuple[str, float]]:
    """
    Extract phone numbers from plain text using regex patterns.
    Returns [(phone_str, confidence), ...]. Validated numbers get higher confidence.
    """
    results = []
    # E.164 / US-style patterns (with country code, international, local)
    patterns = [
        (r"\+1[\s\-\.]?\(?(\d{3})\)?[\s\-\.]?(\d{3})[\s\-\.]?(\d{4})\b", 0.9),
        (r"\+?(\d{1,3})[\s\-\.]?\(?(\d{3})\)?[\s\-\.]?(\d{3})[\s\-\.]?(\d{4})\b", 0.85),
        (r"\(?(\d{3})\)?[\s\-\.]?(\d{3})[\s\-\.]?(\d{4})\b", 0.75),
    ]
    seen = set()
    for pat, conf in patterns:
        for m in re.finditer(pat, text):
            num = re.sub(r"\D", "", m.group(0))
            if len(num) >= 10 and num not in seen:
                seen.add(num)
                # Validate with region hint; lower confidence if invalid
                ok, _ = try_with_region(m.group(0), region)
                results.append((m.group(0).strip(), conf if ok else 0.5))
    return results


def _extract_address_from_text(text: str) -> list[tuple[str, float]]:
    """Extract US-style address strings (street + city + state + zip) via regex."""
    results = []
    # US-style: "123 Main St, City, ST 12345"
    pat = r"(\d+[\w\s\.\-]+(?:street|st|ave|avenue|blvd|road|rd|pkwy|drive|dr|lane|ln)[\w\s\.\-]*(?:,\s*[\w\s]+)?(?:,\s*[A-Z]{2})?\s*\d{5}(?:-\d{4})?)"
    for m in re.finditer(pat, text, re.I):
        addr = m.group(1).strip()
        if len(addr) > 15 and addr not in [r[0] for r in results]:
            results.append((addr, 0.7))
    return results


def _extract_emails(text: str) -> list[tuple[str, float]]:
    """Extract email addresses; exclude example/domain placeholders."""
    results = []
    pat = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    for m in re.finditer(pat, text):
        email = m.group(0)
        if "example" not in email.lower() and "domain" not in email.lower():
            results.append((email, 0.85))
    return results


def _extract_socials(html: str) -> list[tuple[str, float]]:
    """Extract social media URLs (Facebook, Twitter, Instagram, LinkedIn) from hrefs."""
    results = []
    urls = re.findall(r'href=["\']([^"\']+)["\']', html, re.I)
    for u in urls:
        lower = u.lower()
        if "facebook.com" in lower or "twitter.com" in lower or "instagram.com" in lower or "linkedin.com" in lower:
            if "share" not in lower and "sharer" not in lower:
                results.append((u, 0.8))
    return results


def extract_fields(html: str, url: str, query: dict, region: str) -> dict[str, list[tuple[str, float]]]:
    """
    Extract business fields from HTML: name, phones, address, category, emails, socials.
    Priority: Schema.org LocalBusiness JSON-LD > title/h1 > regex from text.
    Returns dict of field -> [(value, confidence), ...]
    """
    out = {
        "name": [],
        "phones": [],
        "address": [],
        "category": [],
        "emails": [],
        "socials": [],
    }

    text = html
    if BeautifulSoup:
        try:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)
            # Name: title tag (0.8), h1 (0.85), Schema.org LocalBusiness (0.95)
            title = soup.find("title")
            if title and title.get_text(strip=True):
                t = title.get_text(strip=True)
                if len(t) > 2 and len(t) < 150:
                    out["name"].append((t, 0.8))
            h1 = soup.find("h1")
            if h1 and h1.get_text(strip=True):
                t = h1.get_text(strip=True)
                if len(t) > 2 and len(t) < 150:
                    out["name"].append((t, 0.85))
            # Schema.org LocalBusiness JSON-LD: highest confidence structured data
            for ld in soup.find_all("script", type="application/ld+json"):
                try:
                    data = json.loads(ld.string or "{}")
                    if isinstance(data, dict):
                        if data.get("@type") == "LocalBusiness":
                            n = data.get("name")
                            if n:
                                out["name"].append((str(n), 0.95))
                            tel = data.get("telephone")
                            for p in ([tel] if isinstance(tel, str) else (tel or [])):
                                if isinstance(p, str):
                                    ok, _ = try_with_region(p, region)
                                    out["phones"].append((p, 0.95 if ok else 0.7))
                            addr = data.get("address")
                            if isinstance(addr, dict):
                                freeform = addr.get("streetAddress") or addr.get("addressLocality") or str(addr)
                                if freeform:
                                    out["address"].append((str(freeform), 0.9))
                            elif isinstance(addr, str):
                                out["address"].append((addr, 0.85))
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and item.get("@type") == "LocalBusiness":
                                n = item.get("name")
                                if n:
                                    out["name"].append((str(n), 0.95))
                                tel = item.get("telephone")
                                for p in ([tel] if isinstance(tel, str) else (tel or [])):
                                    if isinstance(p, str):
                                        ok, _ = try_with_region(p, region)
                                        out["phones"].append((p, 0.95 if ok else 0.7))
                                addr = item.get("address")
                                if isinstance(addr, dict):
                                    freeform = addr.get("streetAddress") or addr.get("addressLocality") or str(addr)
                                    if freeform:
                                        out["address"].append((str(freeform), 0.9))
                except (json.JSONDecodeError, TypeError):
                    pass
        except Exception:
            text = html
    else:
        # No BeautifulSoup: strip script/style tags with regex
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.I)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.I)

    # Fallback: regex extraction when Schema.org/title/h1 didn't yield results
    if not out["phones"]:
        for val, conf in _extract_phones_from_text(text, region):
            out["phones"].append((val, conf))
    if not out["address"]:
        for val, conf in _extract_address_from_text(text):
            out["address"].append((val, conf))
    for val, conf in _extract_emails(text):
        out["emails"].append((val, conf))
    for val, conf in _extract_socials(html):
        out["socials"].append((val, conf))

    return out


def fetch_url(url: str, timeout: int = 10) -> tuple[int, str, str]:
    """
    Fetch URL via HTTP GET. Returns (status_code, final_url_after_redirects, html_body).
    """
    if not requests:
        return 0, url, ""
    try:
        r = requests.get(
            _normalize_url(url),
            timeout=timeout,
            allow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; PlacesEvidence/1.0)"},
        )
        return r.status_code, r.url, r.text or ""
    except requests.RequestException:
        return 0, url, ""
    except Exception:
        return 0, url, ""


