"""
Processes each row in sequence:
1. Test website accessibility for base and alt
2. If one works → point to that source; if both → compare names, prefer more info, select correlating website
3. If neither works → escalate to online search
4. Fetch website content, compare with base/alt to determine accuracy
5. Use LLM to aggregate keywords and pick better category/description
6. If both same → select alt
"""

import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .evidence import (
    CLASS_RELEVANT,
    _list_strs,
    _first_str,
    _alternate_categories,
    _parse_json_val,
    _region_from_address,
    build_query,
    classify_page,
    extract_fields,
    fetch_url,
    find_contact_or_about_links,
)
from .evidence import _normalize_url  # noqa: F401 - used

# Optional LLM
try:
    from .llm import aggregate_website_keywords, compare_categories_with_llm
except ImportError:
    aggregate_website_keywords = None
    compare_categories_with_llm = None

# Get list of website URLs from base or other dict.
def _urls_from_source(source: dict, key: str = "website") -> list[str]:
    urls = []
    for u in _list_strs(source.get(key)):
        if u.startswith(("http://", "https://")) and u not in urls:
            urls.append(u)
    return urls

# Test accessibility of URLs. Fetches once per URL.
# Returns (any_accessible, first_accessible_url, html_or_none).
# When accessible, returns HTML for reuse (avoids double fetch).
def _test_website_accessibility(
    urls: list[str],
    delay: float = 0.3,
    timeout: int = 8,
) -> tuple[bool, str | None, str | None]:
    for url in urls[:3]:  # Limit to 3 URLs per source
        url = _normalize_url(url)
        if not url:
            continue
        status, final_url, html = fetch_url(url, timeout=timeout)
        time.sleep(delay)
        if 200 <= status < 400:
            return True, final_url, html
    return False, None, None


def _name_has_more_info(name_a: str, name_b: str) -> int:
    """
    Compare two names. Returns 1 if name_a has more info, -1 if name_b, 0 if tie.
    "More info" = longer, or contains additional meaningful tokens (e.g. location, type).
    """
    if not name_a and not name_b:
        return 0
    if not name_a:
        return -1
    if not name_b:
        return 1
    a = name_a.strip()
    b = name_b.strip()
    if a == b:
        return 0
    # Heuristic: longer often has more (e.g. "Goin' Postal Jacksonville" vs "Goin' Postal")
    if len(a) > len(b) * 1.2:
        return 1
    if len(b) > len(a) * 1.2:
        return -1
    # Check if one is substring of other
    if a.lower() in b.lower():
        return -1  # b has more
    if b.lower() in a.lower():
        return 1  # a has more
    return 0


def _names_mostly_match(name_a: str, name_b: str, threshold: float = 0.6) -> bool:
    """True if names are mostly the same (e.g. same core, one has extra)."""
    if not name_a or not name_b:
        return False
    a = set(re.findall(r"\w+", name_a.lower()))
    b = set(re.findall(r"\w+", name_b.lower()))
    if not a or not b:
        return name_a.lower() == name_b.lower()
    overlap = len(a & b) / max(len(a), len(b))
    return overlap >= threshold


def _url_belongs_to_source(url: str, source_urls: list[str]) -> bool:
    """True if url has same host as any source URL."""
    try:
        host = urlparse(url.strip().lower()).netloc or ""
        if host.startswith("www."):
            host = host[4:]
        for s in source_urls:
            shost = urlparse(s.strip().lower()).netloc or ""
            if shost.startswith("www."):
                shost = shost[4:]
            if host == shost:
                return True
    except Exception:
        pass
    return False


def _normalize_phone_no_leading_zero(s: str) -> str:
    """Digits only. Do NOT strip leading 0 (per user: not for leading 0s)."""
    if not s:
        return ""
    return re.sub(r"\D", "", str(s))


def _phones_equivalent(a: str | None, b: str | None) -> bool:
    """
    True if phones match with/without area code. NOT for leading 0s.
    E.g. 9049989600 vs +19049989600: match (10 vs 11 with country code 1).
    E.g. 0137573800 vs +27137573800: do NOT strip leading 0, so 0137573800 stays different.
    """
    if not a or not b:
        return False
    na = _normalize_phone_no_leading_zero(a)
    nb = _normalize_phone_no_leading_zero(b)
    if len(na) < 9 or len(nb) < 9:
        return False
    if na == nb:
        return True
    # US/CA: 10 digits vs 11 with leading 1 (country code)
    if len(na) == 10 and len(nb) == 11 and nb.startswith("1"):
        return na == nb[1:]
    if len(nb) == 10 and len(na) == 11 and na.startswith("1"):
        return nb == na[1:]
    # International: longer has country code, compare suffixes (but NOT leading 0)
    shorter, longer = (na, nb) if len(na) < len(nb) else (nb, na)
    if len(longer) >= len(shorter) and longer.endswith(shorter):
        return True
    if len(shorter) >= len(longer) and shorter.endswith(longer):
        return True
    return False


def _any_phones_equivalent(base_vals: list[str], other_vals: list[str]) -> bool:
    for b in base_vals:
        for o in other_vals:
            if _phones_equivalent(b, o):
                return True
    return False


def _urls_equivalent(a: str | None, b: str | None) -> bool:
    """True if same host."""
    if not a or not b:
        return False
    try:
        pa = urlparse(a.strip().lower())
        pb = urlparse(b.strip().lower())
        ha = pa.netloc or pa.path or ""
        hb = pb.netloc or pb.path or ""
        if ha.startswith("www."):
            ha = ha[4:]
        if hb.startswith("www."):
            hb = hb[4:]
        return ha == hb and bool(ha)
    except Exception:
        return False


def _any_urls_equivalent(base_vals: list[str], other_vals: list[str]) -> bool:
    for b in base_vals:
        for o in other_vals:
            if _urls_equivalent(b, o):
                return True
    return False


def _addresses_equivalent(base_val: Any, other_val: Any) -> bool:
    """True if addresses are effectively the same."""
    def _addr_parts(val):
        parsed = _parse_json_val(val)
        if isinstance(parsed, list) and parsed:
            item = parsed[0] if isinstance(parsed[0], dict) else {}
            freeform = str(item.get("freeform", "") or "").strip()
            locality = str(item.get("locality", "") or "").strip()
            postcode = str(item.get("postcode", "") or "").strip()
            return (freeform, locality, postcode)
        if isinstance(parsed, dict):
            return (
                str(parsed.get("freeform", "") or "").strip(),
                str(parsed.get("locality", "") or "").strip(),
                str(parsed.get("postcode", "") or "").strip(),
            )
        s = str(parsed).strip()
        return (s, "", "")
    bf, bl, bp = _addr_parts(base_val)
    of, ol, op = _addr_parts(other_val)
    if not bf and not of:
        return bl == ol and bp == op
    bp_norm = re.sub(r"\D", "", bp)
    op_norm = re.sub(r"\D", "", op)
    bf_core = re.sub(r"\s+", " ", bf.lower())[:20]
    of_core = re.sub(r"\s+", " ", of.lower())[:20]
    return bf_core in of_core or of_core in bf_core or bf.lower() == of.lower()


def _online_search_escalation(query: dict, config: dict | None) -> list[str]:
    """
    Escalate to online search when neither base nor alt websites are accessible.
    Returns list of URLs to try. Uses duckduckgo-search if available.
    """
    try:
        from ddgs import DDGS
        name = query.get("name", "")
        locality = query.get("locality", "")
        region = query.get("region", "")
        search_term = f"{name} {locality} {region}".strip() or name
        if not search_term:
            return []
        with DDGS() as ddgs:
            results = list(ddgs.text(search_term, max_results=5))
        urls = []
        for r in results:
            u = r.get("href") or r.get("url")
            if u and u.startswith(("http://", "https://")) and u not in urls:
                urls.append(u)
        return urls[:5]
    except ImportError:
        return []
    except Exception:
        return []


def _get_visible_text(html: str) -> str:
    """Extract visible text from HTML for LLM."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)
    except Exception:
        return re.sub(r"<[^>]+>", " ", html)


def process_row(
    row: dict,
    *,
    delay: float = 0.5,
    use_llm: bool = True,
    llm_model: str | None = None,
    debug: bool = False,
) -> dict:
    """
    Process one row through the unified agentic flow.

    Returns row dict with label, base_score, alt_score, attr_winners, etc.
    """
    dbg = {} if debug else None

    base = row.get("base") or {}
    other = row.get("other") or {}
    base_urls = _urls_from_source(base)
    alt_urls = _urls_from_source(other)

    base_score = 0
    alt_score = 0
    # attr_winners: 1=base, -1=alt, 0=none (inconclusive), 2=both (equivalent)
    winners = {"name": 0, "phones": 0, "website": 0, "address": 0, "category": 0}

    # --- Step 0: Name comparison (always run) ---
    base_name = _first_str(base.get("name"))
    alt_name = _first_str(other.get("name"))
    if not base_name and not alt_name:
        winners["name"] = 0
        if dbg is not None:
            dbg["name_reason"] = "both_empty"
    elif base_name and not alt_name:
        base_score += 1
        winners["name"] = 1
        if dbg is not None:
            dbg["name_reason"] = "only_base"
    elif alt_name and not base_name:
        alt_score += 1
        winners["name"] = -1
        if dbg is not None:
            dbg["name_reason"] = "only_alt"
    else:
        if _names_mostly_match(base_name or "", alt_name or ""):
            more = _name_has_more_info(base_name or "", alt_name or "")
            if more == 1:
                base_score += 1
                winners["name"] = 1
                if dbg is not None:
                    dbg["name_reason"] = "base_has_more_info"
            elif more == -1:
                alt_score += 1
                winners["name"] = -1
                if dbg is not None:
                    dbg["name_reason"] = "alt_has_more_info"
            else:
                base_score += 1
                alt_score += 1
                winners["name"] = 2
                if dbg is not None:
                    dbg["name_reason"] = "equivalent"
        else:
            winners["name"] = 0
            if dbg is not None:
                dbg["name_reason"] = "differ_no_evidence"

    # --- Step 1: Test website accessibility for base and alt ---
    base_accessible, base_working_url, base_html = _test_website_accessibility(base_urls, delay=delay)
    time.sleep(delay)
    alt_accessible, alt_working_url, alt_html = _test_website_accessibility(alt_urls, delay=delay)

    if dbg is not None:
        dbg["step1_accessibility"] = {
            "base_urls_tried": base_urls[:3],
            "alt_urls_tried": alt_urls[:3],
            "base_accessible": base_accessible,
            "base_working_url": base_working_url,
            "alt_accessible": alt_accessible,
            "alt_working_url": alt_working_url,
        }

    # Website: only from accessibility (no source-vs-source; which URL actually works)
    if base_accessible and not alt_accessible:
        base_score += 1
        winners["website"] = 1
        if dbg is not None:
            dbg["website_reason"] = "only_base_accessible"
    elif alt_accessible and not base_accessible:
        alt_score += 1
        winners["website"] = -1
        if dbg is not None:
            dbg["website_reason"] = "only_alt_accessible"
    elif base_accessible and alt_accessible:
        # Both accessible — website correctness is only from fetch; no name comparison
        base_score += 1
        alt_score += 1
        winners["website"] = 2
        if dbg is not None:
            dbg["website_reason"] = "both_accessible_both"
    else:
        # Neither accessible: escalate to online search
        winners["website"] = 0
        if dbg is not None:
            dbg["website_reason"] = "neither_accessible"

    # --- Step 2: Fetch website content and compare with base/alt ---
    query = build_query(row)
    region = _region_from_address(other.get("address") or base.get("address"))
    html_snippets = []
    extracted_per_url = {}

    # Build list of (source, url, html) - reuse HTML from accessibility test when available
    to_process = []
    if base_working_url and base_html:
        to_process.append(("base", base_working_url, base_html))
    if alt_working_url and alt_working_url != base_working_url and alt_html:
        to_process.append(("alt", alt_working_url, alt_html))

    # If neither base nor alt worked, escalate to online search
    if not to_process:
        search_urls = _online_search_escalation(query, None)
        search_term = f"{query.get('name', '')} {query.get('locality', '')} {query.get('region', '')}".strip()
        for u in search_urls[:3]:
            status, final_url, html = fetch_url(u, timeout=10)
            time.sleep(delay)
            if 200 <= status < 400:
                to_process.append(("search", final_url, html))
        if dbg is not None:
            dbg["search_escalation"] = {"query": search_term, "urls_found": search_urls[:5], "urls_fetched": [t[1] for t in to_process]}

    relevant_pages: list[tuple[str, str]] = []  # (url, html)
    for _src, url, html in to_process:
        classification = classify_page(html, 200, url, query)
        if classification == CLASS_RELEVANT:
            fields = extract_fields(html, url, query, region)
            extracted_per_url[url] = fields
            text = _get_visible_text(html)
            if text:
                html_snippets.append(text[:4000])
            relevant_pages.append((url, html))

    # Scope past first page: fetch contact/about links from relevant pages
    extra_fetched = 0
    for page_url, page_html in relevant_pages:
        if extra_fetched >= 2:
            break
        for link in find_contact_or_about_links(page_html, page_url, max_links=2):
            if link in extracted_per_url:
                continue
            status, final_url, html = fetch_url(link, timeout=10)
            time.sleep(delay)
            if 200 <= status < 400:
                if classify_page(html, 200, final_url, query) == CLASS_RELEVANT:
                    fields = extract_fields(html, final_url, query, region)
                    extracted_per_url[final_url] = fields
                    text = _get_visible_text(html)
                    if text:
                        html_snippets.append(text[:4000])
                    extra_fetched += 1
                    if extra_fetched >= 2:
                        break
    if dbg is not None and extra_fetched:
        dbg["contact_page_fetches"] = extra_fetched

    # --- Step 3: LLM aggregation for category/description ---
    llm_result = None
    if dbg is not None and base.get("category") and other.get("category"):
        base_cat = _first_str(base.get("category"))
        alt_cat = _first_str(other.get("category"))
        if base_cat and alt_cat and base_cat.lower() != alt_cat.lower():
            dbg["category_llm_diagnostic"] = {
                "use_llm": use_llm,
                "html_snippets_count": len(html_snippets),
                "aggregate_available": aggregate_website_keywords is not None,
            }
    if use_llm and html_snippets and aggregate_website_keywords:
        llm_result = aggregate_website_keywords(
            html_snippets,
            query.get("name", ""),
            model=llm_model or None,
        )
        if dbg is not None and base.get("category") and other.get("category"):
            base_cat = _first_str(base.get("category"))
            alt_cat = _first_str(other.get("category"))
            if base_cat and alt_cat and base_cat.lower() != alt_cat.lower():
                diag = dbg.get("category_llm_diagnostic", {})
                diag["llm_called"] = True
                diag["llm_result"] = llm_result is not None
                dbg["category_llm_diagnostic"] = diag
        if llm_result:
            base_cat = _first_str(base.get("category"))
            alt_cat = _first_str(other.get("category"))
            base_alternates = _alternate_categories(base.get("category"))
            alt_alternates = _alternate_categories(other.get("category"))
            if base_cat and alt_cat and compare_categories_with_llm:
                llm_choice = compare_categories_with_llm(
                    base_cat, alt_cat, llm_result,
                    base_alternates=base_alternates or None,
                    alt_alternates=alt_alternates or None,
                    model=llm_model or None,
                )
                if llm_choice == "base":
                    base_score += 1
                    winners["category"] = 1
                    if dbg is not None:
                        dbg["category_reason"] = "llm_prefers_base"
                        dbg["category_winner_debug"] = {"base_category": base_cat, "alt_category": alt_cat, "llm_aggregated": llm_result.get("category"), "decision": "base", "reason": "llm_prefers_base"}
                elif llm_choice == "alt":
                    alt_score += 1
                    winners["category"] = -1
                    if dbg is not None:
                        dbg["category_reason"] = "llm_prefers_alt"
                        dbg["category_winner_debug"] = {"base_category": base_cat, "alt_category": alt_cat, "llm_aggregated": llm_result.get("category"), "decision": "alt", "reason": "llm_prefers_alt"}
                elif llm_choice == "tie":
                    base_score += 1
                    alt_score += 1
                    winners["category"] = 2
                    if dbg is not None:
                        dbg["category_reason"] = "llm_tie"
                        dbg["category_winner_debug"] = {"base_category": base_cat, "alt_category": alt_cat, "llm_aggregated": llm_result.get("category"), "decision": "tie", "reason": "llm_tie"}
                else:
                    if dbg is not None:
                        dbg["category_reason"] = "llm_abstain"
                        dbg["category_winner_debug"] = {"base_category": base_cat, "alt_category": alt_cat, "llm_aggregated": llm_result.get("category"), "decision": "abstain", "reason": "llm_abstain"}
            elif dbg is not None:
                dbg["category_reason"] = "llm_no_choice"
                dbg["category_winner_debug"] = {"base_category": base_cat, "alt_category": alt_cat, "llm_aggregated": llm_result.get("category") if llm_result else None, "decision": "abstain", "reason": "llm_no_choice"}
        elif llm_result is None:
            # LLM unavailable (e.g. 402 Payment Required). Use primary-category fallback so category isn't always 0.
            base_cat = _first_str(base.get("category"))
            alt_cat = _first_str(other.get("category"))
            if base_cat and alt_cat:
                if base_cat.lower() == alt_cat.lower():
                    base_score += 1
                    alt_score += 1
                    winners["category"] = 2
                    if dbg is not None:
                        dbg["category_reason"] = "primary_match_llm_unavailable"
                        dbg["category_winner_debug"] = {"base_category": base_cat, "alt_category": alt_cat, "decision": "tie", "reason": "primary_match_llm_unavailable"}
                else:
                    if dbg is not None:
                        dbg["category_reason"] = "llm_returned_none"
                        dbg["category_winner_debug"] = {"base_category": base_cat, "alt_category": alt_cat, "decision": "abstain", "reason": "llm_returned_none_check_token_or_api"}
    # Category: no website/LLM evidence — abstain (do not use source-vs-source agreement)
    if winners.get("category") is None:
        winners["category"] = 0
        if dbg is not None and (base.get("category") or other.get("category")):
            dbg["category_reason"] = dbg.get("category_reason") or "no_website_evidence"

    # --- Step 4: Phones, address — only from website evidence (not source-vs-source) ---
    base_phones = _list_strs(base.get("phones"))
    alt_phones = _list_strs(other.get("phones"))
    best_phone = None
    for url, fields in extracted_per_url.items():
        for p, _ in fields.get("phones", []):
            if p:
                best_phone = p
                break
        if best_phone:
            break
    if best_phone:
        base_match = any(_phones_equivalent(b, best_phone) for b in base_phones)
        alt_match = any(_phones_equivalent(a, best_phone) for a in alt_phones)
        if base_match and not alt_match:
            base_score += 1
            winners["phones"] = 1
            if dbg is not None:
                dbg["phones_reason"] = "website_evidence_matches_base"
                dbg["evidence_phone"] = best_phone
        elif alt_match and not base_match:
            alt_score += 1
            winners["phones"] = -1
            if dbg is not None:
                dbg["phones_reason"] = "website_evidence_matches_alt"
                dbg["evidence_phone"] = best_phone
        elif base_match and alt_match:
            base_score += 1
            alt_score += 1
            winners["phones"] = 2
            if dbg is not None:
                dbg["phones_reason"] = "website_evidence_matches_both"
                dbg["evidence_phone"] = best_phone
        elif dbg is not None:
            dbg["phones_reason"] = "website_evidence_inconclusive"
            dbg["evidence_phone"] = best_phone
    else:
        winners["phones"] = 0
        if dbg is not None:
            dbg["phones_reason"] = "no_website_evidence"

    # Address — only from website evidence
    base_addr = _first_str(base.get("address")) or str(base.get("address") or "").strip()
    alt_addr = _first_str(other.get("address")) or str(other.get("address") or "").strip()
    best_addr = None
    for url, fields in extracted_per_url.items():
        for a, _ in fields.get("address", []):
            if a:
                best_addr = a
                break
        if best_addr:
            break
    if best_addr:
        base_match = best_addr and base_addr and base_addr.lower() in best_addr.lower()
        alt_match = best_addr and alt_addr and alt_addr.lower() in best_addr.lower()
        if base_match and not alt_match:
            base_score += 1
            winners["address"] = 1
            if dbg is not None:
                dbg["address_reason"] = "website_evidence_matches_base"
                dbg["evidence_address"] = best_addr[:100]
        elif alt_match and not base_match:
            alt_score += 1
            winners["address"] = -1
            if dbg is not None:
                dbg["address_reason"] = "website_evidence_matches_alt"
                dbg["evidence_address"] = best_addr[:100]
        elif base_match and alt_match:
            base_score += 1
            alt_score += 1
            winners["address"] = 2
            if dbg is not None:
                dbg["address_reason"] = "website_evidence_matches_both"
                dbg["evidence_address"] = best_addr[:100]
        elif dbg is not None:
            dbg["address_reason"] = "website_evidence_inconclusive"
            dbg["evidence_address"] = best_addr[:100]
    else:
        winners["address"] = 0
        if dbg is not None:
            dbg["address_reason"] = "no_website_evidence"

    # --- Step 5: Final label (if both same → base; if one slightly better → select it) ---
    if base_score >= alt_score + 2:
        label = 0
        label_reason = "base_score >= alt_score + 2"
    elif alt_score >= base_score + 2:
        label = 1
        label_reason = "alt_score >= base_score + 2"
    elif base_score > alt_score:
        label = 0
        label_reason = "base_slightly_better"
    elif alt_score > base_score:
        label = 1
        label_reason = "alt_slightly_better"
    else:
        # Tie (base_score == alt_score): prefer alt (conflated) when evidence is equivocal
        label = 1
        label_reason = "tie_prefer_alt"

    out = dict(row)
    out["label"] = label
    out["base_score"] = base_score
    out["alt_score"] = alt_score
    out["attr_winners"] = winners
    out["evidence_used"] = bool(extracted_per_url)
    out["row_confidence"] = 0.6 if extracted_per_url else None
    out["base_website_accessible"] = base_accessible
    out["alt_website_accessible"] = alt_accessible
    if llm_result:
        out["llm_aggregated"] = llm_result
    if dbg is not None:
        dbg["label_reason"] = label_reason
        dbg["extracted_evidence"] = {url: {k: [v[0] for v in (vals or [])[:2]] for k, vals in (fields or {}).items()} for url, fields in extracted_per_url.items()}
        out["debug"] = dbg
    return out


def _minimal_output(out: dict) -> dict:
    """Extract minimal fields: id, base_id, base, other, label, base_score, alt_score."""
    return {
        "id": out.get("id"),
        "base_id": out.get("base_id"),
        "base": out.get("base"),
        "other": out.get("other"),
        "label": out.get("label"),
        "base_score": out.get("base_score"),
        "alt_score": out.get("alt_score"),
    }


def _run_debug_analysis(debug_path: Path, golden_path: Path) -> None:
    """Print accuracy analysis by invoking scripts/analyze_debug_output.py."""
    import subprocess
    import sys

    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "scripts" / "analyze_debug_output.py"
    if not script.exists():
        return
    subprocess.run(
        [sys.executable, str(script), "--debug", str(debug_path), "--golden", str(golden_path)],
        cwd=str(project_root),
        check=False,
    )


def _load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file. Handles both single-line and pretty-printed multi-line objects."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    records = []
    i = 0
    while i < len(lines):
        line = lines[i]
        try:
            obj = json.loads(line)
            records.append(obj)
            i += 1
        except json.JSONDecodeError:
            accum = line
            i += 1
            while i < len(lines):
                accum += lines[i]
                i += 1
                try:
                    obj = json.loads(accum)
                    records.append(obj)
                    break
                except json.JSONDecodeError:
                    continue
            else:
                break
    return records


def run_flow(
    input_path: Path,
    output_path: Path,
    *,
    limit: int | None = None,
    delay: float = 0.5,
    llm_model: str | None = None,
    debug: bool = False,
) -> int:
    """
    Run unified agentic flow on rows.
    Always writes minimal output (id, base_id, base, other, label, base_score, alt_score)
    to output_path (agentic_labels.jsonl).
    When debug=True, also writes full output to agentic_labels_debug.jsonl and
    agentic_labels_debug_pretty.json, and prints debug analysis to terminal.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _load_jsonl(input_path)
    if limit:
        rows = rows[:limit]
    count = 0
    full_outputs: list[dict] = []  # for debug files when debug=True

    with open(output_path, "w", encoding="utf-8") as fout:
        for row in rows:
            out = process_row(row, delay=delay, use_llm=True, llm_model=llm_model, debug=debug)
            minimal = _minimal_output(out)
            fout.write(json.dumps(minimal, default=str) + "\n")
            count += 1
            if debug:
                full_outputs.append(out)

    if debug and full_outputs:
        out_dir = output_path.parent
        debug_jsonl = out_dir / "agentic_labels_debug.jsonl"
        debug_pretty = out_dir / "agentic_labels_debug_pretty.json"
        with open(debug_jsonl, "w", encoding="utf-8") as f:
            for o in full_outputs:
                f.write(json.dumps(o, default=str) + "\n")
        with open(debug_pretty, "w", encoding="utf-8") as f:
            json.dump(full_outputs, f, indent=2, default=str)
        print(f"Wrote full debug output to {debug_jsonl} and {debug_pretty}")
        project_root = Path(__file__).resolve().parents[1]
        golden_path = project_root / "inspection" / "golden" / "golden_dataset.json"
        _run_debug_analysis(debug_jsonl, golden_path)

    return count


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Unified row-by-row agentic flow. Input: data/agentic_input.jsonl, Output: agentic_labels.jsonl"
    )
    parser.add_argument(
        "--input",
        default="data/agentic_input.jsonl",
        help="Input JSONL path (default: data/agentic_input.jsonl)",
    )
    parser.add_argument(
        "--out",
        default="out/agentic_labels.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max rows")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between fetches")
    parser.add_argument("--llm-model", default=None, help="LLM model (default: from HF_MODEL or provider default)")
    parser.add_argument("--debug", action="store_true", help="Include debug reasoning per row")
    args = parser.parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input not found: {input_path}")
        return 1
    count = run_flow(
        input_path,
        Path(args.out),
        limit=args.limit,
        delay=args.delay,
        llm_model=args.llm_model or None,
        debug=args.debug,
    )
    print(f"Wrote {count} labeled rows to {args.out}")
    return 0


if __name__ == "__main__":
    exit(main())
