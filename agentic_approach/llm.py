"""
LLM integration for aggregating website keywords and identifying better categories/descriptions.

Uses Hugging Face InferenceClient only. Set HF_TOKEN. See agentic_approach/LLM_SETUP.md.
"""

import json
import os
import re
from typing import Any

try:
    from huggingface_hub import InferenceClient
    _HAS_HF = True
except ImportError:
    _HAS_HF = False


def _get_client(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> tuple[Any, str]:
    """
    Get Hugging Face InferenceClient. Returns (client, model_id).
    Client has chat.completions.create().
    """
    hf_token = api_key or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not _HAS_HF or not hf_token:
        return None, ""

    provider = provider or os.environ.get("HF_PROVIDER", "auto")
    model = model or os.environ.get("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    try:
        client = InferenceClient(
            provider=provider,
            model=model,
            api_key=hf_token,
        )
        return client, model
    except Exception:
        return None, ""


def aggregate_website_keywords(
    html_snippets: list[str],
    business_name: str,
    model: str | None = None,
    api_key: str | None = None,
    provider: str | None = None,
) -> dict[str, Any] | None:
    """
    Use LLM to aggregate keywords from website content and identify best category/description.

    Args:
        html_snippets: List of text snippets extracted from fetched pages.
        business_name: Name of the business for context.
        model: Model to use. Default from env (HF_MODEL).
        api_key: Optional API key override (HF_TOKEN).
        provider: Optional Hugging Face provider (default: HF_PROVIDER or "auto").

    Returns:
        Dict with keys: category, description, keywords; or None if LLM unavailable.
    """
    client, effective_model = _get_client(provider=provider, model=model, api_key=api_key)
    if not client or not html_snippets:
        return None

    combined = "\n\n---\n\n".join(s[:3000] for s in html_snippets if s)[:12000]
    if not combined.strip():
        return None

    prompt = f"""You are analyzing website content for a business to determine its category and description.

Business name: {business_name}

Website content excerpts:
{combined}

Extract and aggregate:
1. **category**: A single, concise primary business category (e.g., "shipping_center", "automotive_repair", "restaurant"). Use snake_case.
2. **alternate_categories**: A JSON array of 5-15 alternate categories that also describe this business. Include as many relevant alternates as you can find (e.g., ["mailbox_center", "post_office", "courier", "packaging", "freight"]).
3. **description**: A 1-2 sentence description of what this business does.
4. **keywords**: A JSON array of 5-10 key terms that describe this business (e.g., ["shipping", "mail", "packaging"]).

Respond with valid JSON only, no markdown:
{{"category": "...", "alternate_categories": [...], "description": "...", "keywords": [...]}}"""

    try:
        response = client.chat.completions.create(
            model=effective_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        text = response.choices[0].message.content.strip()
        # Strip markdown code blocks if present
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return json.loads(text)
    except Exception:
        return None


def compare_categories_with_llm(
    base_category: str,
    alt_category: str,
    llm_aggregated: dict[str, Any] | None,
    base_alternates: list[str] | None = None,
    alt_alternates: list[str] | None = None,
    model: str | None = None,
    api_key: str | None = None,
    provider: str | None = None,
) -> str | None:
    """
    Use LLM to pick the better category between base and alt, given aggregated website info.

    Returns "base", "alt", "tie", or None (abstain).
    """
    if not llm_aggregated:
        return None
    client, effective_model = _get_client(provider=provider, model=model, api_key=api_key)
    if not client:
        return None

    agg_cat = llm_aggregated.get("category", "")
    agg_alternates = llm_aggregated.get("alternate_categories", [])
    agg_desc = llm_aggregated.get("description", "")
    keywords = llm_aggregated.get("keywords", [])

    base_alt_str = ""
    if base_alternates:
        base_alt_str = f"\n  Alternate categories: {base_alternates}"
    alt_alt_str = ""
    if alt_alternates:
        alt_alt_str = f"\n  Alternate categories: {alt_alternates}"

    agg_alt_str = f"\nWebsite-extracted alternate categories: {agg_alternates}" if agg_alternates else ""

    prompt = f"""Given website-extracted information about a business, which category is more accurate?

Website-extracted category: {agg_cat}{agg_alt_str}
Website-extracted description: {agg_desc}
Website keywords: {keywords}

Base (original) record category: {base_category}{base_alt_str}
Alt (conflated) record category: {alt_category}{alt_alt_str}

Which is more accurate? Respond with exactly one word: "base", "alt", or "tie" if they are equivalent."""

    try:
        response = client.chat.completions.create(
            model=effective_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        text = response.choices[0].message.content.strip().lower()
        # Prefer first whole-word match so we handle "base is better than alt" etc.
        for word in ("base", "alt", "tie"):
            if re.search(rf"\b{re.escape(word)}\b", text):
                return word
        return None
    except Exception:
        return None
