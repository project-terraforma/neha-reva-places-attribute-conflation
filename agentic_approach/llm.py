"""
LLM integration for aggregating website keywords and identifying better categories/descriptions.

Uses Hugging Face InferenceClient only. Set HF_TOKEN. See agentic_approach/LLM_SETUP.md.
"""

import json
import os
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
1. **category**: A single, concise business category (e.g., "shipping_center", "automotive_repair", "restaurant"). Use snake_case.
2. **description**: A 1-2 sentence description of what this business does.
3. **keywords**: A JSON array of 5-10 key terms that describe this business (e.g., ["shipping", "mail", "packaging"]).

Respond with valid JSON only, no markdown:
{{"category": "...", "description": "...", "keywords": [...]}}"""

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
    model: str | None = None,
    api_key: str | None = None,
    provider: str | None = None,
) -> str | None:
    """
    Use LLM to pick the better category between base and alt, given aggregated website info.

    Returns "base", "alt", or None (abstain).
    """
    if not llm_aggregated:
        return None
    client, effective_model = _get_client(provider=provider, model=model, api_key=api_key)
    if not client:
        return None

    agg_cat = llm_aggregated.get("category", "")
    agg_desc = llm_aggregated.get("description", "")
    keywords = llm_aggregated.get("keywords", [])

    prompt = f"""Given website-extracted information about a business, which category is more accurate?

Website-extracted category: {agg_cat}
Website-extracted description: {agg_desc}
Website keywords: {keywords}

Base (original) record category: {base_category}
Alt (conflated) record category: {alt_category}

Which is more accurate? Respond with exactly one word: "base", "alt", or "tie" if they are equivalent."""

    try:
        response = client.chat.completions.create(
            model=effective_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        text = response.choices[0].message.content.strip().lower()
        if "base" in text and "alt" not in text:
            return "base"
        if "alt" in text and "base" not in text:
            return "alt"
        if "tie" in text:
            return "base"  # User said: if both same, select base
        return None
    except Exception:
        return None
