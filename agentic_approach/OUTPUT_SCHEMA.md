# Agentic Approach JSONL Output Schema

This document explains each field in the JSONL outputs produced by the agentic pipeline.

---

## 1. `rows_adapted.jsonl` (Step 1: validate)

**Source:** `python -m agentic_approach.validate --input data/project_a_samples.parquet --out data/rows_adapted.jsonl`

**Purpose:** Canonical format for downstream evidence gathering. One JSON object per line.

### Top-level fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Conflated record ID. Unique identifier for the merged place record. |
| `base_id` | string | Base (original) place record ID. The source record before conflation. |
| `base` | object | Attributes from the **base** record (original dataset, e.g. Microsoft). |
| `other` | object | Attributes from the **other/conflated** record (merged from multiple sources). |

### `base` and `other` objects

Each contains the same attribute keys. Values may be JSON strings (from parquet), arrays, or null.

| Attribute | Description |
|-----------|-------------|
| `name` | Place name. Often JSON: `{"primary":"...", "alternate":[...]}`. |
| `address` | Address. Often JSON array: `[{"freeform":"...", "locality":"...", "region":"...", "country":"...", "postcode":"..."}]`. |
| `phones` | Phone numbers. Often JSON array: `["+19049989600"]`. |
| `website` | Website URLs. Often JSON array: `["https://example.com/"]`. |
| `category` | Business category. Often JSON: `{"primary":"...", "alternate":[...]}`. |
| `brand` | Brand info. Often JSON: `{"names":{}}` or null. |
| `socials` | Social media URLs. JSON array or null. |
| `email` | Email addresses. Usually null or sparse. |

---

## 2. `evidence.jsonl` (Step 2: evidence)

**Source:** `python -m agentic_approach.evidence --input data/rows_adapted.jsonl --out out/evidence.jsonl`

**Purpose:** Evidence gathered from websites (and optionally search API). No winner/label decision.

### Top-level fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Conflated record ID (same as rows_adapted). |
| `base_id` | string | Base record ID (same as rows_adapted). |
| `evidence_sources` | array | List of URLs fetched and their metadata. |
| `best_evidence` | object | Best value per field, with conflicts. |
| `evidence_tier_used` | string | `"A"` = direct website fetch, `"B"` = search API, `"none"` = no evidence. |
| `row_confidence` | number | 0–1 confidence in the evidence. Capped at 0.6 if critical fields conflict. |

### `evidence_sources` array

Each element describes one URL that was fetched:

| Field | Type | Description |
|-------|------|-------------|
| `url` | string | Final URL after redirects. |
| `status` | number | HTTP status code (200, 403, etc.). |
| `classification` | string | `"relevant"` = page matches business; `"parked"` = domain for sale; `"irrelevant"` = wrong page; `"blocked"` = 403/captcha/rate limit. |
| `tier` | string | `"A"` = direct website fetch. |
| `tier_score` | number | 1.0 for tier A; 0.7 for tier B. |

### `best_evidence` object

Keys are field names: `name`, `phones`, `address`, `category`, `emails`, `socials`.

Each field value is an object:

| Field | Type | Description |
|-------|------|-------------|
| `value` | string | Best value for this field (highest score). |
| `score` | number | tier_score × extraction_confidence. |
| `tier` | string | Source tier: `"A"` or `"B"`. |
| `conflicts` | array | Alternative values with scores within 10% of best. |

### `conflicts` array (inside each field)

Each conflict object:

| Field | Type | Description |
|-------|------|-------------|
| `value` | string | Alternative value for the field. |
| `score` | number | Score for this alternative. |
| `tier` | string | Source tier. |

### `evidence_tier_used`

| Value | Meaning |
|-------|---------|
| `"A"` | Evidence came from direct website fetches. |
| `"B"` | Evidence came from search API fallback (tier B). |
| `"none"` | No evidence gathered (all URLs failed or no URLs). |

### `row_confidence`

- **Range:** 0.0–1.0
- **Formula:** average(best_evidence scores) × completeness (fields with data / 6)
- **Cap:** If `name`, `phones`, or `address` have conflicts, confidence is capped at 0.6.
