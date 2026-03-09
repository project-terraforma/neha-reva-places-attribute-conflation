# Agentic Approach Output Schema

This document explains the output files produced by the agentic pipeline and how to interpret them. Use it when you need to understand the structure, fields, and meaning of the labeled conflation results.

---

## Output Overview


| File                               | When produced           | Purpose                                                     |
| ---------------------------------- | ----------------------- | ----------------------------------------------------------- |
| `rows_adapted.jsonl`               | Step 1 (validate)       | Canonical input format for the flow                         |
| `agentic_labels.jsonl`             | Every flow run          | **Minimal output** — sources, label, scores only            |
| `agentic_labels_debug.jsonl`       | Flow run with `--debug` | **Full output** — includes reasoning, evidence, diagnostics |
| `agentic_labels_debug_pretty.json` | Flow run with `--debug` | Same as debug JSONL, pretty-printed for inspection          |


---

## 1. `rows_adapted.jsonl` (Input to flow)

**Source:** `python -m agentic_approach.validate --input data/project_a_samples.parquet --out data/rows_adapted.jsonl`

**Purpose:** Canonical format for downstream processing. One JSON object per line. Each row is a pre-matched pair: one place (base) vs another source (other/conflated).

### Top-level fields


| Field     | Type   | Description                                                             |
| --------- | ------ | ----------------------------------------------------------------------- |
| `id`      | string | Conflated record ID. Unique identifier for the merged place record.     |
| `base_id` | string | Base (original) place record ID. The source record before conflation.   |
| `base`    | object | Attributes from the **base** record (original dataset, e.g. Microsoft). |
| `other`   | object | Attributes from the **other** record (merged from multiple sources).    |


### `base` and `other` objects

Each contains the same attribute keys. Values may be JSON strings (from parquet), arrays, or null.


| Attribute  | Description                                                                                                             |
| ---------- | ----------------------------------------------------------------------------------------------------------------------- |
| `name`     | Place name. Often JSON: `{"primary":"...", "alternate":[...]}`.                                                         |
| `address`  | Address. Often JSON array: `[{"freeform":"...", "locality":"...", "region":"...", "country":"...", "postcode":"..."}]`. |
| `phones`   | Phone numbers. Often JSON array: `["+19049989600"]`.                                                                    |
| `website`  | Website URLs. Often JSON array: `["https://example.com/"]`.                                                             |
| `category` | Business category. Often JSON: `{"primary":"...", "alternate":[...]}`.                                                  |
| `brand`    | Brand info. Often JSON: `{"names":{}}` or null.                                                                         |
| `socials`  | Social media URLs. JSON array or null.                                                                                  |
| `email`    | Email addresses. Usually null or sparse.                                                                                |


---

## 2. `agentic_labels.jsonl` (Minimal output — default)

**Source:** `python -m agentic_approach.flow --input data/rows_adapted.jsonl --out out/agentic_labels.jsonl`

**Purpose:** The primary output for downstream use. Contains only the conflation decision and scores — no debug or diagnostic fields. Written on every flow run.

### Fields


| Field        | Type   | Description                                                                    |
| ------------ | ------ | ------------------------------------------------------------------------------ |
| `id`         | string | Conflated record ID (same as input).                                           |
| `base_id`    | string | Base record ID (same as input).                                                |
| `base`       | object | Base source attributes (name, address, phones, website, category, etc.).       |
| `other`      | object | Other/alt source attributes.                                                   |
| `label`      | number | **Conflation decision:** `0` = base wins, `1` = alt wins, `2` = abstain (tie). |
| `base_score` | number | Count of attributes where base was preferred (0–5).                            |
| `alt_score`  | number | Count of attributes where alt was preferred (0–5).                             |


### Interpreting the label


| Value | Meaning                                                                                   |
| ----- | ----------------------------------------------------------------------------------------- |
| `0`   | **Base wins** — The base record has better or more accurate attributes overall.           |
| `1`   | **Alt wins** — The other/conflated record has better or more accurate attributes overall. |
| `2`   | **Abstain** — Tie or inconclusive. Neither source clearly wins.                           |


### Interpreting scores

- `base_score` + `alt_score` typically sum to at most 5 (one attribute per: name, phones, website, address, category).
- Higher `base_score` → more attributes favored base; higher `alt_score` → more favored alt.
- When `base_score > alt_score` by 2+ → label is usually `0` (base). When `alt_score > base_score` by 2+ → label is usually `1` (alt). Otherwise → often `2` (abstain).

---

## 3. `agentic_labels_debug.jsonl` (Full output — debug mode)

**Source:** `python -m agentic_approach.flow --input data/rows_adapted.jsonl --debug`

**Purpose:** Full output including reasoning, evidence, and diagnostics. Use for debugging, analysis, and understanding why a particular label was chosen. Only produced when `--debug` is passed.

### Fields (in addition to minimal output)

All fields from `agentic_labels.jsonl` plus:


| Field                     | Type          | Description                                                                                                                           |
| ------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `attr_winners`            | object        | Per-attribute winner: `name`, `phones`, `website`, `address`, `category` → `1` (base), `-1` (alt), `0` (none), `2` (both equivalent). |
| `evidence_used`           | boolean       | Whether website evidence was available for this row.                                                                                  |
| `row_confidence`          | number | null | Confidence in the evidence (0–1). Null if no evidence.                                                                                |
| `base_website_accessible` | boolean       | Whether the base website URL was reachable.                                                                                           |
| `alt_website_accessible`  | boolean       | Whether the alt website URL was reachable.                                                                                            |
| `llm_aggregated`          | object | null | When LLM was used: `category`, `description`, `keywords` from website content.                                                        |
| `debug`                   | object        | Detailed reasoning and diagnostics (see below).                                                                                       |


### `attr_winners` values


| Value | Meaning                                                        |
| ----- | -------------------------------------------------------------- |
| `1`   | Base won this attribute.                                       |
| `-1`  | Alt won this attribute.                                        |
| `0`   | No winner — inconclusive or no evidence.                       |
| `2`   | Both equivalent — both sources have correct/equivalent values. |


### `debug` object


| Field                     | Description                                                                  |
| ------------------------- | ---------------------------------------------------------------------------- |
| `name_reason`             | Why name was judged (e.g. `equivalent`, `base_more_info`).                   |
| `phones_reason`           | Why phones were judged (e.g. `equivalent`, `evidence_matches_base`).         |
| `address_reason`          | Why address was judged (e.g. `equivalent`, `both_differ_no_evidence`).       |
| `website_reason`          | Why website was judged (e.g. `both_accessible_name_tie_both`, `base_only`).  |
| `category_reason`         | Why category was judged (e.g. `llm_prefers_base`, `different_no_llm`).       |
| `label_reason`            | Why the final label was chosen (e.g. `base_slightly_better`, `tie_abstain`). |
| `step1_accessibility`     | URLs tried and which were accessible.                                        |
| `name_comparison`         | Parsed names and comparison result.                                          |
| `category_winner_debug`   | Base vs alt category, LLM result, and decision.                              |
| `category_llm_diagnostic` | Whether LLM was used, snippets count, result.                                |
| `extracted_evidence`      | Per-URL extracted fields (name, phones, address, etc.) from fetched pages.   |


---

## 4. `agentic_labels_debug_pretty.json` (Pretty-printed debug)

**Source:** Same as `agentic_labels_debug.jsonl` — produced when `--debug` is used.

**Purpose:** Human-readable version of the debug output. Same content as `agentic_labels_debug.jsonl`, but formatted as a single JSON array with indentation for easier inspection in an editor or browser.

---

## 5. `evidence.jsonl` (Legacy 3-step pipeline)

**Source:** `python -m agentic_approach.evidence --input data/rows_adapted.jsonl --out out/evidence.jsonl`

**Purpose:** Evidence gathered from websites (and optionally search API). No winner/label decision. Used only in the legacy 3-step pipeline.

### Top-level fields


| Field                | Type   | Description                                                               |
| -------------------- | ------ | ------------------------------------------------------------------------- |
| `id`                 | string | Conflated record ID.                                                      |
| `base_id`            | string | Base record ID.                                                           |
| `evidence_sources`   | array  | List of URLs fetched and their metadata.                                  |
| `best_evidence`      | object | Best value per field, with conflicts.                                     |
| `evidence_tier_used` | string | `"A"` = direct website fetch, `"B"` = search API, `"none"` = no evidence. |
| `row_confidence`     | number | 0–1 confidence in the evidence.                                           |


---

## Quick reference: label and score logic

1. **Per-attribute comparison:** For each of name, phones, website, address, category, the flow compares base vs alt (using website evidence and optionally LLM for category).
2. **Scores:** `base_score` = count of attributes where base won; `alt_score` = count where alt won.
3. **Label:** If `base_score >= alt_score + 2` → label `0`. If `alt_score >= base_score + 2` → label `1`. Otherwise → label `2` (abstain).

---

## See also

- **README.md** — Project overview and run instructions.

