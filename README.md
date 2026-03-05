# Places Attribute Conflation

**Project A · Winter 2026 · CRWN 102**

Creating a single reliable record from multiple location sources.

---

## Overview

Real-world places often appear in multiple datasets with inconsistent, outdated, or conflicting information. This project tackles the problem of **attribute-level conflation**: given multiple representations of the same place, how do we decide which attributes (phone, website, email, etc.) are the most accurate?

Our goal is to produce a high-quality golden dataset and evaluate different strategies—rule-based logic vs. machine learning—for selecting the best attributes.

This project is developed as part of coursework at the University of California, Santa Cruz, in partnership with the [Overture Maps Foundation](https://overturemaps.org/), and is motivated by the structure and constraints of the Overture Maps Places dataset.

---

### Data Context

This repository works with **pre-matched pairs** of place records. Each row represents a conflation: one place (the *base*) merged with attributes from other sources to produce a conflated record. We use this data to understand and evaluate how well attributes from different datasets can be combined into a single, trustworthy place entry.

### Team

**Neha Ashwin, Reva Agarwal**

---

## Project Structure

```
neha-reva-places-attribute-conflation/
├── data/
│   ├── project_a_samples.parquet   # Main sample (~2,000 pre-matched pairs)
│   ├── rows_adapted.jsonl          # Adapted rows (JSONL) for agentic flow
│   └── sampledata.parquet          # Additional sample data
├── out/
│   ├── agentic_labels.jsonl        # Minimal output (label, scores, sources)
│   ├── agentic_labels_debug.jsonl  # Full output with debug (--debug only)
│   └── agentic_labels_debug_pretty.json  # Pretty-printed debug output (--debug only)
├── agentic_approach/               # Agentic pipeline (validate, flow)
├── analysis/
│   └── inspection/
│       ├── golden/                 # Golden labeling dataset (CSV, 200 records)
│       ├── side_by_side/            # Main side-by-side sample
│       └── attributes/             # Per-attribute pair samples
├── scripts/
│   ├── inspect_parquet.py         # Dataset overview & stats (DuckDB)
│   ├── create_golden_dataset.py   # Create 200-record golden labeling CSV
│   └── attributes/
│       ├── inspect_attr_pair.py   # Shared logic for attribute-pair inspection
│       ├── inspect_categories.py  # base_categories vs categories
│       ├── inspect_addresses.py   # base_addresses vs addresses
│       ├── inspect_phones.py      # base_phones vs phones
│       └── inspect_websites.py    # base_websites vs websites
├── requirements.txt
└── README.md
```

---

## Exploring the Data

From the project root:

```bash
source overture/bin/activate
python scripts/inspect_parquet.py
```

This prints a dataset overview including:

- **Schema** — All columns and types
- **Row count** — Total records
- **Null counts** — Which attributes are often missing
- **Confidence distribution** — Conflated vs base record confidence
- **Sample rows** — Example key attributes
- **Uniqueness** — `id` and `base_id` cardinality

### Attribute-specific scripts

For focused analysis of each important attribute pair, run:

```bash
python scripts/attributes/inspect_categories.py   # base_categories vs categories
python scripts/attributes/inspect_addresses.py    # base_addresses vs addresses
python scripts/attributes/inspect_phones.py       # base_phones vs phones
python scripts/attributes/inspect_websites.py     # base_websites vs websites
```

Each script prints stats (coverage, comparable count, disagreement rate), value examples, disagreement examples, and exports to `analysis/inspection/attributes/{attr}_pair_sample.json`.

**Golden dataset (CSV):**

```bash
python scripts/create_golden_dataset.py
```

Creates `analysis/inspection/golden/golden_labeling_sample.csv` with 200 records and blank `label_*` / `notes_*` columns for manual review.

**Output layout:**

- `analysis/inspection/golden/` — golden labeling dataset (CSV)
- `analysis/inspection/side_by_side/` — main side-by-side sample
- `analysis/inspection/attributes/` — per-attribute pair samples (JSON only)

---

## Agentic Approach Pipeline

### Unified flow (recommended)

Processes each row in sequence: test website accessibility, compare names, escalate to online search when needed, use LLM for categories, and produce labels.

```bash
# Step 1: Adapt rows (parquet → JSONL)
python -m agentic_approach.validate --input data/project_a_samples.parquet --out data/rows_adapted.jsonl

# Step 2: Unified row-by-row flow (replaces evidence + label steps)
python -m agentic_approach.flow --input data/rows_adapted.jsonl --out out/agentic_labels.jsonl
```

**Note:** `--limit N` limits how many records are processed from the input. If you get fewer rows than expected, the input file may have fewer records (e.g. `data/rows_adapted.jsonl` was created with a small limit). Regenerate with more rows: `python -m agentic_approach.validate --input data/project_a_samples.parquet --limit 10 --out data/rows_adapted.jsonl`

**Output:** Every run writes minimal output to `out/agentic_labels.jsonl` (id, base_id, base, other, label, base_score, alt_score). No debug fields.

**Debug mode:** Add `--debug` to also write full output to `out/agentic_labels_debug.jsonl` and `out/agentic_labels_debug_pretty.json`, and print accuracy analysis vs golden labels to the terminal:

```bash
python -m agentic_approach.flow --input data/rows_adapted.jsonl --debug
```

**Flow per row:**

1. Test website accessibility for base and alt
2. If one works → point to that source; if both → compare names (prefer more info), select website that correlates
3. If neither works → escalate to online search (DuckDuckGo)
4. Fetch website content, compare with base/alt to determine accuracy
5. Use LLM to aggregate keywords and pick better category/description
6. If both same → select base

**Phone comparison:** With/without area code; NOT for leading 0s.

**Options:** `--no-llm` to disable LLM, `--limit N` for testing, `--delay` for fetch spacing, `--debug` for full debug output and analysis.

**LLM setup:** Uses Hugging Face only (free tier).

**Output schema:** See [`agentic_approach/OUTPUT_SCHEMA.md`](agentic_approach/OUTPUT_SCHEMA.md) for field descriptions and output formats.

---

## Data Schema

Each row is a pre-matched pair. Columns without a prefix come from the **conflated** record; columns with the `base_` prefix come from the **base** (original) place record.


| Column            | Type    | Description                                                   |
| ----------------- | ------- | ------------------------------------------------------------- |
| `id`              | VARCHAR | Conflated record ID                                           |
| `base_id`         | VARCHAR | Base place record ID                                          |
| `sources`         | VARCHAR | JSON array of contributing sources (e.g., meta, msft)         |
| `names`           | VARCHAR | Conflated names (JSON: `primary`, `alternate`)                |
| `base_names`      | VARCHAR | Base names                                                    |
| `categories`      | VARCHAR | Conflated categories (e.g., `shipping_center`, `post_office`) |
| `base_categories` | VARCHAR | Base categories                                               |
| `confidence`      | DOUBLE  | Conflation confidence score                                   |
| `base_confidence` | DOUBLE  | Base record confidence                                        |
| `websites`        | VARCHAR | Website URLs                                                  |
| `base_websites`   | VARCHAR | Base websites                                                 |
| `socials`         | VARCHAR | Social media links                                            |
| `base_socials`    | VARCHAR | Base socials                                                  |
| `emails`          | INTEGER | Email count (often sparse)                                    |
| `base_emails`     | VARCHAR | Base emails                                                   |
| `phones`          | VARCHAR | Phone numbers                                                 |
| `base_phones`     | VARCHAR | Base phones                                                   |
| `brand`           | VARCHAR | Brand info                                                    |
| `base_brand`      | VARCHAR | Base brand                                                    |
| `addresses`       | VARCHAR | Conflated address (JSON: freeform, locality, region, etc.)    |
| `base_addresses`  | VARCHAR | Base addresses                                                |
| `base_sources`    | VARCHAR | Base source metadata                                          |


### Key Concepts

- **Base record** — The original place from one dataset (e.g., Microsoft); has `base_`* columns.
- **Conflated record** — The merged result, combining attributes from multiple sources; non-prefixed columns.
- **Confidence** — Indicates how reliable the conflation is. Base confidence is typically ~0.77; conflated confidence is often higher (0.95–1.0) when multiple sources agree.

---

## Schema Reference

Overture Places schema (field types, structure, and definitions):

**[Overture Places Schema](https://docs.overturemaps.org/schema/reference/places/place/)**