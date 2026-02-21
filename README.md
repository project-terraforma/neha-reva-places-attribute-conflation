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
│   └── sampledata.parquet          # Additional sample data
├── analysis/
│   └── inspection/
│       ├── golden/                 # Golden labeling dataset (JSON, 200 records)
│       ├── side_by_side/            # Main side-by-side sample
│       └── attributes/             # Per-attribute pair samples
├── scripts/
│   ├── inspect_parquet.py         # Dataset overview & stats (DuckDB)
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

**Output layout:**
- `analysis/inspection/golden/` — golden labeling dataset
- `analysis/inspection/side_by_side/` — main side-by-side sample
- `analysis/inspection/attributes/` — per-attribute pair samples (JSON only)

---

## Data Schema

Each row is a pre-matched pair. Columns without a prefix come from the **conflated** record; columns with the `base_` prefix come from the **base** (original) place record.

| Column | Type | Description |
|--------|------|-------------|
| `id` | VARCHAR | Conflated record ID |
| `base_id` | VARCHAR | Base place record ID |
| `sources` | VARCHAR | JSON array of contributing sources (e.g., meta, msft) |
| `names` | VARCHAR | Conflated names (JSON: `primary`, `alternate`) |
| `base_names` | VARCHAR | Base names |
| `categories` | VARCHAR | Conflated categories (e.g., `shipping_center`, `post_office`) |
| `base_categories` | VARCHAR | Base categories |
| `confidence` | DOUBLE | Conflation confidence score |
| `base_confidence` | DOUBLE | Base record confidence |
| `websites` | VARCHAR | Website URLs |
| `base_websites` | VARCHAR | Base websites |
| `socials` | VARCHAR | Social media links |
| `base_socials` | VARCHAR | Base socials |
| `emails` | INTEGER | Email count (often sparse) |
| `base_emails` | VARCHAR | Base emails |
| `phones` | VARCHAR | Phone numbers |
| `base_phones` | VARCHAR | Base phones |
| `brand` | VARCHAR | Brand info |
| `base_brand` | VARCHAR | Base brand |
| `addresses` | VARCHAR | Conflated address (JSON: freeform, locality, region, etc.) |
| `base_addresses` | VARCHAR | Base addresses |
| `base_sources` | VARCHAR | Base source metadata |

### Key Concepts

- **Base record** — The original place from one dataset (e.g., Microsoft); has `base_*` columns.
- **Conflated record** — The merged result, combining attributes from multiple sources; non-prefixed columns.
- **Confidence** — Indicates how reliable the conflation is. Base confidence is typically ~0.77; conflated confidence is often higher (0.95–1.0) when multiple sources agree.

---

## Schema Reference

Overture Places schema (field types, structure, and definitions):

**[Overture Places Schema](https://docs.overturemaps.org/schema/reference/places/place/)**
