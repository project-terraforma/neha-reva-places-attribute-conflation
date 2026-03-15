# Rule-Based Approach to Attribute Conflation

This folder contains a rule-based method for labeling the golden dataset. Given base (original) and conflated (merged) attribute values for each place, we apply quality rubrics to decide which representation wins for each attribute, then assign an overall label.

## Overview

For each of four attributes—**phones**, **websites**, **addresses**, and **categories**—we compute a quality score for both the base and conflated (alt) values. The higher-quality value wins that attribute. We require a minimum gap between scores to award a win (avoids ties). The final label is determined by the margin of wins: base wins (0), alt wins (1), or abstain (2).

---

## Rules by Attribute

### Phones

**Quality rubric (higher is better):**

| Condition | Score |
|-----------|-------|
| 10–15 digits (valid US or E.164) | +5 |
| Has `+` and 11–15 digits | +2 |
| Too short (<10 digits) | −5 |
| 10 digits with leading zero (suspicious trunk prefix) | −3 |
| Very short (<9 digits) | −5 |

**Winner:** Requires a quality gap of **≥ 2** to award the point. Otherwise tie.

---

### Websites

**Quality rubric (higher is better):**

| Condition | Score |
|-----------|-------|
| Valid host with TLD | +10 |
| HTTPS | +3 |
| Short/canonical path (≤40 chars) | +2 |
| Known shortener (bit.ly, t.co, etc.) | −6 |
| Missing/invalid | −4 |
| Heavy tracking params (utm_, gclid, fbclid) | −2 |

**Winner:** Requires a quality gap of **≥ 3** to award the point. Otherwise tie.

---

### Addresses

**Completeness rubric (higher is better):**

| Condition | Score |
|-----------|-------|
| Freeform present, length ≥ 8 | +3 |
| Locality present | +2 |
| Country present | +2 |
| Region present | +2 |
| Postcode present, length ≥ 4 | +3 |
| Postcode has dash (ZIP+4) or alphanumeric | +1 |
| Freeform contains a number (street number) | +1 |

**Winner:** Requires a quality gap of **≥ 2** to award the point. Otherwise tie.

---

### Categories

**Quality rubric (higher is better):**

| Condition | Score |
|-----------|-------|
| Primary category present | +4 |
| Primary is generic (business, professional_services, retail, services) | −2 |
| Primary missing/empty | −2 |
| Each alternate category (up to 3) | +1 |

**Winner:** Requires a quality gap of **≥ 2** to award the point. Otherwise tie.

---

## Final Label

After counting wins for base vs alt across all four attributes:

| Condition | Label | Meaning |
|-----------|-------|---------|
| base_score ≥ alt_score + 2 | 0 | Base wins |
| alt_score ≥ base_score + 2 | 1 | Alt (conflated) wins |
| Otherwise | 2 | Abstain (no clear winner) |

The margin of 2 is intentionally conservative: we abstain when the outcome is unclear.

---

## Usage

```bash
# From project root
python rule_based_approach/rule_based_labeling.py
```

Reads `inspection/golden/golden_dataset.json` and updates each record with `scores` (per-attribute quality and winner) and `label` (0, 1, or 2).
