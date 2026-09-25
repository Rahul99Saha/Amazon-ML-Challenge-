# Business Entity Resolution Challenge (Amazon ML Challenge 2026)

A high-performance machine learning challenge on **Multi-Source Business Entity Resolution (Record Linkage)** across large-scale, noisy, heterogeneous enterprise data sources.

---

## 1. Executive Summary & Challenge Objective

In large-scale e-commerce and commercial ecosystems, business identity data originates from multiple independent data channels, ingestion feeds, and vendor registries. Each source provides partial, noisy, and unstandardized fragments of information without shared primary keys.

**The Core Task**: Given deduplicated reference business records in **Source 1 (`S1-*`)**, identify and link all corresponding records from **Source 2 (`S2-*`)** and **Source 3 (`S3-*`)** that represent the exact same real-world business entity. A Source 1 entity may match zero (singleton), one, or multiple records across Source 2 and Source 3.

---

## 2. Directory Structure & Verified Dataset Statistics

The uploaded challenge resources are organized under `student_resource/`:

```text
amazon_hack/
├── README.md                               # Comprehensive challenge documentation & guidelines
├── Problem_Statement.md                    # Official problem statement & specifications
├── student_resource/
│   ├── Documentation_template.md           # Methodology write-up template for final submission
│   ├── README.md                           # Original challenge brief
│   ├── dataset/
│   │   ├── train/
│   │   │   ├── train_source1.tsv           # Source 1 reference records (2,206,821 rows | 200.3 MB)
│   │   │   ├── train_source2.tsv           # Source 2 candidate records (5,034,616 rows | 466.6 MB)
│   │   │   ├── train_source3.tsv           # Source 3 candidate records (5,285,603 rows | 480.4 MB)
│   │   │   └── train_ground_truth.tsv      # S1 to S2/S3 ground truth mappings (2,206,821 rows | 121.1 MB)
│   │   └── test/
│   │       ├── test_source1.tsv            # Source 1 test entities (1,732,544 rows | 166.9 MB)
│   │       ├── test_source2.tsv            # Source 2 test pool (4,887,273 rows | 485.9 MB)
│   │       └── test_source3.tsv            # Source 3 test pool (5,082,316 rows | 482.6 MB)
│   └── utils/
│       └── validate_submission.py          # Strict local submission formatting & constraint validator
```

### 2.1 Dataset Scale & Distribution Summary

| Split | File | Record Count | File Size | Country Breakdown |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | `train_source1.tsv` | **2,206,821** | 200.3 MB | US: 1,323,633 (60.0%) \| India: 883,188 (40.0%) |
| **Train** | `train_source2.tsv` | **5,034,616** | 466.6 MB | US: 3,016,817 (59.9%) \| India: 2,017,799 (40.1%) |
| **Train** | `train_source3.tsv` | **5,285,603** | 480.4 MB | US: 3,170,056 (60.0%) \| India: 2,115,547 (40.0%) |
| **Train** | `train_ground_truth.tsv` | **2,206,821** | 121.1 MB | 123,247 Singletons (5.58%) \| 2,083,574 Linked (94.42%)<br>Total Links: 7,638,365 (S2: 3,693,619 \| S3: 3,944,746) |
| **Test** | `test_source1.tsv` | **1,732,544** | 166.9 MB | India: 809,986 (46.8%) \| US: 663,106 (38.3%) \| **France: 259,452 (15.0%)** |
| **Test** | `test_source2.tsv` | **4,887,273** | 485.9 MB | India: 2,312,565 (47.3%) \| US: 1,871,330 (38.3%) \| **France: 703,378 (14.4%)** |
| **Test** | `test_source3.tsv` | **5,082,316** | 482.6 MB | India: 2,408,799 (47.4%) \| US: 1,939,810 (38.2%) \| **France: 733,707 (14.4%)** |

> [!IMPORTANT]
> **Open-Set Country Generalization (`France`)**:
> While the training data exclusively contains `US` and `India`, the test set introduces a 3rd unseen country: **`France` (~14.5% of test data)**.
> - Do **not** hardcode country filters or one-hot vectors strictly to `{US, India}`.
> - Ensure text normalizers, tokenizers, and character n-gram models generalize across multilingual French diacritics and French postal/address naming conventions.
> - Every test entity in `test_source1.tsv` (including French entities) **must** have an output prediction row.

---

## 3. Data Schema & TSV Ingestion

All files are strictly **Tab-Separated Values (`.tsv`)**. Tabs are mandatory because business names, addresses, and comma-delimited ID lists contain commas and quotes.

### 3.1 Entity Record Schema (`*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`)

| Column Name | Type | Description & Examples |
| :--- | :--- | :--- |
| `entity_id` | `string` | Unique record ID. Prefix identifies source: `S1-*`, `S2-*`, or `S3-*`. |
| `business_name` | `string` | Business name. Contains legal forms, abbreviations, transliterations (e.g. Hindi/Devanagari, French), trade names, or typos. |
| `business_address` | `string` | Address string. Landmark cues, PIN/ZIP codes, varying ordering, missing components. |
| `country` | `string` | Country label (`US`, `India`, `France`). |

### 3.2 Ground Truth Schema (`train_ground_truth.tsv`)

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `source1_entity_id` | `string` | The Source 1 reference ID (`S1-xxxxx`). |
| `matched_entity_ids` | `string` | Comma-separated list of matching `S2-*` and/or `S3-*` IDs (e.g., `S2-00047,S3-00812`). Left empty for singletons. |

### 3.3 Efficient Ingestion in Python

```python
import pandas as pd
import csv

# Pandas ingestion (Always specify sep="\t")
df_s1 = pd.read_csv("student_resource/dataset/train/train_source1.tsv", sep="\t", dtype=str)
df_gt = pd.read_csv("student_resource/dataset/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)

# Or using Python stdlib csv module (Memory efficient / stream processing)
def stream_source(tsv_path):
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            yield row
```

---

## 4. Key Noise Patterns & Data Challenges

1. **Multilingual & Script Variations**:
   - Native scripts (Devanagari, e.g., `राम मार्केटिंग प्राइवेट लिमिटेड`) alongside Latin transliterations (`Ram Marketing Pvt Ltd`).
   - Accented European characters in French addresses (e.g., `é`, `è`, `ê`, `à`, `ç`, `ô`, `St-Germain`, `Boulevard`).
2. **Legal Suffix Inconsistencies**:
   - Variations across sources: `Corp`, `Corporation`, `Inc`, `LLC`, `LLP`, `Pvt Ltd`, `Private Limited`, `SA`, `SAS`, `SARL`.
3. **Address Formats & Landmark References**:
   - Indian addresses frequently feature landmark markers (*"Near SBI ATM"*, *"Opp. Metro Pillar 42"*, *"Khasra No."*, *"Hunsur TQ Mysore Dist"*).
   - US addresses feature directional indicators and highway notations (*"1795 Westchester Drive, High Point, NC"*, *"Mack Rd, Haltom City, TX"*).
   - French addresses feature street types (*"Rue"*, *"Avenue"*, *"Allée"*, *"Code Postal"*).
4. **Singletons**:
   - 5.58% of entities in training have no matching counterpart in Source 2 or Source 3. The model must accurately decide when to output an empty list.

---

## 5. Output Specifications & Formatting Rules

Your pipeline must generate **two tab-separated files** in the `output/` directory:

### 5.1 `output/matching_results.tsv` (Scored Leaderboard File)

Contains the final predicted matches for every Source 1 test entity.

```tsv
source1_entity_id	matched_entity_ids
S1-714132312	S2-192345572,S3-462677478
S1-925783039	S2-681193310
S1-999999999	
```

### 5.2 `output/candidate_pairs.tsv` (Blocking Output)

Contains the candidate pool generated by your blocking stage *immediately before* final classifier scoring.

```tsv
source1_entity_id	candidate_entity_ids
S1-714132312	S2-192345572,S2-998811223,S3-462677478,S3-112233445
S1-925783039	S2-681193310,S3-775321672
S1-999999999	
```

### 5.3 Integrity Constraints Checklist

- **Exact Row Count**: Exactly one row per entity in `test_source1.tsv` (1,732,544 rows).
- **Tab Separator**: One tab delimiter between `source1_entity_id` and the matched IDs column.
- **Valid IDs Only**: No self-matches (`S1-*`). IDs must strictly begin with `S2-` or `S3-` and exist in test Source 2 / Source 3.
- **No Duplicates**: No repeated IDs within a row's comma-separated list; no duplicate `source1_entity_id` rows.
- **Singletons**: If an entity has no match/candidate, leave the second column empty.
- **Subset Invariant**: Every ID in `matching_results.tsv` must be present in `candidate_pairs.tsv`.

---

## 6. Local Validation

Before submitting to the portal, run the official validation script provided in `student_resource/utils/`:

```bash
# Basic structural and format check
python3 student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test

# Full ID-existence check across all 11.7M test entities (requires ~4-6 GB RAM)
python3 student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test \
    --check-ids
```

- **Exit code 0 (`PASS`)**: Format is clean and ready for submission.
- **Exit code 1 (`FAIL`)**: Detailed error diagnostics pointing to malformed rows or constraint violations.

---

## 7. Evaluation Metric: Macro-Averaged $F_{0.5}$

Submissions are evaluated using the macro-averaged **$F_{0.5}$ Score** across all Source 1 entities:

$$F_{0.5} = \frac{(1 + 0.5^2) \times \text{Precision} \times \text{Recall}}{0.5^2 \times \text{Precision} + \text{Recall}} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

### Strategic Metric Properties:
- **$2\times$ Precision Weighting**: False merges (linking two distinct companies) are penalized twice as heavily as false negatives (missed links).
- **Macro-Averaged across Entities**: $F_{0.5}$ is computed per Source 1 entity and then averaged across all $N = 1,732,544$ test entities.
- **Singleton Scoring**:
  - True singleton correctly predicted with no matches: $F_{0.5} = 1.0$.
  - True singleton incorrectly assigned any match: $F_{0.5} = 0.0$.

---

## 8. Final Submission Package Structure

At the conclusion of the challenge, teams submit a single `.zip` file containing code, outputs, and documentation:

```text
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv        # Final test predictions (Leaderboard file)
│   └── candidate_pairs.tsv         # Final blocking candidate pool
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # All Python modules & pipeline code
│       ├── README.md               # End-to-end instructions to run pipeline from raw data
│       └── requirements.txt        # Pinned dependencies
└── Documentation_template.md       # Completed technical methodology write-up
```

---

## 9. Constraints & Fair Play Rules

1. **Self-Contained ML**: All models, features, and blocking heuristics must rely solely on the provided dataset and permitted open-source weights.
2. **Model Parameter Limit**: Final models must use MIT or Apache 2.0 licenses with a maximum parameter size of **up to 8 Billion parameters**.
3. **Strict Prohibition on External Lookups**:
   - ❌ No commercial entity resolution APIs.
   - ❌ No government business registration lookups.
   - ❌ No geocoding/places APIs for address normalization.
   - ❌ No external web search, scraping, or external data augmentation.
   - Any violation results in immediate disqualification.