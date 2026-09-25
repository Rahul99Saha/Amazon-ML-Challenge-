# Business Entity Resolution: Progress Log

_Last updated: 2026-09-25_

## 1. The task

For every **Source 1** (S1) business record, find all matching records in **Source 2** (S2) and **Source 3** (S3). The score is **F0.5 averaged per S1 record** (macro), so precision counts twice as much as recall.

- An S1 record with no true match scores 1.0 if we predict nothing, and 0.0 if we predict anything.
- The training data covers the US and India. The test data adds **France**, which never appears in training.

Two constraints on the solution:
- Final models must be MIT or Apache-2.0 licensed and have at most 8B parameters.
- No external lookups of any kind (APIs, geocoding, registries).

## 2. What the data looks like

| Split | S1 | S2 | S3 |
|---|---|---|---|
| Train | 2,206,821 | 5,034,616 | 5,285,603 |
| Test | 1,732,544 (15% France) | 4,887,273 | 5,082,316 |

Findings from exploring the training data:

- **5.6% of S1 records have no match.** Most have 2–5 matches (3.5 on average), split across S2 and S3.
- **No S2/S3 record matches more than one S1 record.** A strict one-to-one assignment is therefore valid.
- **About 26% of S2/S3 records match nothing.** They act as distractors.
- **Name noise:**
  - Names written in native scripts: Devanagari, Tamil, Kannada, Gujarati, Odia
  - Junk prefixes such as `***`, `>>`, `@`, `[..]`
  - Website and handle forms, sometimes with the words reordered (`technologiesdynamicroyal.com`)
  - "X doing business as Y" and "formerly known as" wrappers
  - OCR-style typos (0/o, 1/l, 8/B), injected accents, repeated words
  - Some matches carry a completely unrelated trade name (e.g. "Beloorbigild") at the same address
- **Address noise:**
  - Components in a different order
  - `##` markers, and house numbers with digits dropped (3195 → 319)
  - Ordinals written as words ("2nd" → "SECOND")
  - City aliases (Bombay/Mumbai), full vs abbreviated state names, "N/A" and "NULL" placeholders
  - About 20% of the pool records we missed have no address at all

## 3. Pipeline built so far

Code lives in `business_entity_resolution/src/ber/`.

```
raw TSV ──► prepare (normalise + transliterate) ──► blocking (inverted index, top-200)
        ──► LightGBM re-ranker (top-20 kept) ──► context features ──► matcher (LightGBM
        [+ cross-encoder score]) ──► decision rule (one-to-one + threshold / expected F0.5)
        ──► matching_results.tsv + candidate_pairs.tsv ──► official validator
```

| Module | What it does |
|---|---|
| `io.py` | Loads the TSVs with an explicit tab separator and no quoting, and caches them as parquet. Paths can be overridden with `BER_DATA`, `BER_WORK` and `BER_OUTPUT` (used on Kaggle). |
| `normalize.py` | Vectorized with polars (about 1 minute for 22M records). Converts every script to ASCII (anyascii), handles DBA and "formerly known as" wrappers, strips websites and handles, separates legal suffixes, fixes OCR-style digits, expands address abbreviations (English, Indian and French), maps ordinals, city aliases and region names, and extracts address numbers. |
| `translit.py` | Learns a dictionary from transliterated tokens to Latin tokens, using only the training pairs. It found **14,594 mappings**, e.g. `praivet`→private, `bildrs`→builders, `phaumdesn`→foundation. |
| `prepare.py` | Normalizes all six files once and assigns each record an integer index. |
| `blocking.py` | An inverted index weighted by IDF over 8 key types: name token, compact-name prefix (6 and 10 characters), house number × rare street word, rare street word, **pairs of name tokens**, **pairs of address words**, and name token × address word. Keys are hashed to 64-bit integers and cached on disk, and keys shared by more than 300 records are dropped. |
| `features.py` | About 37 pair features: rapidfuzz ratio / token_set / token_sort / partial / Jaro-Winkler on names and addresses, token and number Jaccard, IDF coverage of the name, agreement of legal suffixes, empty-address flags, native-script flag, source flag, lengths, and blocking signals. |
| `rerank_dev.py` | Trains the LightGBM re-ranker that prunes blocking output. |
| `candidates.py` | Full-scale candidate generation in chunks, writing to disk as it goes and able to resume where it stopped. |
| `context.py` | Features that need the whole candidate set: an S1 record's rank, its gap to the best candidate and its number of strong candidates; **competition** (how many S1 records claim this pool record, and how the best competing S1 scores it); and **cluster consistency** (the candidate's similarity to the S1 record's other top candidates). |
| `matcher.py` | Out-of-fold LightGBM over the full train split, picks the decision rule with the exact metric, and a `final` step that predicts test and writes the submission. |
| `decide.py` | One-to-one assignment (each pool record goes to its best S1 record) and selection of each S1 record's match set by expected F0.5. |
| `metric.py` | A local copy of the official macro F0.5, including the singleton rule. |
| `submit.py` | Writes both output TSVs and runs the official `validate_submission.py`. |
| `cross_encoder.py` | Fine-tunes a cross-encoder based on mDeBERTa-v3-base (MIT) on raw text, using our own candidates as hard negatives. It runs with `accelerate` on 2× T4 and **has not been run yet**. |
| `kaggle/build_notebooks.py` | Generates two self-contained Kaggle notebooks with the source code embedded. |

## 3a. How blocking works, step by step

**Why we need it.** Comparing every S1 record with every pool record is impossible: 1.7M test S1 × 10M S2+S3 is about 17 trillion pairs. Blocking picks a small set of plausible candidates for each S1 record, and only those reach the matching models. Candidates it misses can never be matched, so blocking sets the recall ceiling for the whole pipeline.

Blocking runs in two stages: **(A)** an inverted index scored by IDF produces about 200 candidates per S1, then **(B)** a learned re-ranker cuts that to the final 20 (10 once the cross-encoder is used).

### Stage A: IDF-weighted inverted index (`blocking.py`)

**Step 1: every record emits a set of "keys".** Keys are built from the normalized fields (section 3). Each key starts with its **key type and the record's country** (e.g. `np|India|builders|jai`). That means:
- records from different countries never become candidates for each other;
- France or any other new label works automatically, since the country is just a string inside the key.

Each key is then hashed to a 64-bit integer, so the index stores only integers (record index u32, key u64, key type u8).

| Key type | Built from | Example (S1 "Safe Marine Solutions LLC, 33 Westgate Road, Boston, MA") | Catches | Weight |
|---|---|---|---|---|
| `name` | each core name token (legal suffixes and prefixes like "the" or "shri" removed; ≥2 characters), including tokens of the name before a DBA marker | `safe`, `marine`, `solutions` | normal names | 1.0 |
| `namepair` | every **unordered pair** of core name tokens (up to 6 tokens) | `marine\|safe`, `marine\|solutions`, `safe\|solutions` | common words that are rare **together**; reordered names | 1.0 |
| `compact` | first **6** characters of the name with spaces removed ("and" dropped) | `safema` | websites and handles, glued words | 1.0 |
| `compact10` | first **10** characters of the same | `safemarine` | `safemarinesolutions.com`, `#SAFEMARINE` | 1.5 |
| `addr` | each address number (up to 3) × each of the **3 rarest** street words | `33\|westgate` | exact street address | 1.5 |
| `addrword` | each of the 3 rarest street words alone | `westgate` | a house number mangled or missing | 0.5 |
| `addrpair` | the pair of the 2 rarest street words | `boston\|westgate` | address without a number, components reordered | 0.7 |
| `nameaddr` | each core name token × each of the 2 rarest street words | `marine\|westgate` | noisy name, but the address agrees | 0.7 |

**What is removed before building name keys:** there is no general stop-word list, only three small hand-written ones.
- **Legal suffixes**, after unifying synonyms: inc, llc, ltd/limited, pvt/private, corp/corporation, co/company, llp, lp, pc, plc, pllc, pa, sarl, sas, sa, eurl, sasu, sci, snc, selarl, gmbh, ag, opc. They are kept separately in `name_legal`, and the matcher still sees `legal_eq`.
- **Prefixes**: the, shri, sri, shree, m/s, ms, le, la, les, l.
- **Tokens shorter than 2 characters**, plus "and" (dropped from the compact key only).

Every other common word is handled by the frequency limit in step 2.

Measured effect: the core name becomes empty for only **6 of 2.2M** train S1 records (5 in test). The removed tokens are overwhelmingly real legal forms (US `pc` 50k, `lp` 16k; India `company` 26k; test France `sa` 12.8k, `sci` 8.4k). Real name words hit by the lists (`ms` or `sa` in India) number only about 140–250.

Weak spot: **2.5%** of S1 names are left with a single core token. If that token is common, those records rely on compact and address keys only. This is to be checked in the next round of miss analysis.

"Rarest street words" are ranked by how often each word appears in that country's pool. Generic words are skipped ("street", "road", "unit", "floor", "nagar", "rue", …), as are words shorter than 3 characters. On average a pool record emits **18.5 keys** (191M keys for the 10.3M train pool records).

**Step 2: drop keys that are too common.** For every key an S1 record has, count how many pool records share it. This is the key's *block size*, or document frequency (df). Keys with **df > 300** are dropped: "services" or "private" alone say almost nothing, and keeping them would add millions of pairs. The remaining keys get a weight:

```
idf(key) = ln(N_pool / df(key))            N_pool = 10.3M (train) / 10.0M (test)
```

A key shared by only 2 records gets idf ≈ 15.4. A key shared by 300 gets ≈ 10.4.

**Step 3: score candidate pairs.** Every pool record that shares at least one kept key with an S1 record becomes a candidate. Its score is:

```
block_score(q, p) = Σ over shared keys  idf(key) × weight(key type)
```

The number of shared keys of each type (`bk_name`, `bk_addr`, …) is stored too, as features for later. The **top 200** candidates per S1 record by `block_score` go on to stage B.

**Worked example (a miss that pair keys fixed).** S1 "Urology Associates Inc, 6305 Covington Drive, Rowlett TX" and S2 "UROLOGY ASSOCIATES INC" with an **empty address**:
- "urology" and "associates" each appear in more than 300 US pool records, so both single-token keys are dropped. With only single keys, the pair was never generated.
- The pair key `associates|urology` appears in only a few records. It survives with high IDF, and the pair now ranks near the top.

**Engineering, for 10M+ records in under 7 GB of RAM:**
- **Keys are built once per split.** Records are processed in 1M-row slices and the keys cached to parquet (about 2 GB for the train pool).
- **Queries run in super-chunks** of 150k S1 records.
- **Pass 1 streams the pool keys** and computes block sizes only for keys the chunk's queries use. It never holds all pool rows in memory. The first version did, and was killed at the memory cap.
- **Pass 2 loads pool rows only for keys with df ≤ 300.** The chunk's queries are then joined to them in slices of 50k, summed per (S1, pool) pair, and the top 200 kept.
- **Output goes to disk after every super-chunk**, so an interrupted run resumes where it stopped.
- On the full train set it takes **about 6 minutes per 150k S1 records**, roughly 90 minutes in total.

### Stage B: learned re-ranker (`features.py`, `rerank_dev.py`, `candidates.py`)

The `block_score` is a crude sum. Genuine matches often sit below records that share many address keys, which is why recall at top 50 was only 95.4% although 97.85% of true matches were generated somewhere. So for each of the ~200 candidates we compute about 37 cheap features:
- rapidfuzz similarities on names and addresses (ratio, token_set, token_sort, partial, Jaro-Winkler)
- Jaccard overlap of name tokens, address numbers and address words
- IDF coverage of the name
- agreement of legal suffixes, empty-address and native-script flags, S2/S3 source
- lengths, plus `block_score`, `block_rank` and the per-type key counts

A **LightGBM classifier** trained on 40k train S1 records (their top 300 candidates, labeled from the ground truth) scores every candidate. The **top 20** by that score are kept, with all their features. **This top-20 set is exactly what the matcher runs on, and what gets written to `candidate_pairs.tsv`.**

| Candidates per S1 after re-ranking | Recall of true matches |
|---|---|
| top 5 | 91.5% |
| top 10 | 97.1% |
| top 20 | 97.3% |
| everything generated (ceiling) | 97.4% |

About 2.6% of true matches are still never generated at all. Most are native-script names not covered by the transliteration map, website names with the words reordered, or records with an empty address whose name is heavily corrupted. Planned fixes: dense multilingual embedding retrieval on the Kaggle GPU, and splitting glued website/handle names back into words.

## 4. Results so far (held-out train S1 records)

### Candidate generation (blocking)

| Version | Recall at top 50 | Recall of all generated candidates |
|---|---|---|
| Single keys only | 86.5% | – |
| + pair keys | 94.2% | 96.96% (top 400) |
| + transliteration map + 10-character compact key | 95.4% | **97.85%** (top 400) |
| + LightGBM re-ranker | **97.1% at top 10, 97.3% at top 20** | 97.4% (top 300) |

The re-ranker moves almost every true match that blocking finds into the top 10. So the final candidate set is only 10–20 records per S1, about 35M pairs on test.

### End-to-end F0.5 (first baseline, 20k-query sample)

| Setup | Macro F0.5 |
|---|---|
| Re-ranker probability, threshold 0.7 | **0.934** |
| Expected-F0.5 set selection | 0.931 |
| Upper bound: perfect matcher on the top-20 candidates | 0.990 |

**Main error found:** a pool record that shares an S1 record's address but belongs to a *different* business at that address gets a confident match. Its true owner isn't in the 20k sample, so nothing competes for the record. The competition features and the one-to-one rule need all S1 records at once, which is why the full-scale run below was started.

## 5. Infrastructure decisions

- **Memory safety:** every heavy local job goes through `./run.sh`. It runs the job in a systemd scope with a **7 GB hard cap**, no swap and lower CPU priority, so a job that goes over is killed by itself instead of freezing the machine. Long jobs write to disk in chunks and resume where they stopped.
- **Local machine:** GTX 1650 Ti (4 GB), 14 GB RAM, internet at about 70–380 KB/s. Development runs here on the CPU.
- **Kaggle** handles the GPU work and full-scale runs. The notebooks download the dataset directly from the organizers' Google Drive link, so only code and final outputs pass through the slow local connection.
  - **Notebook A (CPU):** prepare → re-ranker → candidates → context features for train and test.
  - **Notebook B (2× T4):** cross-encoder training, then scoring of the top 10 candidates, then the matcher and the submission files.

## 6. In progress right now

- Full train candidate generation over all 2.2M S1 records (stages A and B above): 900k of 2.2M done, roughly 6 minutes per 150k queries.

## 7. Next steps

1. Build context features on the full train set, then run the out-of-fold matcher to get a **realistic full-scale F0.5** with the one-to-one rule.
2. Generate test candidates, then produce and validate the **first leaderboard submission** locally.
3. Run the Kaggle notebooks: cross-encoder score as a matcher feature. I expect this to give the largest gain, especially on native-script records and France.
4. Improve generation recall (97.4% now):
   - dense multilingual bi-encoder retrieval on GPU
   - segmenting glued website/handle names back into words
5. Check how well France will go, using leave-one-country-out validation (US→India, India→US).
6. Final package: `README.md`, pinned `requirements.txt`, filled-in `Documentation_template.md`, zip.

## 8. How to reproduce what exists

```bash
python3.12 -m venv .venv && .venv/bin/pip install polars pyarrow numpy pandas scikit-learn scipy rapidfuzz anyascii lightgbm tqdm
ln -s Dataset/<...>/student_resource/dataset data
./run.sh ber.prepare --splits train test      # normalise; learns + applies transliteration map
./run.sh ber.eval_blocking                    # blocking recall on 30k train queries
./run.sh ber.rerank_dev                       # re-ranker + recall@K report
./run.sh ber.f05_dev                          # baseline macro F0.5 on the dev sample
./run.sh ber.candidates --splits train test   # full-scale candidates (resumable)
./run.sh ber.matcher ctx --split train        # context features
./run.sh ber.matcher oof                      # full-scale OOF F0.5 + decision rule
./run.sh ber.matcher final                    # test predictions -> output/ + validation
```
