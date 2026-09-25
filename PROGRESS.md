# Business Entity Resolution: Progress Log

_Last updated: 2026-09-25 (evening): the pipeline now runs on Kaggle, with the n-gram BM25 channel added_

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
raw TSV ──► prepare (normalise + transliterate)
        ──► candidates = rare-key index (top-200)  ∪  character n-gram BM25 (top-50 per field)
        ──► 43 pair features ──► LightGBM re-ranker (top-20 kept)
        ──► context features ──► matcher (LightGBM [+ cross-encoder score])
        ──► decision rule (one-to-one + threshold / expected F0.5)
        ──► matching_results.tsv + candidate_pairs.tsv ──► official validator
```

| Module | What it does |
|---|---|
| `io.py` | Loads the TSVs with an explicit tab separator and no quoting, and caches them as parquet. Paths can be overridden with `BER_DATA`, `BER_WORK` and `BER_OUTPUT` (used on Kaggle). |
| `normalize.py` | Vectorized with polars (about 1 minute for 22M records). Converts every script to ASCII (anyascii), handles DBA and "formerly known as" wrappers, strips websites and handles, separates legal suffixes, fixes OCR-style digits, expands address abbreviations (English, Indian and French), maps ordinals, city aliases and region names, and extracts address numbers. |
| `translit.py` | Learns a dictionary from transliterated tokens to Latin tokens, using only the training pairs. It found **14,594 mappings**, e.g. `praivet`→private, `bildrs`→builders, `phaumdesn`→foundation. |
| `prepare.py` | Normalizes all six files once and assigns each record an integer index. |
| `blocking.py` | An inverted index weighted by IDF over 8 key types: name token, compact-name prefix (6 and 10 characters), house number × rare street word, rare street word, **pairs of name tokens**, **pairs of address words**, and name token × address word. Keys are hashed to 64-bit integers and cached on disk, and keys shared by more than 300 records are dropped. |
| `ngram.py` | Candidate channel 2: **BM25 over character 3-grams** of the name and of name + full address. Exact top-k search via a sparse matrix product (`sparse_dot_topn`), one index per country, and a BM25 score for every candidate pair. TF-IDF cosine is kept as a switch (`--weighting tfidf`). |
| `features.py` | **43** pair features: rapidfuzz ratio / token_set / token_sort / partial / Jaro-Winkler on names and addresses, token and number Jaccard, IDF coverage of the name, agreement of legal suffixes, empty-address flags, native-script flag, source flag, lengths, blocking signals, **plus 6 n-gram features** (`ng_name`, `ng_name_addr`, their ranks, and the channel flags `from_keys` / `from_ng`). |
| `rerank_dev.py` | Trains the LightGBM re-ranker on **union** candidates of 40k train S1 records, and evaluates on 20k others. It reports generation recall per channel (keys only / n-grams only / union) and recall@K. |
| `candidates.py` | Full-scale candidate generation: union of both channels, then features, then re-ranker, then top 20. Runs **per country** and **per shard** (`--countries`, `--shard i/n`) so each Kaggle job stays under 12 h. Parts go to disk and resume; `--merge` joins them. |
| `context.py` | Features that need the whole candidate set: an S1 record's rank, its gap to the best candidate and its number of strong candidates; **competition** (how many S1 records claim this pool record, and how the best competing S1 scores it); and **cluster consistency** (the candidate's similarity to the S1 record's other top candidates). |
| `matcher.py` | Out-of-fold LightGBM over the full train split, picks the decision rule with the exact metric, and a `final` step that predicts test and writes the submission. |
| `decide.py` | One-to-one assignment (each pool record goes to its best S1 record) and selection of each S1 record's match set by expected F0.5. |
| `metric.py` | A local copy of the official macro F0.5, including the singleton rule. |
| `submit.py` | Writes both output TSVs and runs the official `validate_submission.py`. |
| `cross_encoder.py` | Fine-tunes a cross-encoder based on mDeBERTa-v3-base (MIT) on raw text, using our own candidates as hard negatives. It runs with `accelerate` on 2× T4 and **has not been run yet**. |
| `kaggle/push_job.py` | Launches a pipeline step as a **private Kaggle script job** from the local terminal (Kaggle CLI). The job mounts the code and data, symlinks the outputs of earlier jobs into its work directory, runs the steps, and cleans up. `--wait` / `--log` fetch the status and log. |
| `kaggle/sync_code.sh` | Uploads `src/ber/` as the private Kaggle dataset `ber-code` (a new version on each call). |
| `kaggle/kernels/ber-data/` | One-off Kaggle job that downloads the organizers' zip from Google Drive (1.1 GB in about 30 s). Its output is the data input for every other job. |
| `kaggle/build_notebooks.py` | Older alternative: generates self-contained notebooks for manual upload. Superseded by `push_job.py`. |
| `run.sh` | Local wrapper: runs any module under a 7 GB memory cap (systemd scope, no swap, low priority). |

## 3a. How blocking works, step by step

**Why we need it.** Comparing every S1 record with every pool record is impossible: 1.7M test S1 × 10M S2+S3 is about 17 trillion pairs. Blocking picks a small set of plausible candidates for each S1 record, and only those reach the matching models. Candidates it misses can never be matched, so blocking sets the recall ceiling for the whole pipeline.

Blocking has three parts. **(A)** An inverted index scored by IDF produces up to 200 candidates per S1. **(A2)** A character n-gram BM25 search adds up to 50 per field (union: about 241 per S1). **(B)** A learned re-ranker cuts the union to the final 20 (10 once the cross-encoder is used).

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

### Stage A2: character n-gram BM25 channel (`ngram.py`), added on Kaggle

A second, rule-free candidate channel, merged with Stage A as a union before the re-ranker:

- **Text:** character **3-grams** of (a) the core name with spaces removed (`name`), and (b) the core name plus the **full** normalized address (`name_addr`). Generic address words like "street" or "road" are deliberately **kept**; common n-grams are handled by weighting alone.
- **Weighting: BM25** (`k1 = 1.2`, `b = 0.75`, `idf = ln((N − df + 0.5)/(df + 0.5) + 1)`). The query side is divided by the query's BM25 score against itself, so a perfect match scores about 1. Scores are then comparable across queries, which makes them usable as features and lets a minimum score (0.2) be applied.
- **Retrieval:** exact top 50 per field through a pruned sparse matrix product (`sparse_dot_topn`, multi-threaded C++). The full query × pool matrix is never built. One index is fitted per (split, country). N-grams appearing in more than 50k pool records are dropped for cost, since their IDF is near zero. There is **no LSH or MinHash**: this is a plain exact search.
- **Features for every candidate pair**, whichever channel found it: `ng_name`, `ng_name_addr` (normalized BM25), their ranks within the S1 record's list, and the channel flags `from_keys` / `from_ng`.
- **Union:** key-index top 200 ∪ n-gram top 50 per field. Pairs found only by n-grams get neutral blocking values (score 0, rank 65535, key counts 0).

> **Design note (resolved on Kaggle, 20k held-out train S1 records):** BM25 vs TF-IDF cosine for the n-gram channel. Both were run with the same code (`--weighting`), union candidates and re-ranker training.
>
> | | **BM25** | TF-IDF cosine |
> |---|---|---|
> | Generation recall: key index only (top 200) | 97.0% | 97.0% |
> | Generation recall: n-grams only | **96.4%** | 95.3% |
> | Generation recall: union | **98.5%** | 98.4% |
> | `ng_name_addr` score alone, recall@10 | **90.3%** | 88.3% |
> | Re-ranker recall@10 / @20 | 97.8% / 97.9% | 97.8% / 98.0% |
> | Feature-building time (60k S1 records) | **24 min** | 35 min |
>
> **Decision: BM25.** It retrieves and ranks better on its own and runs about 30% faster; after re-ranking the two tie. Adding the n-gram channel lifts the recall ceiling from 97.0% to **98.5%**, and re-ranked recall@10 from 97.1% to **97.8%**.

### Stage B: learned re-ranker (`features.py`, `rerank_dev.py`, `candidates.py`)

The `block_score` is a crude sum. Genuine matches often sit below records that share many address keys, which is why recall at top 50 was only 95.4% although 97.85% of true matches were generated somewhere. So for each union candidate (about 241 per S1) we compute 43 cheap features:
- rapidfuzz similarities on names and addresses (ratio, token_set, token_sort, partial, Jaro-Winkler)
- Jaccard overlap of name tokens, address numbers and address words
- IDF coverage of the name
- agreement of legal suffixes, empty-address and native-script flags, S2/S3 source
- lengths, plus `block_score`, `block_rank` and the per-type key counts
- n-gram BM25 scores and ranks (`ng_name`, `ng_name_addr`) and which channel found the pair

A **LightGBM classifier** trained on 40k train S1 records scores every candidate. They are labeled from the ground truth, and there are about 241 union candidates per S1 record. The **top 20** by that score are kept, with all their features. **This top-20 set is exactly what the matcher runs on, and what gets written to `candidate_pairs.tsv`.**

| Candidates per S1 after re-ranking | Keys only (local, v1) | **Keys ∪ n-gram BM25 (Kaggle, current)** |
|---|---|---|
| top 5 | 91.5% | 92.4% |
| top 10 | 97.1% | **97.8%** |
| top 20 | 97.3% | **97.9%** |
| everything generated (ceiling) | 97.4% (top 300) | **98.5%** (top 200 ∪ n-gram) |

The re-ranker's most useful features are address ratio, blocking score, candidate name length, name partial ratio and Jaro-Winkler, **n-gram name rank**, blocking rank, address token-set and **n-gram name+address score and rank**.

About 1.5% of true matches are still never generated at all. Before n-grams most misses were native-script names not covered by the transliteration map, website names with the words reordered, or records with an empty address whose name is heavily corrupted. The n-gram channel recovered a good share of these. Possible further fixes: dense multilingual embedding retrieval on the Kaggle GPU, and splitting glued website/handle names back into words.

## 4. Results so far (held-out train S1 records)

### Candidate generation (blocking)

| Version | Recall at top 50 | Recall of all generated candidates |
|---|---|---|
| Single keys only | 86.5% | – |
| + pair keys | 94.2% | 96.96% (top 400) |
| + transliteration map + 10-character compact key | 95.4% | **97.85%** (top 400) |
| + LightGBM re-ranker | 97.1% at top 10, 97.3% at top 20 | 97.4% (top 300) |
| + **n-gram BM25 channel (union)** + retrained re-ranker | **97.8% at top 10, 97.9% at top 20** | **98.5%** |

The re-ranker moves almost every true match that blocking finds into the top 10. So the final candidate set is only 10–20 records per S1, about 35M pairs on test.

A full local run of the **keys-only** version over all 2.2M train S1 records also finished: 43.8M candidate pairs, 19.9 per S1, about 90 minutes under the 7 GB cap. It is superseded by the Kaggle run with the n-gram union, but confirmed that the full-scale code works end to end.

### End-to-end F0.5 (first baseline, 20k-query sample)

| Setup | Macro F0.5 |
|---|---|
| Re-ranker probability, threshold 0.7 | **0.934** |
| Expected-F0.5 set selection | 0.931 |
| Upper bound: perfect matcher on the top-20 candidates | 0.990 |

**Main error found:** a pool record that shares an S1 record's address but belongs to a *different* business at that address gets a confident match. Its true owner isn't in the 20k sample, so nothing competes for the record. The competition features and the one-to-one rule need all S1 records at once, which is why full-scale candidate generation for every train S1 record is needed (running on Kaggle now, see section 6).

## 5. Infrastructure decisions

- **Memory safety:** every heavy local job goes through `./run.sh`. It runs the job in a systemd scope with a **7 GB hard cap**, no swap and lower CPU priority, so a job that goes over is killed by itself instead of freezing the machine. Long jobs write to disk in chunks and resume where they stopped.
- **Local machine:** GTX 1650 Ti (4 GB), 14 GB RAM, internet at about 70–380 KB/s. Development runs here on the CPU.
- **Everything heavy now runs on Kaggle** (30 GB RAM, 4 CPUs per session, GPUs later). It is driven from the local terminal with the **Kaggle CLI**, using the existing API token in `~/.kaggle/`. All jobs are private in the peeyushprashant account.
  - `ber-data` downloads the dataset from Google Drive once (about 1 minute). `ber-code` is our source, re-uploaded with `sync_code.sh`.
  - Each pipeline step is a script job pushed with `push_job.py`. It attaches the outputs of earlier jobs, so the work directory flows from job to job.
  - Only code and small logs cross the slow local connection.
- **Kaggle constraints we hit and handled:**
  - **Uploaded zips are flattened** (no `ber/` folder), so the runner rebuilds the package with a symlink.
  - **At most 5 CPU sessions at once:** extra jobs are queued and pushed automatically when a slot frees.
  - **12-hour session limit:** candidate generation is split per country and per shard.
  - **Running jobs' logs aren't visible from the CLI:** only status is, and the full log arrives when a job ends. Live progress is on the kaggle.com job pages.

### Kaggle runs so far
| Job | What | Result |
|---|---|---|
| `ber-data` | Download and extract the dataset | Done, ~1 min |
| `ber-prep` | Normalize train and test, learn and apply the transliteration map | Done, 14 min; the same 14,594 mappings as locally (reproducible); 4.6 GB output |
| `ber-rerank` | Train keys, BM25 n-gram index, union candidates, re-ranker (BM25) | Done, 30 min; recall numbers above; `reranker.txt` |
| `ber-rerank-tfidf` | The same with TF-IDF cosine (design-note check) | Done, 43 min; BM25 chosen |

## 6. Status as of 2026-09-25, 22:54

### Running right now (Kaggle, CPU sessions; 5 is Kaggle's maximum at once)

All are full-scale **candidate generation** jobs: key index ∪ BM25 n-grams, then 43 features, then the re-ranker, keeping the top 20 per S1. Every job uses the same inputs, `ber-prep` (normalized data) and `ber-rerank` (`reranker.txt`).

| Job | Covers | S1 records | Started | Est. total | Link |
|---|---|---|---|---|---|
| `ber-cand-train-us0` | train US, super-chunks 0, 2, 4 | 724k | ~22:15 | ~2.1 h | [page](https://www.kaggle.com/code/peeyushprashant/ber-cand-train-us0) |
| `ber-cand-train-us1` | train US, super-chunks 1, 3 | 600k | ~22:15 | ~1.8 h | [page](https://www.kaggle.com/code/peeyushprashant/ber-cand-train-us1) |
| `ber-cand-train-in0` | train India, super-chunks 0, 2 | 583k | ~22:15 | ~1.7 h | [page](https://www.kaggle.com/code/peeyushprashant/ber-cand-train-in0) |
| `ber-cand-train-in1` | train India, super-chunk 1 | 300k | ~22:15 | ~1 h | [page](https://www.kaggle.com/code/peeyushprashant/ber-cand-train-in1) |
| `ber-cand-test-us` | test US | 663k | ~22:15 | ~2 h | [page](https://www.kaggle.com/code/peeyushprashant/ber-cand-test-us) |

The estimates are extrapolated from `ber-rerank` (about 110 S1 records per second on 4 CPUs, plus fixed index-fitting cost) and are ±50%. These 5 jobs started **before** progress logging was added, so their pages show only setup output until they finish.

### Queued (pushed automatically by a local background loop when a slot frees)

| Job | Covers | S1 records | Est. total |
|---|---|---|---|
| `ber-cand-test-in` | test India | 810k | ~2.2 h |
| `ber-cand-test-fr` | test France | 259k | ~1 h |

These two use the new code: **live progress on the job page**, a line every minute with ETA per super-chunk and an OVERALL ETA.

**Expected end of the candidate stage:** about 3.5–4.5 h from 22:54, i.e. **around 02:30–03:30**. The limit is that only 5 CPU sessions can run at once.

**Local machine:** nothing is running, apart from the small loop that pushes the queued jobs.

### Done today
- Local: full keys-only pipeline prototype, baseline F0.5 of 0.934 on a 20k sample, and a full train candidate run with keys only (43.8M pairs, now superseded).
- Kaggle: `ber-data`, `ber-prep`, `ber-rerank` (BM25), `ber-rerank-tfidf` (design-note check; BM25 chosen).
- Code: n-gram BM25 channel and union, retrained re-ranker (recall@10 **97.8%**, ceiling **98.5%**), per-country/shard candidate jobs, Kaggle job tooling, progress logging.

## 7. What is left

### A. To reach the first leaderboard submission (v1, no cross-encoder)
| # | Step | Command / job | Runs on | Needs | Est. |
|---|---|---|---|---|---|
| 1 | Candidate generation (7 jobs) | `ber-cand-*` | Kaggle CPU | `ber-prep`, `ber-rerank` | *running* |
| 2 | Merge parts into `train_cand.parquet` / `test_cand.parquet` | `ber.candidates --splits train test --merge` | Kaggle CPU | all 7 jobs | ~5 min |
| 3 | Context features (competition, cluster) | `ber.matcher ctx --split train` and `--split test` | Kaggle CPU | step 2 | ~30–60 min |
| 4 | Out-of-fold matcher: **full-scale F0.5** plus the decision rule and threshold | `ber.matcher oof` | Kaggle CPU | step 3 (train) | ~30–60 min |
| 5 | Final: train the matcher, predict test, write and validate `output/matching_results.tsv` + `candidate_pairs.tsv` | `ber.matcher final` | Kaggle CPU | steps 3 (test) and 4 | ~20–30 min |
| 6 | Download `matching_results.tsv`, **upload to the portal** | `kaggle kernels output`, then the portal | local, then you | step 5 | – |

Steps 2–5 can run as one chained Kaggle job attaching all 7 candidate outputs.

### B. Improvements after v1
1. **Cross-encoder (v2).** The code is written (`cross_encoder.py`) but **has never been run**. Needed:
   - an `accelerate launch` step type and T4×2 settings in `push_job.py`
   - a smoke test on a small slice (to check correctness and measure throughput)
   - a full train and score run (~4–6 h of GPU quota)
   - the matcher re-run with `ce_logit`, and its out-of-fold F0.5 compared against v1
2. **France check:** leave-one-country-out validation (train on US and test on India, then the reverse) as a stand-in for the unseen country.
3. **Recall beyond 98.5%:** dense multilingual bi-encoder retrieval on GPU, and splitting glued website/handle names into words.
4. **Final package:** `README.md`, pinned `requirements.txt`, filled-in `Documentation_template.md`, and the zip with `output/` + `code/`.

### Open issues and risks
- **Candidate job runtimes are estimates.** If a job gets close to Kaggle's 12 h limit it would lose its output; per-country/shard splitting is the safeguard.
- **The cross-encoder is untested code.** Expect a debugging round on its first run.
- **No labeled France data exists anywhere;** only the leave-one-country-out check can hint at France performance.
- **Git:** everything is on `master` (one commit, `d4cace4`). You wanted the work on a `peeyush` branch (see the commands discussed); newer files (`ngram.py`, `log.py`, Kaggle tooling) are not committed yet.

## 8. How to reproduce what exists

**On Kaggle (current path), from the local terminal:**
```bash
business_entity_resolution/kaggle/sync_code.sh "msg"                 # upload src/ber as ber-code
P=business_entity_resolution/kaggle/push_job.py
.venv/bin/kaggle kernels push -p business_entity_resolution/kaggle/kernels/ber-data   # once
.venv/bin/python $P ber-prep   --steps "ber.prepare --splits train test"
.venv/bin/python $P ber-rerank --after ber-prep --steps "ber.rerank_dev --weighting bm25"
.venv/bin/python $P ber-cand-train-us0 --after ber-prep ber-rerank \
    --steps "ber.candidates --splits train --countries US --shard 0/2"   # likewise us1, in0/in1, test-us/in/fr
.venv/bin/python $P ber-prep --log                                   # read a finished job's log
```

**Locally (development, under the 7 GB cap):**
```bash
python3.12 -m venv .venv && .venv/bin/pip install polars pyarrow numpy pandas scikit-learn scipy rapidfuzz anyascii lightgbm tqdm
ln -s Dataset/<...>/student_resource/dataset data
./run.sh ber.prepare --splits train test      # normalise; learns + applies transliteration map
./run.sh ber.eval_blocking                    # blocking recall on 30k train queries
./run.sh ber.rerank_dev                       # union candidates + re-ranker + channel recall report
./run.sh ber.f05_dev                          # baseline macro F0.5 on the dev sample
./run.sh ber.candidates --splits train test   # full-scale candidates (resumable)
./run.sh ber.matcher ctx --split train        # context features
./run.sh ber.matcher oof                      # full-scale OOF F0.5 + decision rule
./run.sh ber.matcher final                    # test predictions -> output/ + validation
```
