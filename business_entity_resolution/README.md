# Business Entity Resolution: Kernel Squad

This is the pipeline that produces `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the challenge data. For every Source 1 (S1) record it finds the matching Source 2 / Source 3 records.

```
raw TSV → prepare (normalise + learned transliteration map)
        → candidates: rare-key inverted index (top 200) ∪ character 3-gram BM25 (top 50 per field)
        → 43 pair features → LightGBM re-ranker → top 20 per S1  (= candidate_pairs.tsv)
        → context features (query context, competition between S1s, cluster consistency)
        → LightGBM matcher (out-of-fold on train) → one-to-one + threshold tuned on macro F0.5
        → matching_results.tsv → official validator
```

The methodology is described in `Documentation_template.md` at the root of the submission zip.

## Layout

```
src/ber/            pipeline package (run modules with `python -m ber.<module>`)
  io.py             TSV loading and caching; paths via BER_DATA / BER_WORK / BER_OUTPUT
  normalize.py      name/address normalisation (vectorised polars)
  translit.py       transliteration dictionary learned from train pairs only
  prepare.py        normalise all splits → work/*.parquet
  blocking.py       candidate channel 1: IDF-weighted rare-key inverted index
  ngram.py          candidate channel 2: BM25 over character n-grams (+ TF-IDF option)
  features.py       pair features
  candidates.py     union → features → re-ranker → top-K, per country / shard
  rerank_dev.py     trains the re-ranker (reranker.txt) and reports candidate recall
  context.py        context / competition / cluster features
  matcher.py        ctx | oof | final: matcher, decision rule, test predictions
  decide.py         one-to-one assignment, expected-F0.5 set selection
  metric.py         macro F0.5 identical to the official definition
  submit.py         writes both output TSVs and runs utils/validate_submission.py
  cross_encoder.py  optional GPU stage: mDeBERTa-v3-base cross-encoder score as a feature
  log.py            timestamped progress logging with ETA
  eval_blocking.py, f05_dev.py   development diagnostics (not needed for the outputs)
kaggle/             scripts to run the pipeline as Kaggle jobs from a local terminal
requirements.txt
```

## Environment

- Python 3.12. Install with `pip install -r requirements.txt`.
- **Hardware used:** Kaggle CPU sessions (4 vCPU, ~30 GB RAM) for all stages. On machines with less memory, run the candidate stage per country / shard (see below).
- **No external data** is used at any point. The only inputs are the challenge TSVs. The transliteration map and all IDF / BM25 statistics are computed from the provided data.

## Reproduce end to end (single machine)

```bash
export PYTHONPATH=src
export BER_DATA=/path/to/student_resource/dataset   # contains train/ and test/
export BER_WORK=/path/to/work                        # intermediate files (~15 GB)
export BER_OUTPUT=/path/to/output

python -m ber.prepare --splits train test            # normalise; learns + applies transliteration map
python -m ber.rerank_dev --weighting bm25            # re-ranker (40k train S1), writes work/reranker.txt
python -m ber.candidates --splits train test         # union candidates → top 20 per S1 (resumable)
python -m ber.matcher ctx --split train
python -m ber.matcher ctx --split test
python -m ber.matcher oof                            # out-of-fold F0.5 on train; picks decision rule → work/decision.json
python -m ber.matcher final                          # test predictions → $BER_OUTPUT/*.tsv + official validation
```

`ber.candidates` accepts `--countries <label>` and `--shard i/n`, so it can be split across machines or sessions. Then run `python -m ber.candidates --splits train test --merge`.

## Reproduce on Kaggle (how our outputs were produced)

Every stage ran as a private Kaggle script job, pushed from a local terminal with the Kaggle CLI (API token in `~/.kaggle/`):

```bash
kaggle/sync_code.sh "msg"                                  # upload src/ber as dataset <user>/ber-code
kaggle kernels push -p kaggle/kernels/ber-data             # downloads the dataset zip once
P=kaggle/push_job.py
python $P ber-prep   --steps "ber.prepare --splits train test"
python $P ber-rerank --after ber-prep --steps "ber.rerank_dev --weighting bm25"
python $P ber-cand-train-us0 --after ber-prep ber-rerank \
          --steps "ber.candidates --splits train --countries US --shard 0/2"
#   likewise: train US 1/2, train India 0/2 and 1/2, test US, test India, test France
python $P <job> --log                                      # read a finished job's log
```

Each job attaches the outputs of the jobs in `--after`, so the work directory flows from job to job. Kaggle allows 5 CPU sessions at once and 12 h per session; the per-country / per-shard split keeps every job within those limits.

## Runtime (Kaggle, 4 vCPU)

| Stage | Time |
|---|---|
| prepare (train + test) | 14 min |
| re-ranker training | 30 min |
| candidates, per 300k S1 records | ~1.9 h |
| context + matcher + final | *to be measured* |
