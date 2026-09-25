"""Candidate generation = union of two channels, then a learned re-ranker.

  channel 1: rare-key inverted index (blocking.py), top `block_k` by IDF score
  channel 2: character n-gram BM25 (ngram.py), top `ng_k` per field
  union -> pair features (string sims + n-gram scores for every pair) -> LightGBM
  re-ranker -> top `keep_k` per S1 kept with all features.

Processed per country (one n-gram index fit per country) and in super-chunks of
queries written to disk, so RAM stays bounded and interrupted runs resume.

Output: work/{split}_cand.parquet  (q_idx, p_idx, FEATURES..., rr_prob, rr_rank)
"""
import argparse
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from ber.blocking import KINDS, generate_candidates
from ber.features import FEATURES, REC_COLS, attach_records, name_token_idf, pair_features
from ber.io import WORK
from ber.ngram import NgramIndex

RERANKER = WORK / "reranker.txt"


def union_candidates(split: str, q_ids: pl.Series, ng: NgramIndex, block_k: int, ng_k: int) -> pl.DataFrame:
    keys = generate_candidates(split, q_ids, top_k=block_k).with_columns(from_keys=pl.lit(1, pl.Int8))
    ngc = ng.topk(q_ids, top_k=ng_k).with_columns(from_ng=pl.lit(1, pl.Int8))
    u = keys.join(ngc, on=["q_idx", "p_idx"], how="full", coalesce=True)
    return u.with_columns(
        pl.col("from_keys", "from_ng").fill_null(0),
        pl.col("block_score").fill_null(0.0),
        pl.col("block_rank").fill_null(65535),
        *[pl.col(f"bk_{k}").fill_null(0) for k in KINDS],
    )


def featurize(split: str, cand: pl.DataFrame, ng: NgramIndex, tok_idf: pl.DataFrame, chunk: int = 5_000):
    """Yields feature frames (q_idx, p_idx, FEATURES) for chunks of queries."""
    q_ids = cand["q_idx"].unique().sort()
    q = pl.scan_parquet(WORK / f"{split}_s1.parquet").select(REC_COLS).filter(pl.col("idx").is_in(q_ids.implode())).collect()
    p = (
        pl.scan_parquet(WORK / f"{split}_pool.parquet").select(REC_COLS)
        .filter(pl.col("idx").is_in(cand["p_idx"].unique().implode())).collect()
    )
    for s in range(0, len(q_ids), chunk):
        c = cand.filter(pl.col("q_idx").is_in(q_ids.slice(s, chunk).implode()))
        c = ng.pair_features(c)
        yield pair_features(attach_records(c, q, p), tok_idf).select(["q_idx", "p_idx", *FEATURES])


def token_idf(split: str) -> pl.DataFrame:
    path = WORK / f"{split}_name_idf.parquet"
    if not path.exists():
        name_token_idf(WORK / f"{split}_pool.parquet").write_parquet(path)
    return pl.read_parquet(path)


def part_path(split: str, ctry: str, i: int):
    return WORK / f"{split}_cand_part_{ctry.replace(' ', '_')}_{i:03d}.parquet"


def merge(split: str) -> None:
    out = WORK / f"{split}_cand.parquet"
    pl.scan_parquet(str(WORK / f"{split}_cand_part_*.parquet")).sink_parquet(out)
    n_q = pl.scan_parquet(WORK / f"{split}_s1.parquet").select(pl.len()).collect().item()
    n = pl.scan_parquet(out).select(pl.len()).collect().item()
    print(f"{split}: {n:,} candidate pairs ({n / n_q:.1f}/query) -> {out}")


def run(split: str, block_k: int, ng_k: int, keep_k: int, super_chunk: int, weighting: str,
        countries: list[str] | None, shard: tuple[int, int] = (0, 1)) -> None:
    tok_idf = token_idf(split)
    booster = lgb.Booster(model_file=str(RERANKER))
    s1 = pl.read_parquet(WORK / f"{split}_s1.parquet", columns=["idx", "country"])
    t0 = time.time()
    for ctry in sorted(s1["country"].unique().to_list()):
        if countries and ctry not in countries:
            continue
        q_all = s1.filter(pl.col("country") == ctry)["idx"]
        plan = [(part_path(split, ctry, i), s) for i, s in enumerate(range(0, len(q_all), super_chunk))
                if i % shard[1] == shard[0]]
        if all(path.exists() for path, _ in plan):
            continue
        ng = NgramIndex(split, ctry, weighting=weighting)
        print(f"{ctry}: n-gram index fitted ({time.time() - t0:.0f}s)", flush=True)
        for path, s in plan:
            if path.exists():
                continue
            q_ids = q_all.slice(s, super_chunk)
            cand = union_candidates(split, q_ids, ng, block_k, ng_k)
            kept = []
            for f in featurize(split, cand, ng, tok_idf):
                f = f.with_columns(rr_prob=pl.Series(booster.predict(f.select(FEATURES).to_numpy()).astype(np.float32)))
                f = f.with_columns(rr_rank=pl.col("rr_prob").rank("ordinal", descending=True).over("q_idx").cast(pl.UInt16))
                kept.append(f.filter(pl.col("rr_rank") <= keep_k))
            pl.concat(kept).write_parquet(path)
            print(f"  {ctry} {s + len(q_ids):,}/{len(q_all):,}  union {cand.height / len(q_ids):.0f}/q  "
                  f"{time.time() - t0:.0f}s", flush=True)
            del cand, kept
        del ng


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train"])
    ap.add_argument("--block-k", type=int, default=200)
    ap.add_argument("--ng-k", type=int, default=50)
    ap.add_argument("--keep-k", type=int, default=20)
    ap.add_argument("--super-chunk", type=int, default=300_000)
    ap.add_argument("--weighting", default="bm25", choices=["bm25", "tfidf"])
    ap.add_argument("--countries", nargs="*", help="only these countries (one Kaggle job each)")
    ap.add_argument("--merge", action="store_true", help="only merge existing parts into {split}_cand.parquet")
    ap.add_argument("--shard", default="0/1", help="i/n: only every n-th super-chunk, starting at i")
    a = ap.parse_args()
    shard = tuple(int(x) for x in a.shard.split("/"))
    for sp in a.splits:
        if not a.merge:
            run(sp, a.block_k, a.ng_k, a.keep_k, a.super_chunk, a.weighting, a.countries, shard)
        if a.merge or not a.countries:
            merge(sp)
