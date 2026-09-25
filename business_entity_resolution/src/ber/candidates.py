"""Full-scale candidate generation: blocking (top-B) -> features -> LightGBM re-ranker ->
top-K per S1 kept with all features. Streams super-chunks of queries to disk so RAM
stays bounded regardless of split size.

Output: work/{split}_cand.parquet  (q_idx, p_idx, FEATURES..., rr_prob, rr_rank)
"""
import argparse
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from ber.blocking import generate_candidates
from ber.features import FEATURES, REC_COLS, attach_records, name_token_idf, pair_features
from ber.io import WORK

RERANKER = WORK / "reranker_dev.txt"


def run(split: str, block_k: int, keep_k: int, super_chunk: int, chunk: int) -> None:
    out = WORK / f"{split}_cand.parquet"
    idf_path = WORK / f"{split}_name_idf.parquet"
    if not idf_path.exists():
        name_token_idf(WORK / f"{split}_pool.parquet").write_parquet(idf_path)
    tok_idf = pl.read_parquet(idf_path)
    booster = lgb.Booster(model_file=str(RERANKER))

    all_q = pl.read_parquet(WORK / f"{split}_s1.parquet", columns=["idx"])["idx"]
    parts = []
    t0 = time.time()
    for si, s in enumerate(range(0, len(all_q), super_chunk)):
        part = out.with_suffix(f".part{si}.parquet")
        parts.append(part)
        if part.exists():
            continue
        q_ids = all_q.slice(s, super_chunk)
        cand = generate_candidates(split, q_ids, top_k=block_k)
        q = pl.scan_parquet(WORK / f"{split}_s1.parquet").select(REC_COLS).filter(pl.col("idx").is_in(q_ids.implode())).collect()
        p = (
            pl.scan_parquet(WORK / f"{split}_pool.parquet").select(REC_COLS)
            .filter(pl.col("idx").is_in(cand["p_idx"].unique().implode())).collect()
        )
        kept = []
        for c0 in range(0, len(q_ids), chunk):
            c = cand.filter(pl.col("q_idx").is_in(q_ids.slice(c0, chunk).implode()))
            if c.height == 0:
                continue
            f = pair_features(attach_records(c, q, p), tok_idf).select(["q_idx", "p_idx", *FEATURES])
            f = f.with_columns(rr_prob=pl.Series(booster.predict(f.select(FEATURES).to_numpy()).astype(np.float32)))
            f = f.with_columns(rr_rank=pl.col("rr_prob").rank("ordinal", descending=True).over("q_idx").cast(pl.UInt16))
            kept.append(f.filter(pl.col("rr_rank") <= keep_k))
        pl.concat(kept).write_parquet(part)
        del cand, q, p, kept
        print(f"  super-chunk {si}: {s + len(q_ids):,}/{len(all_q):,} queries  {time.time() - t0:.0f}s", flush=True)
    pl.concat([pl.scan_parquet(x) for x in parts]).sink_parquet(out)
    for x in parts:
        x.unlink()
    n = pl.scan_parquet(out).select(pl.len()).collect().item()
    print(f"{split}: {n:,} candidate pairs ({n / len(all_q):.1f}/query) -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train"])
    ap.add_argument("--block-k", type=int, default=200)
    ap.add_argument("--keep-k", type=int, default=20)
    ap.add_argument("--super-chunk", type=int, default=300_000)
    ap.add_argument("--chunk", type=int, default=5_000)
    a = ap.parse_args()
    for sp in a.splits:
        run(sp, a.block_k, a.keep_k, a.super_chunk, a.chunk)
