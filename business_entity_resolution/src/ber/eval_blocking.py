"""Measure blocking recall on a sample of train S1 queries against the full train pool."""
import argparse
import time

import polars as pl

from ber.blocking import generate_candidates
from ber.io import WORK


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30_000)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-block", type=int, default=300)
    a = ap.parse_args()

    q_idx = pl.read_parquet(WORK / "train_s1.parquet", columns=["idx"])["idx"].sample(a.n, seed=0)
    gt = pl.read_parquet(WORK / "train_gt_pairs.parquet").filter(pl.col("q_idx").is_in(q_idx.implode()))
    t = time.time()
    cand = generate_candidates("train", q_idx, top_k=a.top_k, max_block=a.max_block)
    el = time.time() - t
    hit = gt.join(cand, on=["q_idx", "p_idx"], how="left")
    print(f"queries {a.n:,}  time {el:.0f}s  cand/query {cand.height / a.n:.1f}")
    print(f"pair recall {hit['block_score'].is_not_null().mean():.4f}")
    for k in (5, 10, 20, 30, 50):
        print(f"  recall@{k}: {(hit['block_rank'].fill_null(9999) < k).mean():.4f}")
    hit.filter(pl.col("block_score").is_null()).select("q_idx", "p_idx").write_parquet(
        WORK / "blocking_misses.parquet"
    )


if __name__ == "__main__":
    main()
