"""Normalise every source file once and cache as parquet.

work/{split}_s1.parquet    S1 records, idx = row number
work/{split}_pool.parquet  S2 + S3 records, idx = row number (S2 first)
work/train_gt_pairs.parquet  (q_idx, p_idx) true pairs in index space
"""
import argparse
import time

import polars as pl

from ber.io import WORK, load_ground_truth, load_source
from ber import translit
from ber.normalize import normalize


def prepare_split(split: str) -> None:
    WORK.mkdir(exist_ok=True)
    tmap = translit.load()
    print(f"translit mappings: {len(tmap):,}")
    t = time.time()
    s1 = normalize(load_source(split, 1), tmap).with_row_index("idx")
    s1.write_parquet(WORK / f"{split}_s1.parquet")
    print(f"{split} s1 {s1.height:,} rows  {time.time() - t:.0f}s", flush=True)
    del s1

    parts = []
    for src in (2, 3):
        t = time.time()
        part = normalize(load_source(split, src), tmap)
        path = WORK / f"{split}_s{src}_norm.parquet"
        part.write_parquet(path)
        parts.append(path)
        print(f"{split} s{src} {part.height:,} rows  {time.time() - t:.0f}s", flush=True)
        del part
    pool = pl.concat([pl.scan_parquet(p) for p in parts]).with_row_index("idx")
    pool.sink_parquet(WORK / f"{split}_pool.parquet")
    for p in parts:
        p.unlink()


def prepare_gt() -> None:
    gt = load_ground_truth()
    s1 = pl.read_parquet(WORK / "train_s1.parquet", columns=["idx", "entity_id"])
    pool = pl.read_parquet(WORK / "train_pool.parquet", columns=["idx", "entity_id"])
    pairs = (
        gt.drop_nulls("match")
        .join(s1.rename({"idx": "q_idx", "entity_id": "s1"}), on="s1")
        .join(pool.rename({"idx": "p_idx", "entity_id": "match"}), on="match")
        .select("q_idx", "p_idx")
    )
    assert pairs.height == gt.drop_nulls("match").height, "unmapped ground-truth ids"
    pairs.write_parquet(WORK / "train_gt_pairs.parquet")
    print(f"gt pairs {pairs.height:,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    args = ap.parse_args()
    for sp in args.splits:
        prepare_split(sp)
    if "train" in args.splits:
        prepare_gt()
        if not translit.PATH.exists():
            # Dictionary is learned from the first normalised pass; re-run to apply it.
            translit.learn()
            print("learned transliteration map; re-running normalisation with it")
            for sp in args.splits:
                prepare_split(sp)
