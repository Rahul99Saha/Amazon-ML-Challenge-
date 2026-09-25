"""Dev experiment: wide blocking (top-300) -> LightGBM re-ranker -> recall@K.

Trains on candidates of one random set of train S1 queries, evaluates recall@K on a
disjoint set. Saves feature frames for reuse by the matcher experiments.
"""
import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from ber.blocking import generate_candidates
from ber.features import FEATURES, REC_COLS, attach_records, name_token_idf, pair_features
from ber.io import WORK


def build(q_idx: pl.Series, top_k: int, tok_idf: pl.DataFrame, gt: pl.DataFrame, out: Path, chunk: int = 5_000) -> pl.DataFrame:
    """Features for all candidates of q_idx, written chunk by chunk to `out` (bounded RAM)."""
    cand = generate_candidates("train", q_idx, top_k=top_k)
    p_ids = cand["p_idx"].unique()
    q = pl.scan_parquet(WORK / "train_s1.parquet").select(REC_COLS).filter(pl.col("idx").is_in(q_idx.implode())).collect()
    p = pl.scan_parquet(WORK / "train_pool.parquet").select(REC_COLS).filter(pl.col("idx").is_in(p_ids.implode())).collect()
    gt = gt.with_columns(label=pl.lit(1, pl.Int8))
    parts = []
    qs = q_idx.sort()
    for i, s in enumerate(range(0, len(qs), chunk)):
        c = cand.filter(pl.col("q_idx").is_in(qs.slice(s, chunk).implode()))
        f = pair_features(attach_records(c, q, p), tok_idf).select(["q_idx", "p_idx", *FEATURES])
        f = f.join(gt, on=["q_idx", "p_idx"], how="left").with_columns(pl.col("label").fill_null(0))
        part = out.with_suffix(f".part{i}.parquet")
        f.write_parquet(part)
        parts.append(part)
        del c, f
    del cand, q, p
    pl.concat([pl.scan_parquet(x) for x in parts]).sink_parquet(out)
    for x in parts:
        x.unlink()
    return pl.read_parquet(out)


def recall_at(feats: pl.DataFrame, score: np.ndarray, gt: pl.DataFrame, ks=(5, 10, 20, 30, 40, 50, 75, 100)) -> None:
    r = feats.select("q_idx", "p_idx").with_columns(s=pl.Series(score)).with_columns(
        rk=pl.col("s").rank("ordinal", descending=True).over("q_idx")
    )
    hit = gt.join(r, on=["q_idx", "p_idx"], how="left")
    for k in ks:
        print(f"  recall@{k}: {(hit['rk'].fill_null(10**6) <= k).mean():.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=60_000)
    ap.add_argument("--n-eval", type=int, default=20_000)
    ap.add_argument("--top-k", type=int, default=300)
    a = ap.parse_args()

    all_q = pl.read_parquet(WORK / "train_s1.parquet", columns=["idx"])["idx"].shuffle(seed=42)
    q_tr, q_ev = all_q.slice(0, a.n_train), all_q.slice(a.n_train, a.n_eval)
    gt = pl.read_parquet(WORK / "train_gt_pairs.parquet")
    idf_path = WORK / "train_name_idf.parquet"
    if not idf_path.exists():
        name_token_idf(WORK / "train_pool.parquet").write_parquet(idf_path)
    tok_idf = pl.read_parquet(idf_path)

    t = time.time()
    tr_path, ev_path = WORK / "dev_feats_train.parquet", WORK / "dev_feats_eval.parquet"
    if not (tr_path.exists() and ev_path.exists()):
        build(q_tr, a.top_k, tok_idf, gt.filter(pl.col("q_idx").is_in(q_tr.implode())), tr_path)
        build(q_ev, a.top_k, tok_idf, gt.filter(pl.col("q_idx").is_in(q_ev.implode())), ev_path)
    tr, ev = pl.read_parquet(tr_path), pl.read_parquet(ev_path)
    print(f"features ready {time.time() - t:.0f}s  train pairs {tr.height:,}  eval pairs {ev.height:,}")

    gt_ev = gt.filter(pl.col("q_idx").is_in(q_ev.implode()))
    print("blocking score only:")
    recall_at(ev, ev["block_score"].to_numpy(), gt_ev)

    model = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.08, num_leaves=63, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
    )
    model.fit(tr.select(FEATURES).to_numpy(), tr["label"].to_numpy())
    s = model.predict_proba(ev.select(FEATURES).to_numpy())[:, 1]
    print("re-ranker:")
    recall_at(ev, s, gt_ev)
    imp = sorted(zip(model.booster_.feature_importance("gain"), FEATURES), reverse=True)[:15]
    print("top features:", [f for _, f in imp])
    model.booster_.save_model(str(WORK / "reranker_dev.txt"))


if __name__ == "__main__":
    main()
