"""Dev: macro F0.5 of the re-ranker probabilities under several decision rules."""
import lightgbm as lgb
import numpy as np
import polars as pl

from ber.decide import expected_f05_sets, one_to_one
from ber.features import FEATURES
from ber.io import WORK
from ber.metric import macro_f05


def main() -> None:
    ev = pl.read_parquet(WORK / "dev_feats_eval.parquet")
    booster = lgb.Booster(model_file=str(WORK / "reranker_dev.txt"))
    ev = ev.with_columns(prob=pl.Series(booster.predict(ev.select(FEATURES).to_numpy()).astype(np.float32)))
    # Evaluation universe = the 20k held-out queries (same shuffle as rerank_dev).
    all_q = pl.read_parquet(WORK / "train_s1.parquet", columns=["idx"])["idx"].shuffle(seed=42)
    q_ev = all_q.slice(40_000, 20_000)
    gt = (
        pl.read_parquet(WORK / "train_gt_pairs.parquet")
        .filter(pl.col("q_idx").is_in(q_ev.implode()))
        .rename({"q_idx": "s1", "p_idx": "match"})
    )
    print(f"singleton share in eval: {1 - gt['s1'].n_unique() / len(q_ev):.3f}")

    def score(sel: pl.DataFrame) -> float:
        return macro_f05(sel.select(s1="q_idx", match="p_idx"), gt, q_ev)

    top20 = ev.filter(pl.col("prob").rank("ordinal", descending=True).over("q_idx") <= 20)
    print(f"oracle (perfect matcher on top-20 candidates): {score(top20.filter(pl.col('label') == 1)):.4f}")
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        print(f"threshold {t}: {score(top20.filter(pl.col('prob') >= t)):.4f}   "
              f"+1:1 {score(one_to_one(top20).filter(pl.col('prob') >= t)):.4f}")
    print(f"expected-F0.5 sets: {score(expected_f05_sets(top20)):.4f}   "
          f"+1:1 {score(expected_f05_sets(one_to_one(top20))):.4f}")


if __name__ == "__main__":
    main()
