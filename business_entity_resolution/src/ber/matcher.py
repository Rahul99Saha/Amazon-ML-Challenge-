"""Stage-2 matcher: context features over the full candidate graph, LightGBM with
out-of-fold predictions for every train query, decision rule chosen with the official
macro F0.5 on the full train split (so one-to-one competition is realistic).

  ctx   : work/{split}_cand.parquet -> work/{split}_ctx.parquet
  oof   : 2-fold OOF probabilities on train -> report + work/decision.json
  final : fit on train, predict test, write output/ files and validate

If work/{split}_ce.parquet exists (cross-encoder scores) `ce_logit` becomes a feature;
queries the cross-encoder or the re-ranker was trained on are never used to train or
evaluate the matcher (their scores are optimistic), but they are still predicted so
that one-to-one competition stays complete.
"""
import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from ber import submit
from ber.context import CONTEXT_FEATURES, add_context, competition
from ber.decide import expected_f05_sets, one_to_one
from ber.features import FEATURES
from ber.io import WORK, is_ce_query
from ber.log import Progress, get, timed
from ber.metric import macro_f05

BASE_FEATURES = FEATURES + [f for f in CONTEXT_FEATURES if f not in FEATURES]
PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    verbose=-1, num_threads=8,
)
DECISION = WORK / "decision.json"
log = get("matcher")


def has_ce(split: str) -> bool:
    return (WORK / f"{split}_ce.parquet").exists()


def features(split: str) -> list[str]:
    return BASE_FEATURES + (["ce_logit", "ce_rank", "ce_gap"] if has_ce(split) else [])


def build_ctx(split: str, q_chunk: int = 100_000) -> None:
    src, dst = WORK / f"{split}_cand.parquet", WORK / f"{split}_ctx.parquet"
    slim = pl.read_parquet(src, columns=["q_idx", "p_idx", "rr_prob"])
    qs = slim["q_idx"].unique().sort()
    with timed(log, f"{split}: competition features over {slim.height:,} pairs"):
        claims = competition(slim)
    del slim
    prog = Progress(log, f"{split} context features (queries)", len(qs))
    parts = []
    for i, s in enumerate(range(0, len(qs), q_chunk)):
        ids = qs.slice(s, q_chunk)
        c = pl.scan_parquet(src).filter(pl.col("q_idx").is_in(ids.implode())).collect()
        c = add_context(c, split).join(claims.filter(pl.col("q_idx").is_in(ids.implode())), on=["q_idx", "p_idx"])
        part = dst.with_suffix(f".part{i}.parquet")
        c.write_parquet(part)
        parts.append(part)
        prog.step(len(ids))
    pl.concat([pl.scan_parquet(x) for x in parts]).sink_parquet(dst)
    for x in parts:
        x.unlink()


def _rows(split: str, q_ids: pl.Series) -> pl.DataFrame:
    c = pl.scan_parquet(WORK / f"{split}_ctx.parquet").filter(pl.col("q_idx").is_in(q_ids.implode()))
    if has_ce(split):
        ce = pl.scan_parquet(WORK / f"{split}_ce.parquet").filter(pl.col("q_idx").is_in(q_ids.implode()))
        # scored pairs define the candidate set (top-k by re-ranker)
        c = c.join(ce, on=["q_idx", "p_idx"], how="inner").with_columns(
            ce_rank=pl.col("ce_logit").rank("ordinal", descending=True).over("q_idx").cast(pl.Float32),
            ce_gap=(pl.col("ce_logit").max().over("q_idx") - pl.col("ce_logit")),
        )
    return c.select(["q_idx", "p_idx", *features(split)]).collect()


def _train(df: pl.DataFrame, feats: list[str], rounds: int) -> lgb.Booster:
    ds = lgb.Dataset(df.select(feats).to_numpy(np.float32), df["label"].to_numpy(), free_raw_data=True)
    log.info(f"LightGBM: {df.height:,} pairs, {len(feats)} features, {rounds} rounds, positives {df['label'].mean():.3f}")
    return lgb.train(PARAMS, ds, num_boost_round=rounds,
                     callbacks=[lgb.log_evaluation(100)], valid_sets=[ds], valid_names=["train"])


def _predict(booster: lgb.Booster, split: str, q_ids: pl.Series, chunk: int = 200_000) -> pl.DataFrame:
    feats, outs = features(split), []
    prog = Progress(log, f"{split} predict (queries)", len(q_ids))
    for s in range(0, len(q_ids), chunk):
        c = _rows(split, q_ids.slice(s, chunk))
        p = booster.predict(c.select(feats).to_numpy(np.float32)).astype(np.float32)
        outs.append(c.select("q_idx", "p_idx").with_columns(prob=pl.Series(p)))
        prog.step(min(chunk, len(q_ids) - s))
    return pl.concat(outs)


def _training_rows(q_ids: pl.Series, n_q: int, seed: int) -> pl.DataFrame:
    gt = pl.read_parquet(WORK / "train_gt_pairs.parquet").with_columns(label=pl.lit(1, pl.Int8))
    c = _rows("train", q_ids.sample(min(n_q, len(q_ids)), seed=seed))
    return c.join(gt, on=["q_idx", "p_idx"], how="left").with_columns(pl.col("label").fill_null(0))


def eligible_train_queries() -> tuple[pl.Series, pl.Series]:
    """(all train query idx, queries usable to train/evaluate the matcher)."""
    s1 = pl.read_parquet(WORK / "train_s1.parquet", columns=["idx", "entity_id"])
    if has_ce("train"):
        # only queries that received cross-encoder scores take part
        scored = pl.scan_parquet(WORK / "train_ce.parquet").select("q_idx").unique().collect()["q_idx"]
        s1 = s1.filter(pl.col("idx").is_in(scored.implode()))
    all_q = s1["idx"]
    full = pl.read_parquet(WORK / "train_s1.parquet", columns=["idx"])["idx"]
    bad = full.shuffle(seed=42).slice(0, 40_000)  # re-ranker training queries (rerank_dev)
    ok = ~s1["idx"].is_in(bad.implode())
    if has_ce("train"):
        ok = ok & ~s1.select(is_ce_query(pl.col("entity_id"))).to_series()
    return all_q, all_q.filter(ok)


def oof(n_train_q: int, rounds: int) -> None:
    all_q, good = eligible_train_queries()
    fold = (all_q.hash(seed=7) % 2).cast(pl.Int8)
    preds = []
    for k in (0, 1):
        t = time.time()
        log.info(f"fold {k}: training on other fold, predicting {int((fold == k).sum()):,} queries")
        tr_q = good.filter(good.is_in(all_q.filter(fold != k).implode()))
        booster = _train(_training_rows(tr_q, n_train_q, seed=k), features("train"), rounds)
        preds.append(_predict(booster, "train", all_q.filter(fold == k)))
        log.info(f"fold {k}: {time.time() - t:.0f}s")
        if k == 0:
            imp = sorted(zip(booster.feature_importance("gain"), features("train")), reverse=True)[:20]
            print("top features:", [f for _, f in imp])
    pr = pl.concat(preds)
    pr.write_parquet(WORK / "train_oof.parquet")
    choose_rule(pr, good)


def choose_rule(pr: pl.DataFrame, eval_q: pl.Series) -> dict:
    gt = pl.read_parquet(WORK / "train_gt_pairs.parquet")
    gtl = gt.filter(pl.col("q_idx").is_in(eval_q.implode())).rename({"q_idx": "s1", "p_idx": "match"})

    def f(sel: pl.DataFrame) -> float:
        sel = sel.filter(pl.col("q_idx").is_in(eval_q.implode()))
        return macro_f05(sel.select(s1="q_idx", match="p_idx"), gtl, eval_q)

    results = {}
    oto = one_to_one(pr)
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        results[("threshold", False, t)] = f(pr.filter(pl.col("prob") >= t))
        results[("threshold", True, t)] = f(oto.filter(pl.col("prob") >= t))
    results[("expected", False, 0.0)] = f(expected_f05_sets(pr))
    results[("expected", True, 0.0)] = f(expected_f05_sets(oto))
    for (rule, o, t), v in sorted(results.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {rule:9s} one_to_one={o!s:5s} t={t:.1f}  F0.5={v:.4f}")
    (rule, o, t), v = max(results.items(), key=lambda kv: kv[1])
    best = {"rule": rule, "one_to_one": o, "threshold": t, "oof_f05": v}
    DECISION.write_text(json.dumps(best, indent=2))
    print("chosen:", best)
    return best


def apply_rule(pr: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    x = one_to_one(pr) if cfg["one_to_one"] else pr
    if cfg["rule"] == "expected":
        return expected_f05_sets(x).select("q_idx", "p_idx")
    return x.filter(pl.col("prob") >= cfg["threshold"]).select("q_idx", "p_idx")


def final(n_train_q: int, rounds: int) -> None:
    cfg = json.loads(DECISION.read_text())
    _, good = eligible_train_queries()
    booster = _train(_training_rows(good, n_train_q, seed=123), features("train"), rounds)
    booster.save_model(str(WORK / "matcher.txt"))
    test_q = pl.read_parquet(WORK / "test_s1.parquet", columns=["idx"])["idx"]
    pr = _predict(booster, "test", test_q)
    pr.write_parquet(WORK / "test_probs.parquet")
    matches = apply_rule(pr, cfg)
    cands = pr.select("q_idx", "p_idx")
    print(f"test: {matches.height:,} matches over {test_q.len():,} S1 "
          f"({matches['q_idx'].n_unique() / test_q.len():.3f} non-empty)")
    submit.write(matches, cands)
    assert submit.validate(), "submission failed validation"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["ctx", "oof", "final"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-train-q", type=int, default=300_000)
    ap.add_argument("--rounds", type=int, default=600)
    a = ap.parse_args()
    if a.step == "ctx":
        build_ctx(a.split)
    elif a.step == "oof":
        oof(a.n_train_q, a.rounds)
    else:
        final(a.n_train_q, a.rounds)
