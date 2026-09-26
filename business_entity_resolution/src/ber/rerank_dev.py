"""Train the candidate re-ranker on union candidates (rare keys + n-gram BM25) of a
random set of train S1 queries; report channel recall and recall@K on a disjoint set.

The first 40k queries of shuffle(seed=42) are the training set; matcher.py excludes
them from matcher training/evaluation because their re-ranker scores are optimistic.
"""
import argparse
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from ber.candidates import featurize, reranker_path, token_idf, union_candidates
from ber.embed import DEFAULT_MODEL, EmbeddingIndex
from ber.features import FEATURES, FEATURES_EMB
from ber.io import WORK
from ber.log import get, timed
from ber.ngram import NgramIndex

log = get("rerank")


def build(q_sets: dict[str, pl.Series], block_k: int, ng_k: int, weighting: str,
          embed: bool = False, embed_model: str = DEFAULT_MODEL, emb_k: int = 50,
          tag: str = "bm25") -> dict[str, pl.DataFrame]:
    """Features + labels for every query set, fitting one n-gram (+ optional embedding)
    index per country.

    Checkpoints per country to work/dev_feats_{tag}_{country}.parquet (atomic tmp+rename).
    A country whose indices are already fit and features already built (e.g. India,
    finished overnight) is loaded from its checkpoint instead of redone; only countries
    without one (e.g. the US pool, picked up the next day) do the expensive work. This is
    on top of, not instead of, embed.py's own per-field embedding cache: that protects the
    encode step, this protects the retrieval-and-feature-building step built on top of it.
    """
    gt = pl.read_parquet(WORK / "train_gt_pairs.parquet").with_columns(label=pl.lit(1, pl.Int8))
    tok_idf = token_idf("train")
    country = pl.read_parquet(WORK / "train_s1.parquet", columns=["idx", "country"])
    parts = {k: [] for k in q_sets}
    for ctry in sorted(country["country"].unique().to_list()):
        ctry_slug = ctry.replace(" ", "_")
        ckpt = WORK / f"dev_feats_{tag}_{ctry_slug}.parquet"
        if ckpt.exists():
            log.info(f"{ctry}: loading checkpointed features from {ckpt.name}")
            done = pl.read_parquet(ckpt)
            for name in q_sets:
                parts[name].append(done.filter(pl.col("qset") == name).drop("qset"))
            continue
        in_c = country.filter(pl.col("country") == ctry)["idx"]
        with timed(log, f"{ctry}: fit n-gram index"):
            ng = NgramIndex("train", ctry, weighting=weighting)
        emb = None
        if embed:
            with timed(log, f"{ctry}: fit embedding index"):
                emb = EmbeddingIndex("train", ctry, model_name=embed_model)
        ctry_parts = []
        for name, qs in q_sets.items():
            q = qs.filter(qs.is_in(in_c.implode()))
            if len(q) == 0:
                continue
            log.info(f"{ctry}: {name} set, {len(q):,} queries")
            cand = union_candidates("train", q, ng, block_k, ng_k, emb=emb, emb_k=emb_k)
            for f in featurize("train", cand, ng, tok_idf, emb=emb, label=f"{ctry} {name} features"):
                labeled = f.join(gt, on=["q_idx", "p_idx"], how="left").with_columns(pl.col("label").fill_null(0))
                parts[name].append(labeled)
                ctry_parts.append(labeled.with_columns(qset=pl.lit(name)))
        del ng
        if emb is not None:
            del emb
        tmp = ckpt.with_name(ckpt.stem + ".tmp.parquet")
        pl.concat(ctry_parts).write_parquet(tmp)
        tmp.rename(ckpt)
        log.info(f"{ctry}: checkpointed {sum(p.height for p in ctry_parts):,} feature rows -> {ckpt.name}")
    return {k: pl.concat(v) for k, v in parts.items()}


def recall_at(feats: pl.DataFrame, score: np.ndarray, gt: pl.DataFrame, ks=(5, 10, 15, 20, 30, 50)) -> None:
    r = feats.select("q_idx", "p_idx").with_columns(s=pl.Series(score)).with_columns(
        rk=pl.col("s").rank("ordinal", descending=True).over("q_idx")
    )
    hit = gt.join(r, on=["q_idx", "p_idx"], how="left")
    print("  " + "  ".join(f"@{k}:{(hit['rk'].fill_null(10**6) <= k).mean():.4f}" for k in ks))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=40_000)
    ap.add_argument("--n-eval", type=int, default=20_000)
    ap.add_argument("--block-k", type=int, default=200)
    ap.add_argument("--ng-k", type=int, default=50)
    ap.add_argument("--weighting", default="bm25", choices=["bm25", "tfidf"])
    ap.add_argument("--embed", action="store_true", help="add the dense embedding candidate channel (embed.py)")
    ap.add_argument("--embed-model", default=DEFAULT_MODEL, help="sentence-transformers model name")
    ap.add_argument("--emb-k", type=int, default=50)
    a = ap.parse_args()
    tag = a.weighting + ("_emb" if a.embed else "")
    feats = FEATURES_EMB if a.embed else FEATURES

    all_q = pl.read_parquet(WORK / "train_s1.parquet", columns=["idx"])["idx"].shuffle(seed=42)
    q_tr, q_ev = all_q.slice(0, a.n_train), all_q.slice(a.n_train, a.n_eval)
    gt = pl.read_parquet(WORK / "train_gt_pairs.parquet")
    gt_ev = gt.filter(pl.col("q_idx").is_in(q_ev.implode()))

    t = time.time()
    tr_path, ev_path = WORK / f"dev_feats_train_{tag}.parquet", WORK / f"dev_feats_eval_{tag}.parquet"
    if not (tr_path.exists() and ev_path.exists()):
        sets = build({"train": q_tr, "eval": q_ev}, a.block_k, a.ng_k, a.weighting, a.embed, a.embed_model, a.emb_k, tag)
        sets["train"].write_parquet(tr_path)
        sets["eval"].write_parquet(ev_path)
    tr, ev = pl.read_parquet(tr_path), pl.read_parquet(ev_path)
    print(f"[{tag}] features {time.time() - t:.0f}s  train pairs {tr.height:,}  eval pairs {ev.height:,}  "
          f"({ev.height / a.n_eval:.0f}/query)")

    chan_cols = ["from_keys", "from_ng"] + (["from_emb"] if a.embed else [])
    hit = gt_ev.join(ev.select("q_idx", "p_idx", *chan_cols), on=["q_idx", "p_idx"], how="left")
    in_union = hit["from_keys"].is_not_null()
    msg = (f"generation recall  keys only {(hit['from_keys'].fill_null(0) == 1).mean():.4f}   "
           f"n-gram only {(hit['from_ng'].fill_null(0) == 1).mean():.4f}   ")
    if a.embed:
        msg += f"embedding only {(hit['from_emb'].fill_null(0) == 1).mean():.4f}   "
    print(msg + f"union {in_union.mean():.4f}")
    print("blocking score only:")
    recall_at(ev, ev["block_score"].to_numpy(), gt_ev)
    ng_fields = ["ng_name", "ng_name_addr"] + (["emb_name", "emb_name_addr"] if a.embed else [])
    for f in ng_fields:
        print(f"{f} score only:")
        recall_at(ev, ev[f].to_numpy(), gt_ev)

    model = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.08, num_leaves=63, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
    )
    with timed(log, f"train re-ranker on {tr.height:,} pairs"):
        model.fit(tr.select(feats).to_numpy(), tr["label"].to_numpy())
    print("re-ranker:")
    recall_at(ev, model.predict_proba(ev.select(feats).to_numpy())[:, 1], gt_ev)
    imp = sorted(zip(model.booster_.feature_importance("gain"), feats), reverse=True)[:15]
    print("top features:", [f for _, f in imp])
    path = reranker_path(a.weighting, a.embed)
    model.booster_.save_model(str(path))
    print(f"saved re-ranker to {path}")


if __name__ == "__main__":
    main()
