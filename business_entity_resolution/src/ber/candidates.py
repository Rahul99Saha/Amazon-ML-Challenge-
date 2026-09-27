"""Candidate generation = union of two or three channels, then a learned re-ranker.

  channel 1: rare-key inverted index (blocking.py), top `block_k` by IDF score
  channel 2: character n-gram BM25 (ngram.py), top `ng_k` per field
  channel 3 (optional, --embed): dense multilingual sentence embeddings (embed.py),
    top `emb_k` per field -- targets train-only-learned channels' blind spot: native
    scripts and France (see embed.py's docstring)
  union -> pair features (string sims + n-gram + embedding scores for every pair) ->
  LightGBM re-ranker -> top `keep_k` per S1 kept with all features.

Processed per country (one n-gram / embedding index fit per country) and in
super-chunks of queries written to disk, so RAM stays bounded and interrupted runs
resume.

Output: work/{split}_cand.parquet  (q_idx, p_idx, FEATURES..., rr_prob, rr_rank)
"""
import argparse

import lightgbm as lgb
import numpy as np
import polars as pl

from ber.blocking import KINDS, generate_candidates
from ber.embed import DEFAULT_MODEL, EmbeddingIndex
from ber.features import FEATURES, FEATURES_EMB, REC_COLS, attach_records, name_token_idf, pair_features
from ber.io import WORK
from ber.log import Progress, get, timed
from ber.ngram import NgramIndex

RERANKER = WORK / "reranker.txt"
log = get("cand")


def reranker_path(weighting: str, embed: bool):
    """Matches rerank_dev.py's save path: reranker.txt only for the original
    (bm25, no embedding) config, so existing runs/artifacts are untouched by default."""
    tag = weighting + ("_emb" if embed else "")
    return RERANKER if tag == "bm25" else WORK / f"reranker_{tag}.txt"


def union_candidates(split: str, q_ids: pl.Series, ng: NgramIndex, block_k: int, ng_k: int,
                      emb: EmbeddingIndex | None = None, emb_k: int = 50) -> pl.DataFrame:
    with timed(log, f"key-index candidates for {len(q_ids):,} queries"):
        keys = generate_candidates(split, q_ids, top_k=block_k).with_columns(from_keys=pl.lit(1, pl.Int8))
    with timed(log, f"n-gram top-{ng_k} for {len(q_ids):,} queries"):
        ngc = ng.topk(q_ids, top_k=ng_k).with_columns(from_ng=pl.lit(1, pl.Int8))
    u = keys.join(ngc, on=["q_idx", "p_idx"], how="full", coalesce=True)
    n_emb = 0
    if emb is not None:
        with timed(log, f"embedding top-{emb_k} for {len(q_ids):,} queries"):
            embc = emb.topk(q_ids, top_k=emb_k).with_columns(from_emb=pl.lit(1, pl.Int8))
        n_emb = embc.height
        u = u.join(embc, on=["q_idx", "p_idx"], how="full", coalesce=True)
    log.info(f"union: {u.height:,} pairs ({u.height / max(len(q_ids), 1):.0f}/query; "
             f"keys {keys.height:,}, n-gram {ngc.height:,}" + (f", embedding {n_emb:,}" if emb is not None else "") + ")")
    fills = [
        pl.col("from_keys", "from_ng").fill_null(0),
        pl.col("block_score").fill_null(0.0),
        pl.col("block_rank").fill_null(65535),
        *[pl.col(f"bk_{k}").fill_null(0) for k in KINDS],
    ]
    if emb is not None:
        fills.append(pl.col("from_emb").fill_null(0))
    u = u.with_columns(fills)
    return u if emb is not None else u.with_columns(from_emb=pl.lit(0, pl.Int8))


def featurize(split: str, cand: pl.DataFrame, ng: NgramIndex, tok_idf: pl.DataFrame,
              emb: EmbeddingIndex | None = None, chunk: int = 5_000, label: str = "features"):
    """Yields feature frames (q_idx, p_idx, FEATURES) for chunks of queries."""
    q_ids = cand["q_idx"].unique().sort()
    prog = Progress(log, f"{label} (queries)", len(q_ids))
    q = pl.scan_parquet(WORK / f"{split}_s1.parquet").select(REC_COLS).filter(pl.col("idx").is_in(q_ids.implode())).collect()
    p = (
        pl.scan_parquet(WORK / f"{split}_pool.parquet").select(REC_COLS)
        .filter(pl.col("idx").is_in(cand["p_idx"].unique().implode())).collect()
    )
    for s in range(0, len(q_ids), chunk):
        c = cand.filter(pl.col("q_idx").is_in(q_ids.slice(s, chunk).implode()))
        n_pairs = c.height
        c = ng.pair_features(c)
        feats = FEATURES
        if emb is not None:
            c = emb.pair_features(c)
            feats = FEATURES_EMB
        yield pair_features(attach_records(c, q, p), tok_idf).select(["q_idx", "p_idx", *feats])
        prog.step(min(chunk, len(q_ids) - s), pairs=f"{n_pairs:,}")


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
    log.info(f"{split}: {n:,} candidate pairs ({n / n_q:.1f}/query) -> {out}")


def run(split: str, block_k: int, ng_k: int, keep_k: int, super_chunk: int, weighting: str,
        countries: list[str] | None, shard: tuple[int, int] = (0, 1),
        embed: bool = False, embed_model: str = DEFAULT_MODEL, emb_k: int = 50) -> None:
    log.info(f"candidates split={split} countries={countries or 'all'} shard={shard[0]}/{shard[1]} "
             f"block_k={block_k} ng_k={ng_k} keep_k={keep_k} super_chunk={super_chunk} weighting={weighting} "
             f"embed={embed}" + (f" embed_model={embed_model} emb_k={emb_k}" if embed else ""))
    with timed(log, "name-token idf"):
        tok_idf = token_idf(split)
    rr_path = reranker_path(weighting, embed)
    feats = FEATURES_EMB if embed else FEATURES
    log.info(f"loading re-ranker {rr_path} ({len(feats)} features)")
    booster = lgb.Booster(model_file=str(rr_path))
    s1 = pl.read_parquet(WORK / f"{split}_s1.parquet", columns=["idx", "country"])
    for ctry in sorted(s1["country"].unique().to_list()):
        if countries and ctry not in countries:
            continue
        q_all = s1.filter(pl.col("country") == ctry)["idx"]
        plan = [(part_path(split, ctry, i), s) for i, s in enumerate(range(0, len(q_all), super_chunk))
                if i % shard[1] == shard[0]]
        todo = [(path, s) for path, s in plan if not path.exists()]
        if not todo:
            log.info(f"{ctry}: all {len(plan)} super-chunks already done")
            continue
        n_todo = sum(min(super_chunk, len(q_all) - s) for _, s in todo)
        log.info(f"{ctry}: {len(q_all):,} S1 in country; this job: {len(todo)} super-chunks, {n_todo:,} queries")
        with timed(log, f"{ctry}: fit n-gram index"):
            ng = NgramIndex(split, ctry, weighting=weighting)
        emb = None
        if embed:
            with timed(log, f"{ctry}: fit embedding index"):
                emb = EmbeddingIndex(split, ctry, model_name=embed_model)
        overall = Progress(log, f"{ctry} OVERALL (queries)", n_todo, every=0)
        for k, (path, s) in enumerate(todo):
            q_ids = q_all.slice(s, super_chunk)
            log.info(f"{ctry}: super-chunk {k + 1}/{len(todo)} ({len(q_ids):,} queries) -> {path.name}")
            cand = union_candidates(split, q_ids, ng, block_k, ng_k, emb=emb, emb_k=emb_k)
            kept = []
            for f in featurize(split, cand, ng, tok_idf, emb=emb, label=f"{ctry} sc{k + 1}/{len(todo)} features"):
                f = f.with_columns(rr_prob=pl.Series(booster.predict(f.select(feats).to_numpy()).astype(np.float32)))
                f = f.with_columns(rr_rank=pl.col("rr_prob").rank("ordinal", descending=True).over("q_idx").cast(pl.UInt16))
                kept.append(f.filter(pl.col("rr_rank") <= keep_k))
            out = pl.concat(kept)
            out.write_parquet(path)
            overall.step(len(q_ids), kept_pairs=f"{out.height:,}")
            del cand, kept, out
        del ng
        if emb is not None:
            del emb


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
    ap.add_argument("--embed", action="store_true", help="add the dense embedding candidate channel (embed.py)")
    ap.add_argument("--embed-model", default=DEFAULT_MODEL, help="sentence-transformers model name")
    ap.add_argument("--emb-k", type=int, default=50)
    a = ap.parse_args()
    shard = tuple(int(x) for x in a.shard.split("/"))
    for sp in a.splits:
        if not a.merge:
            run(sp, a.block_k, a.ng_k, a.keep_k, a.super_chunk, a.weighting, a.countries, shard,
                a.embed, a.embed_model, a.emb_k)
        if a.merge or not a.countries:
            merge(sp)
