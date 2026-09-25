"""Second-stage context features over the full candidate graph of a split.

- query context: rank / gap / mass of the re-ranker scores within each S1's list
- pool competition: how many S1s claim the same pool record, and whether a
  competing S1 scores it higher (co-located businesses share addresses)
- cluster consistency: similarity of a candidate to the query's other strong
  candidates (true S2/S3 duplicates of one entity resemble each other)
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from ber.io import WORK

CONTEXT_FEATURES = [
    "rr_prob", "rr_rank", "q_gap_top", "q_n_strong", "q_mass", "q_top1",
    "p_n_claims", "p_rank_among_q", "p_best_other", "p_margin_other",
    "cl_name_max", "cl_addr_max", "cl_name_mean", "cl_addr_mean",
]


def _sim(a: list, b: list, scorer) -> np.ndarray:
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def competition(slim: pl.DataFrame) -> pl.DataFrame:
    """Needs every query of the split at once; slim = (q_idx, p_idx, rr_prob)."""
    return slim.with_columns(
        p_n_claims=pl.len().over("p_idx").cast(pl.UInt16),
        p_rank_among_q=pl.col("rr_prob").rank("ordinal", descending=True).over("p_idx").cast(pl.UInt16),
        _p1=pl.col("rr_prob").max().over("p_idx"),
        _p2=pl.col("rr_prob").sort(descending=True).get(1, null_on_oob=True).over("p_idx").fill_null(0.0),
    ).with_columns(
        p_best_other=pl.when(pl.col("p_rank_among_q") == 1).then(pl.col("_p2")).otherwise(pl.col("_p1")),
    ).with_columns(p_margin_other=pl.col("rr_prob") - pl.col("p_best_other")).select(
        "q_idx", "p_idx", "p_n_claims", "p_rank_among_q", "p_best_other", "p_margin_other"
    )


def add_context(cand: pl.DataFrame, split: str, anchors: int = 3, strong: float = 0.5) -> pl.DataFrame:
    """Query-level and cluster features; safe to run on any subset of whole queries."""
    c = cand.with_columns(
        q_top1=pl.col("rr_prob").max().over("q_idx"),
        q_mass=pl.col("rr_prob").sum().over("q_idx"),
        q_n_strong=(pl.col("rr_prob") >= strong).sum().over("q_idx").cast(pl.UInt8),
    ).with_columns(q_gap_top=pl.col("q_top1") - pl.col("rr_prob"))

    return c.with_columns(_cluster(c, split, anchors))


def _cluster(c: pl.DataFrame, split: str, anchors: int) -> list[pl.Series]:
    """For each candidate: similarity to the query's top-`anchors` candidates (excluding itself)."""
    anc = c.filter(pl.col("rr_rank") <= anchors).select("q_idx", a_idx="p_idx")
    pairs = c.select("q_idx", "p_idx", r=pl.int_range(pl.len())).join(anc, on="q_idx").filter(
        pl.col("a_idx") != pl.col("p_idx")
    )
    need = pl.concat([pairs["p_idx"], pairs["a_idx"]]).unique()
    recs = (
        pl.scan_parquet(WORK / f"{split}_pool.parquet")
        .select("idx", "name_core", "addr_norm")
        .filter(pl.col("idx").is_in(need.implode()))
        .collect()
    )
    pairs = pairs.join(recs.rename({"idx": "p_idx", "name_core": "pn", "addr_norm": "pa"}), on="p_idx").join(
        recs.rename({"idx": "a_idx", "name_core": "an", "addr_norm": "aa"}), on="a_idx"
    )
    pairs = pairs.with_columns(
        sn=pl.Series(_sim(pairs["pn"].to_list(), pairs["an"].to_list(), fuzz.token_set_ratio)),
        sa=pl.Series(_sim(pairs["pa"].to_list(), pairs["aa"].to_list(), fuzz.token_set_ratio)),
    ).with_columns(
        # empty address on either side carries no evidence
        sa=pl.when((pl.col("pa") == "") | (pl.col("aa") == "")).then(None).otherwise(pl.col("sa"))
    )
    agg = pairs.group_by("r").agg(
        cl_name_max=pl.col("sn").max(), cl_addr_max=pl.col("sa").max(),
        cl_name_mean=pl.col("sn").mean(), cl_addr_mean=pl.col("sa").mean(),
    )
    full = pl.DataFrame({"r": np.arange(c.height)}).join(agg, on="r", how="left").sort("r")
    return [full[k].cast(pl.Float32) for k in ("cl_name_max", "cl_addr_max", "cl_name_mean", "cl_addr_mean")]
