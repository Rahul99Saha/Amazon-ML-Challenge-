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
# "twin" features: separate a true match from a near-identical distractor (same street,
# house number shifted a little, different legal form, identical name without address)
TWIN_FEATURES = [
    "t_name_core_eq", "t_name_compact_eq", "t_legal_conflict", "t_num_mindiff_log",
    "t_num_min_reldiff", "t_name_eq_addr_empty", "t_num_count_q", "t_num_count_p",
]
CONTEXT_FEATURES = CONTEXT_FEATURES + TWIN_FEATURES
_TWIN_COLS = ["idx", "name_core", "name_compact", "name_legal", "addr_nums", "addr_len"]


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


def add_twin(c: pl.DataFrame, split: str) -> pl.DataFrame:
    """Adds TWIN_FEATURES to a candidate frame (q_idx, p_idx, ...)."""
    q = (pl.scan_parquet(WORK / f"{split}_s1.parquet").select(_TWIN_COLS)
         .filter(pl.col("idx").is_in(c["q_idx"].unique().implode())).collect())
    p = (pl.scan_parquet(WORK / f"{split}_pool.parquet").select(_TWIN_COLS)
         .filter(pl.col("idx").is_in(c["p_idx"].unique().implode())).collect())
    x = (c.select("q_idx", "p_idx").with_row_index("r")
         .join(q.rename({k: f"q_{k}" for k in _TWIN_COLS}), left_on="q_idx", right_on="q_idx")
         .join(p.rename({k: f"p_{k}" for k in _TWIN_COLS}), left_on="p_idx", right_on="p_idx"))

    def nums(col: str) -> pl.Expr:
        return pl.col(col).list.eval(pl.element().str.slice(0, 9).cast(pl.Int64, strict=False)).list.drop_nulls()

    x = x.with_columns(qn=nums("q_addr_nums"), pn=nums("p_addr_nums"))
    # closest pair of address numbers (house / plot numbers), absolute and relative
    d = (x.select("r", "qn", "pn").explode("qn").explode("pn").drop_nulls()
         .with_columns(ad=(pl.col("qn") - pl.col("pn")).abs())
         .with_columns(rd=pl.col("ad") / pl.max_horizontal(pl.col("qn").abs(), pl.col("pn").abs(), pl.lit(1)))
         .group_by("r").agg(mind=pl.col("ad").min(), minrd=pl.col("rd").min()))
    x = x.join(d, on="r", how="left")
    legal_q, legal_p = pl.col("q_name_legal").fill_null(""), pl.col("p_name_legal").fill_null("")
    x = x.select(
        "r",
        t_name_core_eq=(pl.col("q_name_core") == pl.col("p_name_core")).cast(pl.Int8),
        t_name_compact_eq=(pl.col("q_name_compact") == pl.col("p_name_compact")).cast(pl.Int8),
        t_legal_conflict=((legal_q != "") & (legal_p != "") & (legal_q != legal_p)).cast(pl.Int8),
        t_num_mindiff_log=(pl.col("mind").cast(pl.Float64) + 1).log().cast(pl.Float32),
        t_num_min_reldiff=pl.col("minrd").cast(pl.Float32),
        t_name_eq_addr_empty=((pl.col("q_name_core") == pl.col("p_name_core")) & (pl.col("p_addr_len") == 0)).cast(pl.Int8),
        t_num_count_q=pl.col("qn").list.len().cast(pl.UInt8),
        t_num_count_p=pl.col("pn").list.len().cast(pl.UInt8),
    )
    out = c.with_row_index("r").join(x, on="r", how="left").drop("r")
    assert out.height == c.height
    return out


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
