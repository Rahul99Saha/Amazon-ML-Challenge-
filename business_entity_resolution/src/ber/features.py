"""Pairwise features for (S1 query, pool candidate) pairs. Vectorised: rapidfuzz cpdist
for string similarities, polars list set-ops for token/number overlap."""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

REC_COLS = [
    "idx", "entity_id", "name_core", "name_core_tokens", "name_compact", "name_alt",
    "name_legal", "name_native", "has_dba", "addr_norm", "addr_nums", "addr_words", "addr_len",
]


def _sim(a: pl.Series, b: pl.Series, scorer) -> np.ndarray:
    return process.cpdist(a.to_list(), b.to_list(), scorer=scorer, workers=-1, dtype=np.float32)


def _jacc(a: str, b: str) -> pl.Expr:
    inter = pl.col(a).list.set_intersection(pl.col(b)).list.len()
    union = pl.col(a).list.set_union(pl.col(b)).list.len()
    return pl.when(union > 0).then(inter / union).otherwise(None).cast(pl.Float32)


def attach_records(cand: pl.DataFrame, q: pl.DataFrame, p: pl.DataFrame) -> pl.DataFrame:
    return cand.join(q.select([pl.col(c).alias("q_" + c) for c in REC_COLS]), left_on="q_idx", right_on="q_idx").join(
        p.select([pl.col(c).alias("p_" + c) for c in REC_COLS]), left_on="p_idx", right_on="p_idx"
    )


def pair_features(x: pl.DataFrame, tok_idf: pl.DataFrame | None = None) -> pl.DataFrame:
    """x: output of attach_records (plus any blocking columns)."""
    qn, pn = x["q_name_core"], x["p_name_core"]
    qc, pc = x["q_name_compact"], x["p_name_compact"]
    qa, pa = x["q_addr_norm"], x["p_addr_norm"]
    feats = {
        "n_ratio": _sim(qn, pn, fuzz.ratio),
        "n_tset": _sim(qn, pn, fuzz.token_set_ratio),
        "n_tsort": _sim(qn, pn, fuzz.token_sort_ratio),
        "n_partial": _sim(qc, pc, fuzz.partial_ratio),
        "n_jw": _sim(qc, pc, JaroWinkler.normalized_similarity),
        "n_alt_tset": _sim(qn, x["p_name_alt"], fuzz.token_set_ratio),
        "a_ratio": _sim(qa, pa, fuzz.ratio),
        "a_tset": _sim(qa, pa, fuzz.token_set_ratio),
        "a_partial": _sim(qa, pa, fuzz.partial_token_set_ratio),
    }
    out = x.with_columns([pl.Series(k, v) for k, v in feats.items()]).with_columns(
        n_jacc=_jacc("q_name_core_tokens", "p_name_core_tokens"),
        a_num_jacc=_jacc("q_addr_nums", "p_addr_nums"),
        a_num_inter=pl.col("q_addr_nums").list.set_intersection(pl.col("p_addr_nums")).list.len().cast(pl.UInt8),
        a_word_jacc=_jacc("q_addr_words", "p_addr_words"),
        legal_eq=(pl.col("q_name_legal") == pl.col("p_name_legal")).cast(pl.Int8),
        legal_any_empty=((pl.col("q_name_legal") == "") | (pl.col("p_name_legal") == "")).cast(pl.Int8),
        p_addr_empty=(pl.col("p_addr_len") == 0).cast(pl.Int8),
        q_addr_empty=(pl.col("q_addr_len") == 0).cast(pl.Int8),
        p_native=pl.col("p_name_native").cast(pl.Int8),
        p_dba=pl.col("p_has_dba").cast(pl.Int8),
        p_is_s3=pl.col("p_entity_id").str.starts_with("S3-").cast(pl.Int8),
        n_len_q=pl.col("q_name_core").str.len_chars().cast(pl.UInt16),
        n_len_p=pl.col("p_name_core").str.len_chars().cast(pl.UInt16),
        a_len_q=pl.col("q_addr_len").cast(pl.UInt16),
        a_len_p=pl.col("p_addr_len").cast(pl.UInt16),
        n_tok_q=pl.col("q_name_core_tokens").list.len().cast(pl.UInt8),
        n_tok_p=pl.col("p_name_core_tokens").list.len().cast(pl.UInt8),
    )
    if tok_idf is not None:
        out = out.with_columns(_idf_overlap(out, tok_idf))
    return out


def _idf_overlap(x: pl.DataFrame, tok_idf: pl.DataFrame) -> pl.Series:
    """Share of the query's name IDF mass that the candidate also contains."""
    q = (
        x.select(r=pl.int_range(pl.len()), t=pl.col("q_name_core_tokens"), pt=pl.col("p_name_core_tokens"))
        .explode("t")
        .drop_nulls("t")
        .join(tok_idf, on="t", how="left")
        .with_columns(pl.col("idf").fill_null(tok_idf["idf"].max()))
        .with_columns(hit=pl.col("pt").list.contains(pl.col("t")))
        .group_by("r")
        .agg(num=(pl.col("idf") * pl.col("hit")).sum(), den=pl.col("idf").sum())
    )
    r = pl.DataFrame({"r": np.arange(x.height)}).join(q, on="r", how="left").sort("r")
    return (r["num"] / r["den"]).fill_null(0).cast(pl.Float32).alias("n_idf_cover")


NG_FIELDS = ["name", "name_addr"]
NG_FEATURES = [f"ng_{f}" for f in NG_FIELDS] + [f"ng_{f}_rank" for f in NG_FIELDS] + ["from_keys", "from_ng"]

EMB_FIELDS = ["name", "name_addr"]
EMB_FEATURES = [f"emb_{f}" for f in EMB_FIELDS] + [f"emb_{f}_rank" for f in EMB_FIELDS] + ["from_emb"]

FEATURES = NG_FEATURES + EMB_FEATURES + [
    "block_score", "block_rank", "bk_name", "bk_compact", "bk_addr", "bk_addrword",
    "bk_namepair", "bk_addrpair", "bk_nameaddr", "bk_compact10",
    "n_ratio", "n_tset", "n_tsort", "n_partial", "n_jw", "n_alt_tset", "a_ratio", "a_tset",
    "a_partial", "n_jacc", "a_num_jacc", "a_num_inter", "a_word_jacc", "legal_eq",
    "legal_any_empty", "p_addr_empty", "q_addr_empty", "p_native", "p_dba", "p_is_s3",
    "n_len_q", "n_len_p", "a_len_q", "a_len_p", "n_tok_q", "n_tok_p", "n_idf_cover",
]


def name_token_idf(pool_path) -> pl.DataFrame:
    df = (
        pl.scan_parquet(pool_path)
        .select(t=pl.col("name_core_tokens"))
        .explode("t")
        .drop_nulls("t")
        .group_by("t")
        .len("df")
        .collect(engine="streaming")
    )
    n = pl.scan_parquet(pool_path).select(pl.len()).collect().item()
    return df.select("t", idf=(n / pl.col("df")).log().cast(pl.Float32))
