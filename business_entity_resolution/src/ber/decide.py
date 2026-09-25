"""Turn pair probabilities into per-S1 match sets that maximise expected macro F0.5."""
import polars as pl


def one_to_one(scores: pl.DataFrame, margin: float = 0.0) -> pl.DataFrame:
    """Each pool record keeps only its best S1; drop it everywhere if the runner-up is
    within `margin` (ambiguous)."""
    r = scores.with_columns(
        _rk=pl.col("prob").rank("ordinal", descending=True).over("p_idx"),
        _second=pl.col("prob").sort(descending=True).get(1, null_on_oob=True).over("p_idx").fill_null(0.0),
    )
    return r.filter((pl.col("_rk") == 1) & (pl.col("prob") - pl.col("_second") >= margin)).drop("_rk", "_second")


def expected_f05_sets(scores: pl.DataFrame, max_k: int = 12, floor: float = 0.05) -> pl.DataFrame:
    """For each query choose k (0..max_k) of its top candidates maximising
    E[F0.5] ~= 1.25 * sum(p_top_k) / (k + 0.25 * N), N = expected true count;
    k=0 scores P(no match) = prod(1 - p). Returns selected (q_idx, p_idx, prob)."""
    s = (
        scores.filter(pl.col("prob") >= floor)
        .sort(["q_idx", "prob"], descending=[False, True])
        .with_columns(k=pl.int_range(1, pl.len() + 1).over("q_idx"))
        .filter(pl.col("k") <= max_k)
    )
    stats = scores.group_by("q_idx").agg(
        n_exp=pl.col("prob").sum(),
        p_none=(1 - pl.col("prob")).log().sum().exp(),
    )
    s = s.join(stats, on="q_idx").with_columns(tp=pl.col("prob").cum_sum().over("q_idx")).with_columns(
        ef=1.25 * pl.col("tp") / (pl.col("k") + 0.25 * pl.col("n_exp").clip(lower_bound=1e-6))
    )
    best = s.group_by("q_idx").agg(best_k=pl.col("k").get(pl.col("ef").arg_max()), best_ef=pl.col("ef").max())
    best = best.join(stats.select("q_idx", "p_none"), on="q_idx").filter(pl.col("best_ef") > pl.col("p_none"))
    return s.join(best.select("q_idx", "best_k"), on="q_idx").filter(pl.col("k") <= pl.col("best_k")).select(
        "q_idx", "p_idx", "prob"
    )
