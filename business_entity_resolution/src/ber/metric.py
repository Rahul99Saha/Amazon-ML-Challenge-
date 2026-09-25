import polars as pl


def macro_f05(pred: pl.DataFrame, gt: pl.DataFrame, s1_ids: pl.Series | None = None) -> float:
    """pred, gt: long frames with columns s1, match (match null = no match row).

    Mirrors the official scorer: F0.5 per S1 entity, averaged over all S1 entities;
    an entity with no true matches scores 1 iff the prediction is empty.
    """
    pred = pred.filter(pl.col("match").is_not_null()).select("s1", "match").unique()
    gt_pairs = gt.filter(pl.col("match").is_not_null()).select("s1", "match").unique()
    if s1_ids is None:
        s1_ids = gt["s1"].unique()
    base = pl.DataFrame({"s1": s1_ids.unique()})
    n_pred = pred.group_by("s1").len("n_pred")
    n_true = gt_pairs.group_by("s1").len("n_true")
    tp = pred.join(gt_pairs, on=["s1", "match"], how="inner").group_by("s1").len("tp")
    per = (
        base.join(n_pred, on="s1", how="left")
        .join(n_true, on="s1", how="left")
        .join(tp, on="s1", how="left")
        .fill_null(0)
        .with_columns(
            p=pl.col("tp") / pl.col("n_pred"),
            r=pl.col("tp") / pl.col("n_true"),
        )
        .with_columns(
            f=pl.when(pl.col("n_true") == 0)
            .then((pl.col("n_pred") == 0).cast(pl.Float64))
            .when(pl.col("tp") == 0)
            .then(0.0)
            .otherwise(1.25 * pl.col("p") * pl.col("r") / (0.25 * pl.col("p") + pl.col("r")))
        )
    )
    return float(per["f"].mean())
