"""Error analysis of out-of-fold matcher predictions (which slices lose F0.5, and why)."""
import argparse

import polars as pl

from ber.io import WORK
from ber.log import get
from ber.matcher import apply_rule, eligible_train_queries
from ber.metric import macro_f05

log = get("analysis")


def per_entity(sel: pl.DataFrame, gt: pl.DataFrame, q: pl.Series) -> pl.DataFrame:
    """Per-S1 tp / n_pred / n_true / F0.5 (official definition)."""
    pred = sel.select("q_idx", "p_idx").unique()
    base = pl.DataFrame({"q_idx": q})
    n_pred = pred.group_by("q_idx").len("n_pred")
    n_true = gt.group_by("q_idx").len("n_true")
    tp = pred.join(gt, on=["q_idx", "p_idx"]).group_by("q_idx").len("tp")
    return (
        base.join(n_pred, on="q_idx", how="left").join(n_true, on="q_idx", how="left").join(tp, on="q_idx", how="left")
        .fill_null(0)
        .with_columns(p=pl.col("tp") / pl.col("n_pred"), r=pl.col("tp") / pl.col("n_true"))
        .with_columns(
            f=pl.when(pl.col("n_true") == 0).then((pl.col("n_pred") == 0).cast(pl.Float64))
            .when(pl.col("tp") == 0).then(0.0)
            .otherwise(1.25 * pl.col("p") * pl.col("r") / (0.25 * pl.col("p") + pl.col("r")))
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="train_oof.parquet")
    ap.add_argument("--decision", default="decision.json")
    ap.add_argument("--examples", type=int, default=40)
    a = ap.parse_args()
    import json

    cfg = json.loads((WORK / a.decision).read_text())
    pr = pl.read_parquet(WORK / a.oof)
    _, q = eligible_train_queries()
    gt = pl.read_parquet(WORK / "train_gt_pairs.parquet").filter(pl.col("q_idx").is_in(q.implode()))
    sel = apply_rule(pr, cfg).filter(pl.col("q_idx").is_in(q.implode()))
    log.info(f"rule {cfg}  eval S1 {len(q):,}  predicted pairs {sel.height:,}  true pairs {gt.height:,}")

    ent = per_entity(sel, gt, q)
    log.info(f"macro F0.5 {ent['f'].mean():.4f}  (check: {macro_f05(sel.select(s1='q_idx', match='p_idx'), gt.rename({'q_idx': 's1', 'p_idx': 'match'}), q):.4f})")
    micro_p = ent["tp"].sum() / max(ent["n_pred"].sum(), 1)
    micro_r = ent["tp"].sum() / max(ent["n_true"].sum(), 1)
    log.info(f"pair-level precision {micro_p:.4f}  recall {micro_r:.4f}")

    # ceiling: perfect decisions on our candidates
    cand = pl.scan_parquet(WORK / "train_cand.parquet").select("q_idx", "p_idx", "rr_rank").filter(
        pl.col("q_idx").is_in(q.implode())).collect()
    oracle = cand.join(gt, on=["q_idx", "p_idx"])
    log.info(f"ceiling (perfect matcher on top-20): {per_entity(oracle, gt, q)['f'].mean():.4f}")

    # attributes for slicing
    s1 = pl.read_parquet(WORK / "train_s1.parquet", columns=["idx", "country", "name_native", "addr_len"]).rename({"idx": "q_idx"})
    ent = ent.join(s1, on="q_idx").with_columns(
        true_bucket=pl.when(pl.col("n_true") == 0).then(pl.lit("0 (singleton)"))
        .when(pl.col("n_true") == 1).then(pl.lit("1"))
        .when(pl.col("n_true") <= 3).then(pl.lit("2-3")).otherwise(pl.lit("4+")),
        addr_empty=pl.col("addr_len") == 0,
    )
    total_loss = (1 - ent["f"]).sum()
    with pl.Config(tbl_rows=30, tbl_width_chars=200):
        for col in ("country", "true_bucket", "addr_empty"):
            t = ent.group_by(col).agg(
                n=pl.len(), f05=pl.col("f").mean(), loss_share=(1 - pl.col("f")).sum() / total_loss,
                prec=pl.col("tp").sum() / pl.col("n_pred").sum(), rec=pl.col("tp").sum() / pl.col("n_true").sum(),
                empty_pred=(pl.col("n_pred") == 0).mean(),
            ).sort(col)
            print(f"\n== by {col} ==\n{t}", flush=True)

        # error decomposition
        fn = gt.join(sel, on=["q_idx", "p_idx"], how="anti")
        fn = fn.join(cand, on=["q_idx", "p_idx"], how="left").with_columns(
            cause=pl.when(pl.col("rr_rank").is_null()).then(pl.lit("not in top-20 candidates"))
            .otherwise(pl.lit("in candidates, rejected by matcher"))).join(pr, on=["q_idx", "p_idx"], how="left")
        print(f"\n== missed true pairs: {fn.height:,} ==\n{fn.group_by('cause').len().sort('len', descending=True)}", flush=True)
        fp = sel.join(gt, on=["q_idx", "p_idx"], how="anti").join(pr, on=["q_idx", "p_idx"], how="left")
        owner = pl.read_parquet(WORK / "train_gt_pairs.parquet").rename({"q_idx": "true_owner"})
        fp = fp.join(owner, on="p_idx", how="left").with_columns(
            kind=pl.when(pl.col("true_owner").is_null()).then(pl.lit("pool record belongs to no S1 (distractor)"))
            .otherwise(pl.lit("pool record belongs to another S1")))
        print(f"\n== false-positive pairs: {fp.height:,} ==\n{fp.group_by('kind').len().sort('len', descending=True)}", flush=True)
        print(f"FP prob quantiles: {fp['prob'].quantile(0.1):.3f} / {fp['prob'].median():.3f} / {fp['prob'].quantile(0.9):.3f}", flush=True)

        # errors by attributes of the S2/S3 record involved
        pool_attr = pl.scan_parquet(WORK / "train_pool.parquet").select(
            p_idx="idx", cand_native="name_native", cand_addr_empty=pl.col("addr_len") == 0,
            cand_src=pl.col("entity_id").str.slice(0, 2)).filter(
            pl.col("p_idx").is_in(pl.concat([fn["p_idx"], fp["p_idx"], gt["p_idx"]]).unique().implode())).collect()
        for name, df in (("missed (FN)", fn), ("false positive (FP)", fp), ("all true pairs", gt)):
            x = df.select("p_idx").join(pool_attr, on="p_idx")
            print(f"\n== {name}: share by candidate attribute ==\n"
                  f"  native-script name {x['cand_native'].mean():.3f}   empty address {x['cand_addr_empty'].mean():.3f}   "
                  f"from S3 {(x['cand_src'] == 'S3').mean():.3f}", flush=True)

        # examples
        rec_cols = ["idx", "business_name", "business_address"]
        s1r = pl.scan_parquet(WORK / "train_s1.parquet").select(rec_cols).collect().rename(
            {"idx": "q_idx", "business_name": "s1_name", "business_address": "s1_addr"})
        pool = pl.scan_parquet(WORK / "train_pool.parquet").select(rec_cols)
        fn_in = fn.filter(pl.col("rr_rank").is_not_null())
        ex = {"FALSE POSITIVES": fp.sample(min(a.examples, fp.height), seed=0),
              "MISSED MATCHES (in candidates)": fn_in.sample(min(a.examples, fn_in.height), seed=0)}
        need = pl.concat([d["p_idx"] for d in ex.values()]).unique()
        pr_ = pool.filter(pl.col("idx").is_in(need.implode())).collect().rename(
            {"idx": "p_idx", "business_name": "cand_name", "business_address": "cand_addr"})
        for title, df in ex.items():
            x = df.join(s1r, on="q_idx").join(pr_, on="p_idx", how="left")
            print(f"\n== {title} ==", flush=True)
            for r in x.iter_rows(named=True):
                print(f"  p={r.get('prob', float('nan')):.3f} | {r['s1_name'][:45]:45} | {str(r['cand_name'])[:45]:45} | "
                      f"{str(r['s1_addr'])[:50]:50} | {str(r['cand_addr'])[:50]}")


if __name__ == "__main__":
    main()
