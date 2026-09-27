import os
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("BER_DATA", ROOT / "data"))
WORK = Path(os.environ.get("BER_WORK", ROOT / "work"))
OUTPUT = Path(os.environ.get("BER_OUTPUT", ROOT / "output"))

SOURCE_SCHEMA = {
    "entity_id": pl.Utf8,
    "business_name": pl.Utf8,
    "business_address": pl.Utf8,
    "country": pl.Utf8,
}


def _read_tsv(path: Path, schema: dict) -> pl.DataFrame:
    # quote_char=None: names contain stray quotes; the files are plain TSV with no quoting.
    return pl.read_csv(
        path,
        separator="\t",
        schema=schema,
        quote_char=None,
        missing_utf8_is_empty_string=True,
    )


def is_ce_query(s1_id: pl.Expr, share: float = 0.10) -> pl.Expr:
    """Fixed ~10% of train S1 ids reserved for training the cross-encoder."""
    return (s1_id.hash(seed=11) % 1000) < int(share * 1000)


def in_score_share(s1_id: pl.Expr, share: float) -> pl.Expr:
    """Fixed hash-based sample of S1 ids that the cross-encoder scores on train."""
    return (s1_id.hash(seed=5) % 1000) < int(share * 1000)


TAG = os.environ.get("BER_TAG", "")


def tagged(name: str) -> str:
    """'train_oof.parquet' -> 'train_oof_v2.parquet' when BER_TAG=v2 (outputs of one version never
    overwrite another's, which on Kaggle are attached read-only)."""
    if not TAG:
        return name
    stem, dot, ext = name.partition(".")
    return f"{stem}_{TAG}{dot}{ext}"


def load_source(split: str, source: int) -> pl.DataFrame:
    cache = WORK / f"{split}_source{source}.parquet"
    if cache.exists():
        try:
            return pl.read_parquet(cache)
        except Exception:
            cache.unlink(missing_ok=True)
    df = _read_tsv(DATA / split / f"{split}_source{source}.tsv", SOURCE_SCHEMA)
    WORK.mkdir(exist_ok=True)
    tmp = cache.with_name(cache.stem + ".tmp.parquet")
    df.write_parquet(tmp)
    tmp.rename(cache)
    return df


def load_ground_truth() -> pl.DataFrame:
    """Long format: one row per (s1, matched id); singletons kept with match=None."""
    cache = WORK / "train_gt_long.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    gt = _read_tsv(
        DATA / "train" / "train_ground_truth.tsv",
        {"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8},
    )
    long = (
        gt.with_columns(
            pl.col("matched_entity_ids").str.split(",").alias("match")
        )
        .explode("match")
        .with_columns(
            pl.when(pl.col("match") == "").then(None).otherwise(pl.col("match")).alias("match")
        )
        .select(pl.col("source1_entity_id").alias("s1"), "match")
    )
    WORK.mkdir(exist_ok=True)
    long.write_parquet(cache)
    return long


def write_id_lists(df: pl.DataFrame, s1_ids: pl.Series, list_col: str, out_col: str, path: Path) -> None:
    """df has columns s1, <list_col> (one id per row). Writes one row per s1 in s1_ids."""
    grouped = df.group_by("s1").agg(pl.col(list_col).unique(maintain_order=True).str.join(","))
    full = (
        pl.DataFrame({"s1": s1_ids})
        .join(grouped, on="s1", how="left")
        .with_columns(pl.col(list_col).fill_null(""))
        .rename({"s1": "source1_entity_id", list_col: out_col})
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    full.write_csv(path, separator="\t", quote_style="never")
