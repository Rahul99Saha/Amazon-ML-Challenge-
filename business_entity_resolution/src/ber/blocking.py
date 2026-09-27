"""Stage-1 candidate generation: IDF-weighted multi-key inverted index.

Every record emits hashed keys (name tokens, compact-name prefix, house-number x
rare-address-word, rare address word). An S1 query and a pool record become a
candidate pair when they share keys; pairs are scored by the summed IDF of shared
keys and the top-K per query are kept. Keys whose pool block exceeds `max_block`
are dropped, which bounds the work to O(queries x keys x max_block).

Keys are built in row slices and cached as (idx u32, key u64, kind u8) parquet so
the 10M-record pool never needs to be materialised with its string columns.
"""
from pathlib import Path

import time

import polars as pl

from ber.io import WORK
from ber.log import Progress, get, timed

log = get("blocking")

KINDS = ["name", "compact", "addr", "addrword", "namepair", "addrpair", "nameaddr", "compact10"]
KEY_COLS = ["idx", "country", "name_core_tokens", "name_alt", "name_compact", "addr_words", "addr_nums"]

GENERIC_ADDR = [
    "street", "road", "avenue", "boulevard", "drive", "court", "place", "lane",
    "highway", "parkway", "circle", "trail", "terrace", "square", "unit", "floor",
    "ground", "building", "number", "near", "opposite", "block", "sector", "phase",
    "north", "south", "east", "west", "house", "post", "district", "nagar", "colony",
    "rue", "chemin", "route", "impasse", "allee", "quai", "cours", "de",
    "du", "des", "la", "le", "les", "of", "the", "and", "flat", "plot", "shop",
    "office", "room", "main", "cross", "box", "apartments", "industrial", "area",
]


def _addr_words(df: pl.LazyFrame | pl.DataFrame):
    return (
        df.select("idx", "country", w=pl.col("addr_words"))
        .explode("w")
        .drop_nulls("w")
        .filter(~pl.col("w").is_in(GENERIC_ADDR) & (pl.col("w").str.len_chars() >= 3))
    )


def word_df(pool_path: Path) -> pl.DataFrame:
    return _addr_words(pl.scan_parquet(pool_path)).group_by("country", "w").len("df").collect()


def _hash(parts: list) -> pl.Expr:
    return pl.concat_str(parts).hash(seed=17)


def record_keys(df: pl.DataFrame, wdf: pl.DataFrame) -> pl.DataFrame:
    ctry = pl.col("country").fill_null("")
    name_tok = (
        df.select(
            "idx", "country",
            t=pl.col("name_core_tokens")
            .list.concat(pl.col("name_alt").str.split(" "))
            .list.eval(pl.element().filter(pl.element().str.len_chars() >= 2))
            .list.unique(),
        )
        .explode("t")
        .drop_nulls("t")
        .select("idx", key=_hash([pl.lit("n|"), ctry, pl.lit("|"), pl.col("t")]), kind=pl.lit(0, pl.UInt8))
    )
    compact = df.filter(pl.col("name_compact").str.len_chars() >= 4).select(
        "idx",
        key=_hash([pl.lit("c|"), ctry, pl.lit("|"), pl.col("name_compact").str.slice(0, 6)]),
        kind=pl.lit(1, pl.UInt8),
    )
    compact10 = df.filter(pl.col("name_compact").str.len_chars() >= 10).select(
        "idx",
        key=_hash([pl.lit("c10|"), ctry, pl.lit("|"), pl.col("name_compact").str.slice(0, 10)]),
        kind=pl.lit(7, pl.UInt8),
    )
    words = (
        _addr_words(df)
        .join(wdf, on=["country", "w"], how="left")
        .with_columns(pl.col("df").fill_null(0))
        .sort(["idx", "df"])
        .group_by("idx", maintain_order=True)
        .head(3)
        .select("idx", "country", "w")
    )
    nums = df.select("idx", n=pl.col("addr_nums").list.head(3)).explode("n").drop_nulls("n")
    
    addr = words.join(nums, on="idx").select(
        "idx",
        key=_hash([pl.lit("a|"), ctry, pl.lit("|"), pl.col("n"), pl.lit("|"), pl.col("w")]),
        kind=pl.lit(2, pl.UInt8),
    )
    addr_w = words.select(
        "idx", key=_hash([pl.lit("w|"), ctry, pl.lit("|"), pl.col("w")]), kind=pl.lit(3, pl.UInt8)
    )
    core = df.select(
        "idx", "country",
        t=pl.col("name_core_tokens")
        .list.eval(pl.element().filter(pl.element().str.len_chars() >= 2))
        .list.unique().list.sort().list.head(6),
    )
    name_pair = _pairs(core.rename({"t": "a"}), core.rename({"t": "b"}), same=True).select(
        "idx", key=_hash([pl.lit("np|"), ctry, pl.lit("|"), pl.col("a"), pl.lit("|"), pl.col("b")]),
        kind=pl.lit(4, pl.UInt8),
    )
    w2 = words.group_by("idx", maintain_order=True).head(2)
    wl = w2.group_by("idx").agg(pl.col("country").first(), pl.col("w").sort())
    addr_pair = _pairs(wl.rename({"w": "a"}), wl.rename({"w": "b"}), same=True).select(
        "idx", key=_hash([pl.lit("ap|"), ctry, pl.lit("|"), pl.col("a"), pl.lit("|"), pl.col("b")]),
        kind=pl.lit(5, pl.UInt8),
    )
    name_addr = (
        core.explode("t").drop_nulls("t")
        .join(w2.select("idx", "w"), on="idx")
        .select(
            "idx", key=_hash([pl.lit("na|"), ctry, pl.lit("|"), pl.col("t"), pl.lit("|"), pl.col("w")]),
            kind=pl.lit(6, pl.UInt8),
        )
    )
    return pl.concat(
        [name_tok, compact, addr, addr_w, name_pair, addr_pair, name_addr, compact10]
    ).unique(["idx", "key"])


def _pairs(left: pl.DataFrame, right: pl.DataFrame, same: bool) -> pl.DataFrame:
    """All unordered pairs a < b of list elements per idx."""
    a = left.explode("a").drop_nulls("a")
    b = right.select("idx", "b").explode("b").drop_nulls("b")
    return a.join(b, on="idx").filter(pl.col("a") < pl.col("b"))


def build_keys(src: Path, dst: Path, wdf: pl.DataFrame, rows: int = 1_000_000) -> str:
    """Returns a path pl.scan_parquet can read for the keys table: either `dst` itself
    (a legacy, already-merged file from before this function stopped merging) or a glob
    over the per-slice parts.

    The parts are deliberately never concatenated into `dst`. Merging would need the
    parts (~2-3GB for the full train pool) AND the growing merged file to coexist on
    disk simultaneously until the merge finishes -- up to 2x the size, for no benefit:
    pl.scan_parquet reads a glob of files exactly like one file. This is what silently
    killed a real run immediately after it had *just* barely fit India's embeddings:
    generate_candidates' first call builds keys for the *entire* pool (both countries,
    ~10.3M records for train), not just the queries' own country, so this cost lands
    regardless of which country's candidates are being generated. Building part-by-part
    also means a restart only redoes whichever parts are missing, not everything.
    """
    if dst.exists():
        return str(dst)
    glob = str(dst.parent / f"{dst.stem}.part*.parquet")
    n = pl.scan_parquet(src).select(pl.len()).collect().item()
    offsets = list(range(0, n, rows))
    parts = [dst.with_suffix(f".part{i}.parquet") for i in range(len(offsets))]
    if parts and all(p.exists() for p in parts):
        return glob
    for i, off in enumerate(offsets):
        if parts[i].exists():
            continue
        df = pl.scan_parquet(src).select(KEY_COLS).slice(off, rows).collect()
        record_keys(df, wdf).write_parquet(parts[i])
    return glob


def generate_candidates(
    split: str,
    q_idx: pl.Series | None = None,
    top_k: int = 50,
    max_block: int = 300,
    chunk: int = 50_000,
    kind_weight: dict | None = None,
) -> pl.DataFrame:
    """Candidates for S1 queries of `split` (optionally a subset q_idx) against its pool.

    Returns (q_idx, p_idx, block_score, block_rank, bk_<kind> counts).
    """
    kind_weight = kind_weight or {
        "name": 1.0, "compact": 1.0, "addr": 1.5, "addrword": 0.5,
        "namepair": 1.0, "addrpair": 0.7, "nameaddr": 0.7, "compact10": 1.5,
    }
    pool_path = WORK / f"{split}_pool.parquet"
    wdf_path = WORK / f"{split}_addr_wdf.parquet"
    if not wdf_path.exists():
        word_df(pool_path).write_parquet(wdf_path)
    wdf = pl.read_parquet(wdf_path)
    with timed(log, "pool/S1 keys (built once, cached)"):
        pk_path = build_keys(pool_path, WORK / f"{split}_pool_keys.parquet", wdf)
        qk_path = build_keys(WORK / f"{split}_s1.parquet", WORK / f"{split}_s1_keys.parquet", wdf)

    n_pool = pl.scan_parquet(pool_path).select(pl.len()).collect().item()
    kw = pl.DataFrame(
        {"kind": [KINDS.index(k) for k in kind_weight], "kw": list(kind_weight.values())},
        schema={"kind": pl.UInt8, "kw": pl.Float32},
    )
    qk = pl.scan_parquet(qk_path)
    if q_idx is not None:
        qk = qk.filter(pl.col("idx").is_in(q_idx.implode()))
    qk = qk.collect(engine="streaming")
    log.info(f"query keys: {qk.height:,}")
    # Pass 1 (streaming, never materialises pool rows): block size of each query key.
    # Pass 2: load pool rows only for keys small enough to keep.
    t_pass = time.time()
    bsize = (
        pl.scan_parquet(pk_path)
        .join(qk.select("key").unique().lazy(), on="key", how="semi")
        .group_by("key").len("df")
        .filter(pl.col("df") <= max_block)
        .with_columns(idf=(n_pool / pl.col("df")).log().cast(pl.Float32))
        .select("key", "idf")
        .collect(engine="streaming")
    )
    log.info(f"pass 1: {bsize.height:,} keys with block size <= {max_block} ({time.time() - t_pass:.0f}s)")
    t_pass = time.time()
    pk = (
        pl.scan_parquet(pk_path)
        .join(bsize.lazy(), on="key")
        .select("key", "idx", "idf")
        .collect(engine="streaming")
    )
    log.info(f"pass 2: {pk.height:,} pool key rows loaded ({time.time() - t_pass:.0f}s)")
    qk = qk.join(bsize.select("key"), on="key")
    del bsize

    q_ids = qk["idx"].unique().sort()
    prog = Progress(log, "key-index scoring (queries)", len(q_ids))
    outs = []
    for start in range(0, len(q_ids), chunk):
        ids = q_ids.slice(start, chunk)
        pairs = (
            qk.filter(pl.col("idx").is_in(ids.implode()))
            .join(pk, on="key", suffix="_p")
            .join(kw, on="kind")
            .group_by("idx", "idx_p")
            .agg(
                block_score=(pl.col("idf") * pl.col("kw")).sum(),
                **{f"bk_{k}": (pl.col("kind") == KINDS.index(k)).sum().cast(pl.UInt8) for k in kind_weight},
            )
            .sort(["idx", "block_score"], descending=[False, True])
            .group_by("idx", maintain_order=True)
            .head(top_k)
            .with_columns(block_rank=pl.int_range(pl.len()).over("idx").cast(pl.UInt16))
        )
        outs.append(pairs)
        prog.step(len(ids))
    return pl.concat(outs).rename({"idx": "q_idx", "idx_p": "p_idx"})
