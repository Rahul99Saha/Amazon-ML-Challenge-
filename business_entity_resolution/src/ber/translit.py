"""Learn a transliterated-token -> Latin-token dictionary from training matches.

Native-script names (Devanagari, Tamil, Kannada, ...) are romanised by anyascii into
phonetic spellings ("praivet limitet") that share no tokens with the Latin record
("private limited"). Matched train pairs where exactly one side was non-ASCII and both
sides have the same token count are aligned position by position; frequent,
consistent alignments become the dictionary. Learned only from provided train data.
"""
import polars as pl

from ber.io import WORK

PATH = WORK / "translit_map.parquet"


def learn(min_count: int = 3, min_share: float = 0.5) -> pl.DataFrame:
    cols = ["idx", "business_name", "name_tokens"]
    pairs = pl.read_parquet(WORK / "train_gt_pairs.parquet")
    s1 = pl.read_parquet(WORK / "train_s1.parquet", columns=cols)
    pool = pl.read_parquet(WORK / "train_pool.parquet", columns=cols)
    nonascii = pl.col("business_name").str.contains(r"[^\x00-\x7F]")
    latin = pool.filter(~nonascii).select("idx", lat=pl.col("name_tokens"))
    native = pool.filter(nonascii).select("idx", nat=pl.col("name_tokens"))
    s1l = s1.filter(~nonascii).select(q_idx="idx", lat=pl.col("name_tokens"))

    # Native pool record aligned against its S1 record and against Latin pool siblings.
    nat_p = pairs.join(native.rename({"idx": "p_idx"}), on="p_idx")
    a = nat_p.join(s1l, on="q_idx").select("nat", "lat")
    sib = pairs.join(latin.rename({"idx": "p_idx"}), on="p_idx").select("q_idx", "lat")
    b = nat_p.select("q_idx", "nat").join(sib, on="q_idx").select("nat", "lat")
    al = (
        pl.concat([a, b])
        .filter(pl.col("nat").list.len() == pl.col("lat").list.len())
        .with_row_index("r")
        .explode(["nat", "lat"])
        .filter(pl.col("nat") != pl.col("lat"))
    )
    counts = al.group_by("nat", "lat").len("n")
    tot = al.group_by("nat").len("tot")
    best = (
        counts.join(tot, on="nat")
        .sort("n", descending=True)
        .group_by("nat", maintain_order=True)
        .first()
        .filter((pl.col("n") >= min_count) & (pl.col("n") / pl.col("tot") >= min_share))
        .select("nat", "lat", "n", "tot")
    )
    best.write_parquet(PATH)
    return best


def load() -> dict:
    if not PATH.exists():
        return {}
    m = pl.read_parquet(PATH)
    return dict(zip(m["nat"].to_list(), m["lat"].to_list()))


if __name__ == "__main__":
    m = learn()
    print(m.height, "mappings")
    with pl.Config(tbl_rows=40):
        print(m.sort("n", descending=True).head(40))
