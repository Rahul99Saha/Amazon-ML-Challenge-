"""Vectorised (polars) normalisation of business names and addresses.

Language-agnostic by design: transliterate any script to ASCII, then apply
generic cleanup. Hand-written synonym tables only encode public abbreviation
conventions (no external lookup).
"""
import polars as pl
from anyascii import anyascii

LEGAL = {
    "inc", "incorporated", "llc", "ltd", "limited", "pvt", "private", "corp",
    "corporation", "co", "company", "llp", "lp", "pc", "plc", "pllc", "pa",
    "sarl", "sas", "sa", "eurl", "sasu", "sci", "snc", "selarl", "gmbh", "ag",
    "opc",
}
NAME_PREFIX = {"the", "shri", "sri", "shree", "m/s", "ms", "le", "la", "les", "l"}

NAME_SYN = {
    "pvt": "private", "prv": "private", "ltd": "limited", "ltda": "limited",
    "corp": "corporation", "co": "company", "cos": "company", "inc": "incorporated",
    "intl": "international", "int'l": "international", "mfg": "manufacturing",
    "mgmt": "management", "svcs": "services", "svc": "services", "srvs": "services",
    "tech": "technologies", "techs": "technologies", "assoc": "associates",
    "bros": "brothers", "ent": "enterprises", "ents": "enterprises", "grp": "group",
    "hosp": "hospital", "ctr": "center", "centre": "center", "st": "saint",
    "ste": "sainte", "cie": "compagnie", "ets": "etablissements", "et": "and",
}

ADDR_SYN = {
    # English street types
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "avn": "avenue", "blvd": "boulevard", "bd": "boulevard", "bvd": "boulevard",
    "dr": "drive", "drv": "drive", "ct": "court", "crt": "court", "pl": "place",
    "ln": "lane", "hwy": "highway", "pkwy": "parkway", "pky": "parkway", "cir": "circle",
    "trl": "trail", "ter": "terrace", "terr": "terrace", "sq": "square", "pt": "point",
    "cres": "crescent", "xing": "crossing", "expy": "expressway", "fwy": "freeway",
    "mt": "mount", "ft": "fort", "hts": "heights", "jct": "junction",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    # units / buildings
    "apt": "unit", "apartment": "unit", "ste": "unit", "suite": "unit", "rm": "room",
    "fl": "floor", "flr": "floor", "grd": "ground", "gr": "ground", "bldg": "building",
    "bldng": "building", "apts": "apartments", "appt": "unit", "no": "number",
    "num": "number", "nr": "near", "opp": "opposite", "blk": "block", "sec": "sector",
    "sect": "sector", "ph": "phase", "ind": "industrial", "indl": "industrial",
    "estt": "estate", "mkt": "market", "nagr": "nagar", "clny": "colony",
    "po": "post", "dist": "district", "distt": "district", "tal": "taluka",
    "tq": "taluka", "vill": "village", "vil": "village", "hno": "house", "h": "house",
    # French
    "r": "rue", "ch": "chemin", "che": "chemin", "rte": "route", "imp": "impasse",
    "all": "allee", "fbg": "faubourg", "qu": "quai", "crs": "cours",
    "pas": "passage", "res": "residence", "zi": "zone", "za": "zone",
    "zac": "zone", "bat": "batiment",
    # ordinal words
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th",
    "sixth": "6th", "seventh": "7th", "eighth": "8th", "ninth": "9th", "tenth": "10th",
    "eleventh": "11th", "twelfth": "12th", "thirteenth": "13th", "fourteenth": "14th",
    "fifteenth": "15th", "sixteenth": "16th", "seventeenth": "17th",
    "eighteenth": "18th", "nineteenth": "19th", "twentieth": "20th",
    "3nd": "3rd", "2th": "2nd", "1th": "1st", "3th": "3rd",
    # city aliases
    "bombay": "mumbai", "bengaluru": "bangalore", "bengalooru": "bangalore",
    "calcutta": "kolkata", "madras": "chennai", "gurgaon": "gurugram",
    "poona": "pune", "baroda": "vadodara", "trivandrum": "thiruvananthapuram",
    "cochin": "kochi", "mysuru": "mysore", "orissa": "odisha", "pondicherry": "puducherry",
    "allahabad": "prayagraj", "benares": "varanasi", "banaras": "varanasi",
    "vizag": "visakhapatnam", "cawnpore": "kanpur", "simla": "shimla",
    "belgaum": "belagavi", "mangalore": "mangaluru",
}

# Multi-word region names -> short codes (applied on space-padded strings).
REGION_PHRASES = {
    # US
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "district of columbia": "dc",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id", "illinois": "il",
    "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi",
    "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc", "north dakota": "nd",
    "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn",
    "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy", "puerto rico": "pr",
    # India
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "chattisgarh": "cg", "goa": "ga", "gujarat": "gj",
    "haryana": "hr", "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka",
    "kerala": "kl", "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn",
    "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "tg",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk",
    "west bengal": "wb", "delhi": "dl", "jammu and kashmir": "jk", "jammu & kashmir": "jk",
    "ladakh": "la", "chandigarh": "ch", "puducherry": "py",
}

_DBA_RE = (
    r"\b(?:doing business as|d\s?/?\s?b\s?/?\s?a|formerly known as|formerly|"
    r"f\s?/\s?k\s?/\s?a|fka|also known as|a\s?/\s?k\s?/\s?a|trading as|t\s?/\s?a)\b"
)


def _to_ascii(col: str) -> pl.Expr:
    c = pl.col(col).fill_null("")
    return (
        pl.when(c.str.contains(r"[^\x00-\x7F]"))
        .then(c.map_elements(anyascii, return_dtype=pl.Utf8))
        .otherwise(c)
        .str.to_lowercase()
    )


def _tokens(expr: pl.Expr) -> pl.Expr:
    return expr.str.strip_chars().str.split(" ").list.eval(
        pl.element().filter(pl.element() != "")
    )


def _map_tokens(tok: pl.Expr, mapping: dict) -> pl.Expr:
    return tok.list.eval(pl.element().replace(mapping))


def _dedupe_adjacent(tok: pl.Expr) -> pl.Expr:
    return tok.list.eval(
        pl.element().filter(
            (pl.element() != pl.element().shift(1)) | pl.element().shift(1).is_null()
        )
    )


def normalize_names(df: pl.DataFrame, translit: dict | None = None) -> pl.DataFrame:
    base = (
        _to_ascii("business_name")
        .str.replace_all(r"\bwww\.", " ")
        .str.replace_all(r"\.(com|net|org|in|co\.in|fr|biz|info|io|us)\b", " ")
        .str.replace_all(r"[&+]", " and ")
        .str.replace_all(r"\b([a-z])\.", "$1")  # l.l.c. -> llc, p.c. -> pc
        .str.replace_all(r"'", "")
        .str.replace_all(r"\d{7,}", " ")  # phone / registration numbers
    )
    df = df.with_columns(name_ascii=base)
    df = df.with_columns(
        pl.col("name_ascii").str.replace(_DBA_RE, "\u0000").alias("_dba")
    ).with_columns(
        pl.col("_dba").str.split_exact("\u0000", 1).struct.rename_fields(["_n1", "_n2"]).alias("_parts")
    ).unnest("_parts").with_columns(
        # name after the DBA marker is the operating name; keep the other as alt.
        pl.when(pl.col("_n2").is_not_null() & (pl.col("_n2").str.strip_chars() != ""))
        .then(pl.col("_n2")).otherwise(pl.col("_n1")).alias("_main"),
        pl.when(pl.col("_n2").is_not_null() & (pl.col("_n2").str.strip_chars() != ""))
        .then(pl.col("_n1")).otherwise(pl.lit("")).alias("name_alt"),
        pl.col("_dba").str.contains("\u0000").alias("has_dba"),
    )

    def clean(e: pl.Expr) -> pl.Expr:
        return e.str.replace_all(r"[^a-z0-9 ]", " ").str.replace_all(r"\s+", " ")

    tok = _tokens(clean(pl.col("_main")))
    # OCR-style digit->letter fixes inside alphabetic tokens (keeps 2nd/3rd, pure numbers).
    tok = tok.list.eval(
        pl.when(
            pl.element().str.contains(r"[a-z]")
            & pl.element().str.contains(r"[0-9]")
            & ~pl.element().str.contains(r"^\d+(st|nd|rd|th)$")
        )
        .then(
            pl.element()
            .str.replace_all("0", "o").str.replace_all("1", "l").str.replace_all("8", "b")
            .str.replace_all("5", "s").str.replace_all("3", "e").str.replace_all("4", "a")
        )
        .otherwise(pl.element())
    )
    if translit:
        tok = (
            pl.when(pl.col("business_name").fill_null("").str.contains(r"[^\x00-\x7F]"))
            .then(_map_tokens(tok, translit))
            .otherwise(tok)
        )
    tok = _dedupe_adjacent(_map_tokens(tok, NAME_SYN))
    legal_full = {NAME_SYN.get(t, t) for t in LEGAL}
    df = df.with_columns(name_tokens=tok).with_columns(
        name_core_tokens=pl.col("name_tokens").list.eval(
            pl.element().filter(
                ~pl.element().is_in(list(legal_full)) & ~pl.element().is_in(list(NAME_PREFIX))
            )
        ),
        name_legal=pl.col("name_tokens").list.eval(
            pl.element().filter(pl.element().is_in(list(legal_full)))
        ).list.sort().list.join(" "),
    ).with_columns(
        name_norm=pl.col("name_tokens").list.join(" "),
        name_core=pl.col("name_core_tokens").list.join(" "),
        name_compact=pl.col("name_core_tokens")
        .list.eval(pl.element().filter(pl.element() != "and"))
        .list.join(""),
        name_alt=clean(pl.col("name_alt")).str.strip_chars(),
    )
    return df.drop(["_dba", "_n1", "_n2", "_main", "name_ascii"])


def normalize_addresses(df: pl.DataFrame) -> pl.DataFrame:
    a = (
        _to_ascii("business_address")
        .str.replace_all(r"\b(?:n/a|null|none|nan|unknown)\b", " ")
        .str.replace_all(r"[^a-z0-9 ]", " ")
        .str.replace_all(r"\s+", " ")
    )
    # Double spaces so every token owns its padding and adjacent matches both hit.
    a = pl.concat_str([pl.lit(" "), a.str.replace_all(" ", "  "), pl.lit(" ")])
    phrases = sorted(REGION_PHRASES, key=len, reverse=True)
    a = a.str.replace_many(
        [" " + p.replace(" ", "  ") + " " for p in phrases],
        [f" {REGION_PHRASES[p]} " for p in phrases],
    )
    tok = _map_tokens(_tokens(a), ADDR_SYN)
    # "414 1 2" (from 414 1/2) and "B-##1" style noise leave pure-number tokens; keep them.
    df = df.with_columns(addr_tokens=tok).with_columns(
        addr_norm=pl.col("addr_tokens").list.join(" "),
        addr_nums=pl.col("addr_tokens").list.eval(
            pl.element().str.extract(r"(\d+)", 1).drop_nulls().str.strip_chars_start("0")
        ).list.eval(pl.element().filter(pl.element() != "")).list.unique().list.sort(),
        addr_words=pl.col("addr_tokens").list.eval(
            pl.element().filter(~pl.element().str.contains(r"\d"))
        ).list.unique().list.sort(),
        postal=pl.col("addr_tokens").list.eval(
            pl.element().filter(pl.element().str.contains(r"^\d{5,6}$"))
        ).list.first(),
        addr_len=pl.col("addr_tokens").list.len(),
    )
    return df


def normalize(df: pl.DataFrame, translit: dict | None = None) -> pl.DataFrame:
    return normalize_addresses(normalize_names(df, translit)).with_columns(
        name_native=pl.col("business_name").fill_null("").str.contains(r"[^\x00-\x7F]"),
    )
