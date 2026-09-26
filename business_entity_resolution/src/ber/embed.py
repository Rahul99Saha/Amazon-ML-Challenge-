"""Candidate channel 3 + features: dense multilingual sentence-embedding retrieval.

Why a third channel. The rare-key index (blocking.py) and the n-gram BM25 channel
(ngram.py) are both tuned entirely from *train* pairs: the transliteration dictionary,
IDF weights and key lists all come from data seen during training. They have no way to
generalise to test-only France or to native-script names the transliteration map never
saw (PROGRESS.md, section 3a: "About 1.5% of true matches are still never generated at
all ... native-script names not covered by the transliteration map"). A pretrained
multilingual sentence embedding model brings that generalisation from its own
pretraining instead of from our training pairs, so it is expected to help most on
exactly those residual misses.

Model. Default is sentence-transformers/LaBSE (Apache-2.0, ~471M params, well under the
8B/permissive-license constraint): trained for cross-lingual sentence-level semantic
similarity across 100+ languages, which is the right objective for matching a Devanagari
/ Tamil / Kannada / Gujarati / Odia name against its Latin-script counterpart. Any other
sentence-transformers model can be swapped in via `model_name` (e.g. the smaller, MIT
licensed intfloat/multilingual-e5-base) to trade recall for memory/speed -- run
rerank_dev.py --embed with both to compare, the same way ngram.py's BM25-vs-TF-IDF
choice was settled.

Retrieval. An EmbeddingIndex is fitted once per (split, country): pool text is encoded
and L2-normalised, then held in a FAISS IndexFlatIP (exact inner product == cosine on
unit vectors) -- FAISS plays the same role here that sparse_dot_topn plays for the BM25
channel: an optimised library doing exact top-k, not a hand-rolled loop and not an
approximate index. Encoding happens per-country and per-field, one field at a time, so
only one field's vectors are resident at once.

Memory, at full scale. A pool of P records encoded at dimension d in float32 costs
4*P*d bytes per field. LaBSE is d=768: sharded per country (~5M records for the larger
train pools), that is ~15GB for one field -- fits Kaggle's ~30GB CPU RAM one field at a
time (as done here) but leaves little headroom. If that is a problem in practice, the
two levers are (a) a smaller-dimension model (e.g. paraphrase-multilingual-MiniLM-L12-v2,
d=384, half the memory) or (b) a FAISS scalar-quantized index (IndexScalarQuantizer,
int8) in place of IndexFlatIP -- not implemented here since it needs at-scale testing to
validate the recall/memory trade-off.

Text. Unlike ngram.py's character-3-gram fields, spaces are kept: a transformer tokenises
into words/subwords, so gluing the name into one token (as the n-gram channel does for
website-style names) would only hurt it here.
"""
import os
import sys

# faiss (libomp) and torch (libiomp5/MKL) each bundle their own OpenMP runtime. Loading
# both in one process aborts on macOS ("OMP: Error #179") or segfaults intermittently
# once both libraries' thread pools are active at once (e.g. an encode() call racing a
# faiss search()). Pinning both to one thread avoids it; Linux/Kaggle doesn't hit this,
# so it's left untouched there rather than paying for single-threaded faiss on the full
# 10M+ record runs.
_IS_MACOS = sys.platform == "darwin"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if _IS_MACOS:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import polars as pl

from ber.features import EMB_FIELDS
from ber.io import WORK
from ber.log import get, timed

log = get("embed")

DEFAULT_MODEL = "sentence-transformers/LaBSE"

FIELDS = {
    "name": pl.col("name_core"),
    "name_addr": pl.concat_str([pl.col("name_core"), pl.col("addr_norm")], separator=" "),
}

_MODEL_CACHE: dict[str, object] = {}


def _get_model(model_name: str, device: str | None):
    key = f"{model_name}@{device}"
    if key not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        log.info(f"loading model {model_name} (device={device or 'auto'})")
        _MODEL_CACHE[key] = SentenceTransformer(model_name, device=device)
    return _MODEL_CACHE[key]


class EmbeddingIndex:
    def __init__(self, split: str, country: str, fields=EMB_FIELDS, model_name: str = DEFAULT_MODEL,
                 batch_size: int = 256, device: str | None = None):
        self.split, self.fields, self.model_name, self.batch_size = split, list(fields), model_name, batch_size
        self.model = _get_model(model_name, device)
        pool = (
            pl.scan_parquet(WORK / f"{split}_pool.parquet")
            .filter(pl.col("country") == country)
            .select("idx", *[FIELDS[f].alias(f) for f in self.fields])
            .collect()
        )
        self.p_ids = pool["idx"].to_numpy()
        self.p_row = pl.DataFrame({"p_idx": pool["idx"], "p_row": np.arange(pool.height, dtype=np.int64)})
        self.index, self.P = {}, {}
        log.info(f"{country}: {pool.height:,} pool records, fields {self.fields}, model {model_name}")
        for f in self.fields:
            with timed(log, f"{country}: encode '{f}' ({pool.height:,} records)"):
                vecs = self._encode(pool[f].to_list())
            self.P[f] = vecs
            import faiss
            if _IS_MACOS:
                faiss.omp_set_num_threads(1)  # env var alone isn't always honoured; see note above
            idx = faiss.IndexFlatIP(vecs.shape[1])
            idx.add(vecs)
            self.index[f] = idx
            log.info(f"{country} '{f}': {vecs.shape[0]:,} vectors, dim {vecs.shape[1]}")

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=self.batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        ).astype(np.float32)

    def _queries(self, q_ids: pl.Series) -> pl.DataFrame:
        return (
            pl.scan_parquet(WORK / f"{self.split}_s1.parquet")
            .filter(pl.col("idx").is_in(q_ids.implode()))
            .select("idx", *[FIELDS[f].alias(f) for f in self.fields])
            .collect()
        )

    def topk(self, q_ids: pl.Series, top_k: int = 50, min_score: float = 0.5) -> pl.DataFrame:
        """(q_idx, p_idx) for the union over fields of each query's top-k neighbours by cosine sim."""
        q = self._queries(q_ids)
        outs = []
        for f in self.fields:
            Qv = self._encode(q[f].to_list())
            sims, cols = self.index[f].search(Qv, top_k)
            rows = np.repeat(np.arange(len(q)), top_k)
            cols, sims = cols.ravel(), sims.ravel()
            keep = (cols >= 0) & (sims >= min_score)
            log.info(f"top-{top_k} '{f}': {q.height:,} queries -> {keep.sum():,} pairs")
            outs.append(pl.DataFrame({"q_idx": q["idx"].to_numpy()[rows[keep]], "p_idx": self.p_ids[cols[keep]]}))
        return pl.concat(outs).unique()

    def pair_features(self, pairs: pl.DataFrame) -> pl.DataFrame:
        """Adds emb_<field> cosine score and emb_<field>_rank (within query) for every pair."""
        q = self._queries(pairs["q_idx"].unique())
        q_row = pl.DataFrame({"q_idx": q["idx"], "q_row": np.arange(q.height, dtype=np.int64)})
        x = pairs.join(q_row, on="q_idx").join(self.p_row, on="p_idx")
        qi, pj = x["q_row"].to_numpy(), x["p_row"].to_numpy()
        cols = {}
        for f in self.fields:
            Qv = self._encode(q[f].to_list())
            cols[f"emb_{f}"] = np.einsum("ij,ij->i", Qv[qi], self.P[f][pj]).astype(np.float32)
        x = x.drop("q_row", "p_row").with_columns([pl.Series(k, v) for k, v in cols.items()])
        return x.with_columns(
            [pl.col(f"emb_{f}").rank("ordinal", descending=True).over("q_idx").cast(pl.UInt16).alias(f"emb_{f}_rank")
             for f in self.fields]
        )
