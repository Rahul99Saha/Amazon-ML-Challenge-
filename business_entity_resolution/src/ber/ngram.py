"""Candidate channel 2 + features: character n-gram retrieval, BM25 by default.

An NgramIndex is fitted once per (split, country) on the pool (the "documents").

weighting="bm25" (default)
  doc weight   w(p, g) = idf(g) * tf*(k1+1) / (tf + k1*(1 - b + b*len(p)/avg_len))
  idf(g)       = ln((N - df + 0.5) / (df + 0.5) + 1)
  score(q, p)  = sum_g tf_q(g) * w(p, g) / BM25(q, q)
  Dividing by the query's self-score makes scores comparable across queries (a
  perfect self-match ~ 1), so they work as a feature and with a minimum threshold.
weighting="tfidf"
  sublinear tf x idf, L2-normalised rows, score = cosine. Kept so the two weightings
  can be compared on the same held-out queries (see PROGRESS.md design notes).

Uses: retrieval (exact top-k via a pruned sparse product, sparse_dot_topn, never the
full query x pool matrix) and features (score for any pair, whichever channel found
it). N-grams in more than `max_df` pool records are dropped: their idf is ~0 so ranking
barely changes, but their posting lists dominate the cost.
"""
import time

import numpy as np
import polars as pl
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from ber.features import NG_FIELDS
from ber.io import WORK
from ber.log import get, timed

log = get("ngram")

FIELDS = {
    # spaces removed: "safe marine" / "safemarine.com" / "#SAFEMARINE" all share 3-grams
    "name": pl.col("name_core").str.replace_all(" ", ""),
    # full normalised address; common words are left to idf + max_df pruning
    "name_addr": pl.concat_str([pl.col("name_core"), pl.col("addr_norm")], separator=" "),
}


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75, n=(3, 3), max_df: int = 50_000):
        self.k1, self.b = k1, b
        self.cv = CountVectorizer(analyzer="char", ngram_range=n, max_df=max_df, lowercase=False, dtype=np.float32)

    def fit_transform(self, docs: list[str]) -> sparse.csr_matrix:
        C = self.cv.fit_transform(docs).tocsr()
        df = np.bincount(C.indices, minlength=C.shape[1])
        self.idf = np.log((C.shape[0] - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
        self.avg_len = float(C.sum(axis=1).mean()) or 1.0
        return self._weights(C)

    def _weights(self, C: sparse.csr_matrix) -> sparse.csr_matrix:
        dl = np.asarray(C.sum(axis=1)).ravel()
        norm = self.k1 * (1 - self.b + self.b * dl / self.avg_len)
        W = C.copy().astype(np.float32)
        row = np.repeat(np.arange(C.shape[0]), np.diff(C.indptr))
        tf = W.data
        W.data = self.idf[W.indices] * tf * (self.k1 + 1) / (tf + norm[row])
        return W

    def queries(self, texts: list[str]) -> sparse.csr_matrix:
        """Query term counts scaled by 1 / BM25(q, q)."""
        Qc = self.cv.transform(texts).tocsr().astype(np.float32)
        self_score = np.asarray(Qc.multiply(self._weights(Qc)).sum(axis=1)).ravel()
        scale = np.where(self_score > 0, 1.0 / np.maximum(self_score, 1e-12), 0.0).astype(np.float32)
        return (sparse.diags(scale) @ Qc).tocsr()


class TfidfCosine:
    def __init__(self, n=(3, 3), max_df: int = 50_000):
        self.v = TfidfVectorizer(analyzer="char", ngram_range=n, sublinear_tf=True, max_df=max_df,
                                 dtype=np.float32, lowercase=False)

    def fit_transform(self, docs: list[str]) -> sparse.csr_matrix:
        return self.v.fit_transform(docs).tocsr()

    def queries(self, texts: list[str]) -> sparse.csr_matrix:
        return self.v.transform(texts).tocsr()


def make_weighting(weighting: str, **kw):
    return BM25(**kw) if weighting == "bm25" else TfidfCosine(**{k: v for k, v in kw.items() if k in ("n", "max_df")})


class NgramIndex:
    def __init__(self, split: str, country: str, fields=NG_FIELDS, weighting: str = "bm25", **kw):
        self.split, self.fields, self.weighting = split, list(fields), weighting
        pool = (
            pl.scan_parquet(WORK / f"{split}_pool.parquet")
            .filter(pl.col("country") == country)
            .select("idx", *[FIELDS[f].alias(f) for f in self.fields])
            .collect()
        )
        self.p_ids = pool["idx"].to_numpy()
        self.p_row = pl.DataFrame({"p_idx": pool["idx"], "p_row": np.arange(pool.height, dtype=np.int64)})
        self.w, self.P, self.PT = {}, {}, {}
        log.info(f"{country}: {pool.height:,} pool records, fields {self.fields}, weighting {weighting}")
        for f in self.fields:
            with timed(log, f"{country}: fit {weighting} on '{f}'"):
                w = make_weighting(weighting, **kw)
                self.P[f] = w.fit_transform(pool[f].to_list())
                self.PT[f] = sparse.csr_matrix(self.P[f].T)
                self.w[f] = w
            log.info(f"{country} '{f}': vocab {self.P[f].shape[1]:,} n-grams, {self.P[f].nnz:,} non-zeros")

    def _queries(self, q_ids: pl.Series) -> pl.DataFrame:
        return (
            pl.scan_parquet(WORK / f"{self.split}_s1.parquet")
            .filter(pl.col("idx").is_in(q_ids.implode()))
            .select("idx", *[FIELDS[f].alias(f) for f in self.fields])
            .collect()
        )

    def topk(self, q_ids: pl.Series, top_k: int = 50, min_score: float = 0.2, threads: int = 4) -> pl.DataFrame:
        """(q_idx, p_idx) for the union over fields of each query's top-k neighbours."""
        q = self._queries(q_ids)
        outs = []
        for f in self.fields:
            t = time.time()
            Q = self.w[f].queries(q[f].to_list())
            R = sp_matmul_topn(Q, self.PT[f], top_n=top_k, threshold=min_score, n_threads=threads).tocoo()
            log.info(f"top-{top_k} '{f}': {q.height:,} queries -> {R.nnz:,} pairs ({time.time() - t:.0f}s)")
            outs.append(pl.DataFrame({"q_idx": q["idx"].to_numpy()[R.row], "p_idx": self.p_ids[R.col]}))
        return pl.concat(outs).unique()

    def pair_features(self, pairs: pl.DataFrame) -> pl.DataFrame:
        """Adds ng_<field> score and ng_<field>_rank (within query) for every pair."""
        q = self._queries(pairs["q_idx"].unique())
        q_row = pl.DataFrame({"q_idx": q["idx"], "q_row": np.arange(q.height, dtype=np.int64)})
        x = pairs.join(q_row, on="q_idx").join(self.p_row, on="p_idx")
        qi, pj = x["q_row"].to_numpy(), x["p_row"].to_numpy()
        cols = {}
        for f in self.fields:
            Q = self.w[f].queries(q[f].to_list())
            cols[f"ng_{f}"] = np.asarray(Q[qi].multiply(self.P[f][pj]).sum(axis=1)).ravel().astype(np.float32)
        x = x.drop("q_row", "p_row").with_columns([pl.Series(k, v) for k, v in cols.items()])
        return x.with_columns(
            [pl.col(f"ng_{f}").rank("ordinal", descending=True).over("q_idx").cast(pl.UInt16).alias(f"ng_{f}_rank")
             for f in self.fields]
        )
