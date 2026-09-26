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

Disk, at full scale. This is the tighter constraint in practice: Kaggle's /kaggle/working
is capped independently of session RAM (~20GB, seen directly -- a run OOM'd once on RAM,
then later filled /kaggle/working solid and stopped). Each field's finished cache is the
same ~15GB per country as above, so two fields for one country can already exceed the
quota with nothing else on disk. _encode_pool writes its in-progress checkpoint as a
plain appended-bytes file (not a pre-sized memmap) specifically so a field that's 10%
done only occupies 10% of its eventual size, not all of it up front; __init__ also
evicts its own already-consumed sibling caches (already loaded into FAISS this run) if
free space runs low before starting the next field, trading resumability for the field
that's already been used for feasibility of the one that hasn't.

Text. Unlike ngram.py's character-3-gram fields, spaces are kept: a transformer tokenises
into words/subwords, so gluing the name into one token (as the n-gram channel does for
website-style names) would only hurt it here.
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

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


def _model_slug(model_name: str) -> str:
    return model_name.replace("/", "__")


DISK_SAFETY_MARGIN = 2 * 1024**3  # headroom left for parquet/model/OS after a field's cache


class EmbeddingIndex:
    def __init__(self, split: str, country: str, fields=EMB_FIELDS, model_name: str = DEFAULT_MODEL,
                 batch_size: int = 256, device: str | None = None):
        self.split, self.fields, self.model_name, self.batch_size = split, list(fields), model_name, batch_size
        self.model = _get_model(model_name, device)
        self._dim: int | None = None
        pool = (
            pl.scan_parquet(WORK / f"{split}_pool.parquet")
            .filter(pl.col("country") == country)
            .select("idx", *[FIELDS[f].alias(f) for f in self.fields])
            .collect()
        )
        self.p_ids = pool["idx"].to_numpy()
        self.p_row = pl.DataFrame({"p_idx": pool["idx"], "p_row": np.arange(pool.height, dtype=np.int64)})
        self.index = {}
        log.info(f"{country}: {pool.height:,} pool records, fields {self.fields}, model {model_name}")
        ctry_slug = country.replace(" ", "_")

        model_slug = _model_slug(model_name)
        for f in self.fields:
            # The pool encode is the expensive, slow step (minutes to hours per field on a
            # multi-million-record country pool) and everything here otherwise lives only
            # in memory, so a crashed/killed/disconnected session loses all of it. Caching
            # each finished field to disk means a restart resumes instead of re-encoding.
            cache = WORK / f"emb_{model_slug}_{split}_{ctry_slug}_{f}.npy"
            if cache.exists():
                log.info(f"{country} '{f}': loading cached embeddings from {cache.name}")
                vecs = np.load(cache, mmap_mode="r")
                assert vecs.shape[0] == pool.height, (
                    f"{cache}: cached {vecs.shape[0]:,} vectors but pool has {pool.height:,} records "
                    "(stale cache from a different data version?)"
                )
            else:
                self._make_room(pool.height, keep_stem=cache.stem)
                with timed(log, f"{country}: encode '{f}' ({pool.height:,} records)"):
                    vecs = self._encode_pool(pool[f].to_list(), cache)
            import faiss
            if _IS_MACOS:
                faiss.omp_set_num_threads(1)  # env var alone isn't always honoured; see note above
            idx = faiss.IndexFlatIP(vecs.shape[1])
            idx.add(vecs)  # FAISS copies vecs into its own storage; we deliberately don't
            self.index[f] = idx  # keep a second copy (see _reconstruct) -- that copy is what OOM'd
            log.info(f"{country} '{f}': {vecs.shape[0]:,} vectors, dim {vecs.shape[1]}")
            del vecs

    def _get_dim(self) -> int:
        if self._dim is None:
            self._dim = (self.model.get_embedding_dimension if hasattr(self.model, "get_embedding_dimension")
                         else self.model.get_sentence_embedding_dimension)()
        return self._dim

    def _make_room(self, n_rows: int, keep_stem: str) -> None:
        """Evicts already-finished caches AND abandoned in-progress partials for this
        split+model (any country, any field other than `keep_stem`) if there isn't
        enough free disk for the field about to be encoded.

        `keep_stem` is that field's own cache stem and is never touched here: if it has
        a valid partial, _encode_pool resumes it; this function only clears away
        everything else that could be competing for the same disk quota.

        Two kinds of victim, both safe to evict for the same reason -- once a field is
        no longer the one being worked on, either it's fully consumed (finished, in a
        FAISS index already: a completed country's EmbeddingIndex is built once, used
        for both train/eval query sets, then dropped -- see rerank_dev.py/candidates.py
        -- so nothing reads its on-disk .npy again) or it's abandoned (an in-progress
        .partial.raw/.progress.json from an earlier attempt that moved on, restarted, or
        crashed before reaching that field again). Both cases are indistinguishable from
        the filesystem alone and both are equally reclaimable:
          - .npy: a completed field, any country -- e.g. a fresh instance for a larger
            country (India -> US) reclaiming the previous country's leftovers.
          - .partial.raw / .progress.json: an in-progress field NOT currently being
            processed. This is what a real run's disk-quota crash traced back to: a
            stale name_addr.partial.raw survived, untouched, through an entire later
            attempt that started 'name' from scratch (its own .npy having already been
            evicted by an earlier round of this same logic) -- because name_addr was
            still a legitimate field of that run, just not the one being touched *yet*.
            The two are only actually distinguishable at all by `keep_stem`: whatever
            field this specific call is about to encode.
        """
        needed = n_rows * self._get_dim() * 4 + DISK_SAFETY_MARGIN
        free = shutil.disk_usage(WORK).free
        if free >= needed:
            return
        model_slug = _model_slug(self.model_name)
        victims = []
        for suffix in (".npy", ".partial.raw", ".progress.json"):
            for p in WORK.glob(f"emb_{model_slug}_{self.split}_*{suffix}"):
                if p.name.removesuffix(suffix) != keep_stem:
                    victims.append(p)
        victims.sort(key=lambda p: p.stat().st_mtime)
        for victim in victims:
            if free >= needed:
                break
            if not victim.exists():  # a .npy and its stray .progress.json can co-list; already gone
                continue
            size = victim.stat().st_size
            victim.unlink()
            log.info(f"freed {size / 1e9:.2f}GB by evicting {victim.name} to make room for '{keep_stem}'")
            free = shutil.disk_usage(WORK).free
        if free < needed:
            log.warning(f"only {free / 1e9:.1f}GB free, wanted {needed / 1e9:.1f}GB, and nothing left to evict "
                        "-- the encode below may still hit the disk quota")

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=self.batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        ).astype(np.float32)

    def _encode_pool(self, texts: list[str], cache: Path, chunk: int = 200_000) -> np.ndarray:
        """Encodes a whole country pool field in resumable chunks, logging every 10%.

        The in-progress file (".partial.raw") is plain appended bytes, not a pre-sized
        memmap: a field that's 10% done occupies ~10% of its eventual disk footprint, not
        100% of it from the first write. That matters on Kaggle's ~20GB /kaggle/working
        quota, where a field newly starting would otherwise instantly claim its full
        ~15GB before a single record finishes (this is what filled the quota solid on a
        real run: the just-finished field's cache plus the next field's pre-allocated,
        still-empty file together exceeded it). Progress (".progress.json") is written
        after every chunk via atomic tmp+rename, so a crash resumes from the last
        checkpoint. The finished file becomes a normal .npy via a streamed header+copy
        that never materialises the whole array to do it.
        """
        n, dim = len(texts), self._get_dim()
        progress = cache.with_name(cache.stem + ".progress.json")
        raw = cache.with_name(cache.stem + ".partial.raw")

        done = 0
        if progress.exists() and raw.exists():
            state = json.loads(progress.read_text())
            if (state.get("n") == n and state.get("dim") == dim
                    and raw.stat().st_size >= state["done"] * dim * 4):
                done = state["done"]
                log.info(f"{cache.stem}: resuming from checkpoint {done:,}/{n:,} ({100 * done / max(n, 1):.0f}%)")
            else:
                log.info(f"{cache.stem}: stale/short checkpoint -- restarting from 0")
        if done:
            with open(raw, "r+b") as f:
                f.truncate(done * dim * 4)  # drop any partial trailing write from a crash mid-chunk
        else:
            raw.unlink(missing_ok=True)

        next_pct = (int(100 * done / max(n, 1)) // 10 + 1) * 10
        t0 = time.time()
        with open(raw, "ab" if done else "wb") as f:
            for s in range(done, n, chunk):
                e = min(s + chunk, n)
                f.write(self._encode(texts[s:e]).tobytes())
                f.flush()
                os.fsync(f.fileno())
                done = e
                tmp = progress.with_name(progress.stem + ".tmp" + progress.suffix)
                tmp.write_text(json.dumps({"n": n, "dim": dim, "done": done}))
                tmp.rename(progress)
                pct = 100 * done / max(n, 1)
                while next_pct <= pct and next_pct <= 100:
                    rate = done / (time.time() - t0) if time.time() > t0 else 0.0
                    eta_min = (n - done) / rate / 60 if rate > 0 else float("nan")
                    log.info(f"{cache.stem}: {next_pct}% ({done:,}/{n:,})  {rate:.0f} rec/s  ETA {eta_min:.0f}m")
                    next_pct += 10

        tmp_npy = cache.with_name(cache.stem + ".tmp.npy")
        with open(tmp_npy, "wb") as out, open(raw, "rb") as inp:
            np.lib.format.write_array_header_1_0(out, {"descr": "<f4", "fortran_order": False, "shape": (n, dim)})
            shutil.copyfileobj(inp, out, length=64 * 1024 * 1024)
        tmp_npy.rename(cache)
        raw.unlink()
        progress.unlink(missing_ok=True)
        return np.load(cache, mmap_mode="r")

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

    def _reconstruct(self, f: str, rows: np.ndarray) -> np.ndarray:
        """Pulls specific pool vectors back out of the FAISS index (see __init__: we don't
        keep a second copy of the full pool matrix around just for this)."""
        idx = self.index[f]
        if hasattr(idx, "reconstruct_batch"):
            return idx.reconstruct_batch(rows.astype(np.int64))
        return np.stack([idx.reconstruct(int(i)) for i in rows])

    def pair_features(self, pairs: pl.DataFrame) -> pl.DataFrame:
        """Adds emb_<field> cosine score and emb_<field>_rank (within query) for every pair."""
        q = self._queries(pairs["q_idx"].unique())
        q_row = pl.DataFrame({"q_idx": q["idx"], "q_row": np.arange(q.height, dtype=np.int64)})
        x = pairs.join(q_row, on="q_idx").join(self.p_row, on="p_idx")
        qi, pj = x["q_row"].to_numpy(), x["p_row"].to_numpy()
        cols = {}
        for f in self.fields:
            Qv = self._encode(q[f].to_list())
            cols[f"emb_{f}"] = np.einsum("ij,ij->i", Qv[qi], self._reconstruct(f, pj)).astype(np.float32)
        x = x.drop("q_row", "p_row").with_columns([pl.Series(k, v) for k, v in cols.items()])
        return x.with_columns(
            [pl.col(f"emb_{f}").rank("ordinal", descending=True).over("q_idx").cast(pl.UInt16).alias(f"emb_{f}_rank")
             for f in self.fields]
        )
