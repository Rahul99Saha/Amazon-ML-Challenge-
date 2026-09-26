"""Cross-encoder pair scorer (GPU). Fine-tunes microsoft/mdeberta-v3-base (MIT) on
(S1 record, candidate record) pairs built from our own blocking candidates, so the
negatives are exactly the hard cases the matcher faces. Raw (un-normalised) text is
used: the multilingual tokenizer reads Devanagari/Tamil/... and accented French directly.

  train : fine-tune on candidate pairs of the `ce` query subset of train
  score : add `ce_logit` to work/{split}_cand.parquet pairs (all queries not used for
          training get honest scores; training queries are flagged)

Multi-GPU: launch with `accelerate launch --multi_gpu --num_processes 2 -m ber.cross_encoder ...`.
"""
import argparse
import math
import os
import time

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

import json

from ber.io import WORK, in_score_share, is_ce_query
from ber.log import Progress, get

log = get("ce")

MODEL = os.environ.get("BER_CE_MODEL", "microsoft/mdeberta-v3-base")
CE_DIR = WORK / "ce_model"
MAX_LEN = 96


def record_text(split: str, which: str) -> pl.DataFrame:
    path = WORK / (f"{split}_s1.parquet" if which == "s1" else f"{split}_pool.parquet")
    return pl.scan_parquet(path).select(
        "idx",
        "entity_id",
        text=pl.concat_str(
            [pl.col("business_name").fill_null(""), pl.col("business_address").fill_null("")], separator=" | "
        ),
    ).collect()


def pair_frame(split: str, keep_k: int, only_ce: bool | None, limit_q: int = 0) -> pl.DataFrame:
    c = pl.scan_parquet(WORK / f"{split}_cand.parquet").filter(pl.col("rr_rank") <= keep_k).select(
        "q_idx", "p_idx", "rr_rank"
    ).collect()
    q = record_text(split, "s1")
    c = c.join(q.rename({"idx": "q_idx", "entity_id": "s1_id", "text": "q_text"}), on="q_idx")
    if only_ce is not None:
        c = c.filter(is_ce_query(pl.col("s1_id")) if only_ce else ~is_ce_query(pl.col("s1_id")))
    if limit_q:
        keep = c["q_idx"].unique().sort().sample(min(limit_q, c["q_idx"].n_unique()), seed=1)
        c = c.filter(pl.col("q_idx").is_in(keep.implode()))
    p = record_text(split, "pool")
    return c.join(p.rename({"idx": "p_idx", "entity_id": "cand_id", "text": "p_text"}), on="p_idx")


class Pairs(Dataset):
    def __init__(self, a: list, b: list, y: np.ndarray | None):
        self.a, self.b, self.y = a, b, y

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return self.a[i], self.b[i], (self.y[i] if self.y is not None else 0.0)


def _collate(tok):
    def f(batch):
        a, b, y = zip(*batch)
        enc = tok(list(a), list(b), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(y, dtype=torch.float32)
        return enc
    return f


def train(keep_k: int, neg_per_q: int, epochs: float, lr: float, bs: int, limit_q: int, freeze_emb: bool) -> None:
    from accelerate import Accelerator
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    acc = Accelerator(mixed_precision="fp16")
    df = pair_frame("train", keep_k, only_ce=True, limit_q=limit_q)
    gt = pl.read_parquet(WORK / "train_gt_pairs.parquet").with_columns(label=pl.lit(1.0, pl.Float32))
    df = df.join(gt, on=["q_idx", "p_idx"], how="left").with_columns(pl.col("label").fill_null(0.0))
    # all positives + the hardest negatives (by re-ranker rank) per query
    df = df.filter((pl.col("label") == 1) | (pl.col("rr_rank") <= neg_per_q + 4)).sample(fraction=1.0, shuffle=True, seed=0)
    if acc.is_main_process:
        log.info(f"train pairs {df.height:,} from {df['q_idx'].n_unique():,} S1; positives {df['label'].mean():.3f}; "
                 f"GPUs {acc.num_processes}; model {MODEL}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    # the checkpoint is stored in fp16; AMP needs fp32 master weights
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).float()
    if freeze_emb:
        # the 250k-token multilingual embedding matrix is ~2/3 of all parameters; freezing it
        # makes each optimizer step far cheaper with little effect on fine-tuning quality
        model.get_input_embeddings().weight.requires_grad_(False)
    if acc.is_main_process:
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        smp = df.sample(min(2000, df.height), seed=0)
        lens = [len(x) for x in tok(smp["q_text"].to_list(), smp["p_text"].to_list())["input_ids"]]
        log.info(f"trainable params {n_tr / 1e6:.0f}M of {n_all / 1e6:.0f}M; pair length median "
                 f"{int(np.median(lens))} tokens, {100 * np.mean(np.array(lens) > MAX_LEN):.1f}% longer than {MAX_LEN}")
    dl = DataLoader(Pairs(df["q_text"].to_list(), df["p_text"].to_list(), df["label"].to_numpy()),
                    batch_size=bs, shuffle=True, collate_fn=_collate(tok), num_workers=2)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    steps = math.ceil(len(dl) * epochs / acc.num_processes)
    # accelerate steps a prepared scheduler once per process, so size it accordingly
    total = steps * acc.num_processes
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    model, opt, dl, sched = acc.prepare(model, opt, dl, sched)
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    step, t = 0, time.time()
    prog = Progress(log, "ce train (steps)", steps) if acc.is_main_process else None
    run_loss = 0.0
    while step < steps:
        for batch in dl:
            y = batch.pop("labels")
            loss = lossf(model(**batch).logits.squeeze(-1), y)
            acc.backward(loss)
            acc.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            run_loss = 0.98 * run_loss + 0.02 * loss.item() if step > 1 else loss.item()
            if prog:
                prog.step(1, loss=f"{run_loss:.4f}", lr=f"{sched.get_last_lr()[0]:.2e}",
                          pairs_per_s=f"{step * bs * acc.num_processes / (time.time() - t):.0f}")
            if step >= steps:
                break
    acc.wait_for_everyone()
    if acc.is_main_process:
        acc.unwrap_model(model).save_pretrained(CE_DIR)
        tok.save_pretrained(CE_DIR)
        log.info(f"saved model to {CE_DIR} after {step} steps ({time.time() - t:.0f}s)")


@torch.no_grad()
def score(split: str, keep_k: int, bs: int, share: float, limit_q: int, gate: str, band: tuple[float, float]) -> None:
    from accelerate import Accelerator
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    acc = Accelerator(mixed_precision="fp16")
    df = pair_frame(split, keep_k, only_ce=False if split == "train" else None, limit_q=limit_q)
    if share < 1.0:
        df = df.filter(in_score_share(pl.col("s1_id"), share))
    n_before = df.height
    if gate:
        # only pairs the v1 matcher is unsure about; confident pairs gain nothing from the cross-encoder
        g = pl.read_parquet(WORK / gate, columns=["q_idx", "p_idx", "prob"])
        df = df.join(g, on=["q_idx", "p_idx"], how="inner").filter(pl.col("prob").is_between(*band)).drop("prob")
        if acc.is_main_process:
            log.info(f"gate {gate} band {band}: {df.height:,} of {n_before:,} pairs kept ({df.height / max(n_before, 1):.1%})")
    df = df.sort(["q_idx", "p_idx"])
    if acc.is_main_process:
        log.info(f"scoring {split}: {df.height:,} pairs from {df['q_idx'].n_unique():,} S1 on {acc.num_processes} GPU(s)")
    shard = df.with_row_index("r").filter(pl.col("r") % acc.num_processes == acc.process_index)
    # sort by length within the shard to minimise padding
    shard = shard.with_columns(L=pl.col("q_text").str.len_chars() + pl.col("p_text").str.len_chars()).sort("L")
    tok = AutoTokenizer.from_pretrained(CE_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(CE_DIR).to(acc.device).eval().half()
    dl = DataLoader(Pairs(shard["q_text"].to_list(), shard["p_text"].to_list(), None),
                    batch_size=bs, shuffle=False, collate_fn=_collate(tok), num_workers=2)
    out, t = [], time.time()
    prog = Progress(log, f"ce score {split} (batches)", len(dl)) if acc.is_main_process else None
    for i, batch in enumerate(dl):
        batch.pop("labels")
        batch = {k: v.to(acc.device) for k, v in batch.items()}
        out.append(model(**batch).logits.squeeze(-1).float().cpu().numpy())
        if prog:
            prog.step(1, pairs_per_s=f"{(i + 1) * bs * acc.num_processes / (time.time() - t):.0f}")
    scores = np.concatenate(out) if out else np.zeros(0, np.float32)
    n_bad = int((~np.isfinite(scores)).sum())
    if n_bad:
        log.warning(f"process {acc.process_index}: {n_bad:,} non-finite scores (fp16 overflow?) -> set to -20")
        scores = np.where(np.isfinite(scores), scores, -20.0).astype(np.float32)
    shard.select("q_idx", "p_idx").with_columns(ce_logit=pl.Series(scores)).write_parquet(
        WORK / f"{split}_ce.part{acc.process_index}.parquet"
    )
    acc.wait_for_everyone()
    if acc.is_main_process:
        parts = [WORK / f"{split}_ce.part{i}.parquet" for i in range(acc.num_processes)]
        res = pl.concat([pl.read_parquet(p) for p in parts])
        res.write_parquet(WORK / f"{split}_ce.parquet")
        for p in parts:
            p.unlink()
        log.info(f"wrote {split}_ce.parquet: {res.height:,} pairs, logit mean {res['ce_logit'].mean():.2f} "
                 f"({time.time() - t:.0f}s)")
        meta = {"share": share, "keep_k": keep_k, "gate": gate, "band": list(band)}
        (WORK / f"{split}_ce_meta.json").write_text(json.dumps(meta))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["train", "score"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--keep-k", type=int, default=20)
    ap.add_argument("--neg-per-q", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--share", type=float, default=1.0, help="fraction of queries to score")
    ap.add_argument("--limit-q", type=int, default=0, help="only this many S1 records (smoke tests)")
    ap.add_argument("--no-freeze-emb", action="store_true", help="also fine-tune the word embeddings")
    ap.add_argument("--gate", default="", help="parquet (q_idx, p_idx, prob) of v1 probabilities in WORK")
    ap.add_argument("--band", nargs=2, type=float, default=[0.02, 0.98], help="score pairs with lo <= prob <= hi")
    a = ap.parse_args()
    if a.step == "train":
        train(a.keep_k, a.neg_per_q, a.epochs, a.lr, a.bs, a.limit_q, not a.no_freeze_emb)
    else:
        score(a.split, a.keep_k, a.bs, a.share, a.limit_q, a.gate, tuple(a.band))
