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

from ber.io import WORK, is_ce_query

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


def pair_frame(split: str, keep_k: int, only_ce: bool | None) -> pl.DataFrame:
    c = pl.scan_parquet(WORK / f"{split}_cand.parquet").filter(pl.col("rr_rank") <= keep_k).select(
        "q_idx", "p_idx", "rr_rank"
    ).collect()
    q = record_text(split, "s1")
    c = c.join(q.rename({"idx": "q_idx", "entity_id": "s1_id", "text": "q_text"}), on="q_idx")
    if only_ce is not None:
        c = c.filter(is_ce_query(pl.col("s1_id")) if only_ce else ~is_ce_query(pl.col("s1_id")))
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


def train(keep_k: int, neg_per_q: int, epochs: float, lr: float, bs: int) -> None:
    from accelerate import Accelerator
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    acc = Accelerator(mixed_precision="fp16")
    df = pair_frame("train", keep_k, only_ce=True)
    gt = pl.read_parquet(WORK / "train_gt_pairs.parquet").with_columns(label=pl.lit(1.0, pl.Float32))
    df = df.join(gt, on=["q_idx", "p_idx"], how="left").with_columns(pl.col("label").fill_null(0.0))
    # all positives + the hardest negatives (by re-ranker rank) per query
    df = df.filter((pl.col("label") == 1) | (pl.col("rr_rank") <= neg_per_q + 4)).sample(fraction=1.0, shuffle=True, seed=0)
    acc.print(f"ce train pairs {df.height:,}  positives {df['label'].mean():.3f}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1)
    dl = DataLoader(Pairs(df["q_text"].to_list(), df["p_text"].to_list(), df["label"].to_numpy()),
                    batch_size=bs, shuffle=True, collate_fn=_collate(tok), num_workers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = math.ceil(len(dl) * epochs / acc.num_processes)
    # accelerate steps a prepared scheduler once per process, so size it accordingly
    total = steps * acc.num_processes
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    model, opt, dl, sched = acc.prepare(model, opt, dl, sched)
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    step, t = 0, time.time()
    while step < steps:
        for batch in dl:
            y = batch.pop("labels")
            loss = lossf(model(**batch).logits.squeeze(-1), y)
            acc.backward(loss)
            acc.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 500 == 0:
                acc.print(f"step {step}/{steps} loss {loss.item():.4f} {time.time() - t:.0f}s")
            if step >= steps:
                break
    acc.wait_for_everyone()
    if acc.is_main_process:
        acc.unwrap_model(model).save_pretrained(CE_DIR)
        tok.save_pretrained(CE_DIR)


@torch.no_grad()
def score(split: str, keep_k: int, bs: int, share: float) -> None:
    from accelerate import Accelerator
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    acc = Accelerator(mixed_precision="fp16")
    df = pair_frame(split, keep_k, only_ce=False if split == "train" else None)
    if share < 1.0:
        df = df.filter((pl.col("s1_id").hash(seed=5) % 1000) < int(share * 1000))
    df = df.sort(["q_idx", "p_idx"])
    acc.print(f"scoring {df.height:,} pairs")
    shard = df.with_row_index("r").filter(pl.col("r") % acc.num_processes == acc.process_index)
    # sort by length within the shard to minimise padding
    shard = shard.with_columns(L=pl.col("q_text").str.len_chars() + pl.col("p_text").str.len_chars()).sort("L")
    tok = AutoTokenizer.from_pretrained(CE_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(CE_DIR).to(acc.device).eval().half()
    dl = DataLoader(Pairs(shard["q_text"].to_list(), shard["p_text"].to_list(), None),
                    batch_size=bs, shuffle=False, collate_fn=_collate(tok), num_workers=2)
    out, t = [], time.time()
    for i, batch in enumerate(dl):
        batch.pop("labels")
        batch = {k: v.to(acc.device) for k, v in batch.items()}
        out.append(model(**batch).logits.squeeze(-1).float().cpu().numpy())
        if i % 2000 == 0:
            acc.print(f"{i}/{len(dl)} batches {time.time() - t:.0f}s")
    shard.select("q_idx", "p_idx").with_columns(ce_logit=pl.Series(np.concatenate(out))).write_parquet(
        WORK / f"{split}_ce.part{acc.process_index}.parquet"
    )
    acc.wait_for_everyone()
    if acc.is_main_process:
        parts = [WORK / f"{split}_ce.part{i}.parquet" for i in range(acc.num_processes)]
        pl.concat([pl.read_parquet(p) for p in parts]).write_parquet(WORK / f"{split}_ce.parquet")
        for p in parts:
            p.unlink()


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
    a = ap.parse_args()
    if a.step == "train":
        train(a.keep_k, a.neg_per_q, a.epochs, a.lr, a.bs)
    else:
        score(a.split, a.keep_k, a.bs, a.share)
