"""Write matching_results.tsv / candidate_pairs.tsv for the test split and validate them."""
import subprocess
import sys
from pathlib import Path

import polars as pl

from ber.io import DATA, OUTPUT, WORK, write_id_lists


def _to_ids(pairs: pl.DataFrame, split: str) -> pl.DataFrame:
    s1 = pl.scan_parquet(WORK / f"{split}_s1.parquet").select(q_idx="idx", s1="entity_id").collect()
    pool = pl.scan_parquet(WORK / f"{split}_pool.parquet").select(p_idx="idx", pid="entity_id").collect()
    return pairs.join(s1, on="q_idx").join(pool, on="p_idx").select("s1", "pid")


def write(matches: pl.DataFrame, candidates: pl.DataFrame, split: str = "test", out: Path = OUTPUT) -> None:
    """matches, candidates: (q_idx, p_idx). Every S1 gets a row; matches must be candidates."""
    extra = matches.join(candidates, on=["q_idx", "p_idx"], how="anti")
    assert extra.height == 0, f"{extra.height} matches are not in the candidate set"
    s1_ids = pl.scan_parquet(WORK / f"{split}_s1.parquet").select("entity_id").collect()["entity_id"]
    write_id_lists(_to_ids(matches, split), s1_ids, "pid", "matched_entity_ids", out / "matching_results.tsv")
    write_id_lists(_to_ids(candidates, split), s1_ids, "pid", "candidate_entity_ids", out / "candidate_pairs.tsv")


def validate(out: Path = OUTPUT) -> bool:
    script = DATA.parent / "utils" / "validate_submission.py"
    r = subprocess.run(
        [sys.executable, str(script), "--matching", str(out / "matching_results.tsv"),
         "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", str(DATA / "test")],
        capture_output=True, text=True,
    )
    print(r.stdout[-3000:], r.stderr[-2000:])
    return r.returncode == 0
