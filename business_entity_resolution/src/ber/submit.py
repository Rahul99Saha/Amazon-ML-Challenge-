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
    script = DATA.resolve().parent / "utils" / "validate_submission.py"
    r = subprocess.run(
        [sys.executable, str(script), "--matching", str(out / "matching_results.tsv"),
         "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", str(DATA / "test"), "--check-ids"],
        capture_output=True, text=True,
    )
    print(r.stdout[-3000:], r.stderr[-2000:])
    return r.returncode == 0


def from_candidates(split: str = "test", out: Path = OUTPUT) -> None:
    """Produces matching_results.tsv and candidate_pairs.tsv directly from {split}_cand.parquet."""
    from ber.decide import expected_f05_sets, one_to_one
    cand_path = WORK / f"{split}_cand.parquet"
    assert cand_path.exists(), f"{cand_path} does not exist. Run candidates.py --splits {split} first."
    cand = pl.read_parquet(cand_path)
    cands_df = cand.select("q_idx", "p_idx")
    scores = cand.rename({"rr_prob": "prob"}).select("q_idx", "p_idx", "prob")
    matches = expected_f05_sets(one_to_one(scores))
    print(f"Produced {matches.height:,} final matches from {cands_df.height:,} candidates")
    write(matches.select("q_idx", "p_idx"), cands_df, split=split, out=out)
    if validate(out):
        print("VALIDATION PASSED: Files are ready to submit!")
    else:
        print("Validation check finished.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    from_candidates(args.split)
