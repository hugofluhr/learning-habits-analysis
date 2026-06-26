#!/usr/bin/env python3
"""
Generate BIDS events.tsv files for the learning-habits dataset.

For each subject x run, loads behavioral data via the Subject/Block classes, calls create_event_df(), adds derived trial-level columns (mirroring create_bbt.py), and merges onto every event row.

Output columns:
  onset, duration, trial_type, trial,
  shift, left_stim, right_stim, left_value, right_value,
  left_stim_name, right_stim_name, left_stim_cat, right_stim_cat,
  first_stim, second_stim, first_stim_name, second_stim_name,
  first_stim_cat, second_stim_cat, first_stim_value, second_stim_value,
  first_stim_frequ, second_stim_frequ,
  action, rt, reward, correct,
  stim_chosen, stim_unchosen, reward_chosen, reward_unchosen, diff_val

Usage:
    conda run -n neuroim python generate_events_tsv.py \\
        --dst /path/to/bids_dataset_openneuro [--dry-run]
"""

import argparse
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.data import Subject  # noqa: E402

BASE_DIR = Path("/mnt/data2/learning-habits")

# acquisition not complete for those.
EXCLUDED_SUBJECTS = {"sub-11", "sub-39", "sub-66"}

RUN_MAP = {
    "learning1": ("learning", 1),
    "learning2": ("learning", 2),
    "test":      ("test",     3),
}

KEEP_COLS = [
    "trial",
    # spatial
    "shift", "left_stim", "right_stim", "left_value", "right_value",
    "left_stim_name", "right_stim_name", "left_stim_cat", "right_stim_cat",
    # temporal
    "first_stim", "second_stim", "first_stim_name", "second_stim_name",
    "first_stim_cat", "second_stim_cat",
    "first_stim_value", "second_stim_value",
    "first_stim_frequ", "second_stim_frequ",
    # behavior
    "action", "rt", "reward", "correct",
    # derived
    "stim_chosen", "stim_unchosen", "reward_chosen", "reward_unchosen", "diff_val",
]

# Integer-valued columns that carry NaN on missed trials. Without an explicit
# nullable-int dtype, pandas floats them (e.g. action -> "1.0"), which fails the
# BIDS schema validator (action has categorical Levels "1"/"2"). Cast to "Int64"
# so they serialise as plain ints with "n/a" for missing.
INT_NA_COLS = [
    "action", "reward", "correct",
    "stim_chosen", "stim_unchosen", "reward_chosen", "reward_unchosen",
]


def add_derived_columns(trials):
    """Add stim_chosen/unchosen and reward_chosen/unchosen — mirrors create_bbt.py lines 42-65."""
    has_resp = trials["action"].notna()
    # Fix the known 3-trial mismatch in raw chosen_stim by recomputing from action
    trials["stim_chosen"] = np.where(
        has_resp,
        np.where(trials["action"] == 1, trials["left_stim"], trials["right_stim"]),
        np.nan,
    )
    trials["stim_unchosen"] = np.where(
        has_resp,
        np.where(trials["action"] == 1, trials["right_stim"], trials["left_stim"]),
        np.nan,
    )
    trials["reward_chosen"] = np.where(
        has_resp,
        np.where(trials["action"] == 1, trials["left_value"], trials["right_value"]),
        np.nan,
    )
    trials["reward_unchosen"] = np.where(
        has_resp,
        np.where(trials["action"] == 1, trials["right_value"], trials["left_value"]),
        np.nan,
    )
    trials["diff_val"] = trials["left_value"] - trials["right_value"]
    return trials


def make_events(block):
    block.create_event_df()

    trials = add_derived_columns(block.trials.copy())  # preserve 1-based index
    trials = trials[[c for c in KEEP_COLS if c in trials.columns]]

    events = block.events.merge(trials, on="trial", how="left")

    for col in INT_NA_COLS:
        if col in events.columns:
            events[col] = events[col].astype("Int64")

    lead = ["onset", "duration", "trial_type"]
    rest = [c for c in events.columns if c not in lead]
    return events[lead + rest].sort_values("onset").reset_index(drop=True)


def generate_events(dst: Path, dry_run: bool) -> None:
    subjects = sorted(
        p.name for p in dst.iterdir()
        if p.is_dir() and p.name.startswith("sub-") and p.name not in EXCLUDED_SUBJECTS
    )
    print(f"Found {len(subjects)} subjects in output dataset\n")

    for sub_name in subjects:
        print(f"[{sub_name}]")
        try:
            sub = Subject(base_dir=str(BASE_DIR), subject_id=sub_name)
        except Exception as e:
            print(f"  ERROR loading subject: {e}")
            continue

        for block_attr, (task, run_num) in RUN_MAP.items():
            block = getattr(sub, block_attr)
            try:
                events = make_events(block)
            except Exception as e:
                print(f"  ERROR on {block_attr}: {e}")
                continue

            fname = f"{sub_name}_ses-1_task-{task}_run-{run_num}_events.tsv"
            out_path = dst / sub_name / "ses-1" / "func" / fname
            print(f"  write  {fname}  ({len(events)} rows × {len(events.columns)} cols)")
            if not dry_run:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                events.to_csv(out_path, sep="\t", index=False, na_rep="n/a")

    if dry_run:
        print("\n[dry-run] no files were written")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dst", required=True,
                        help="Path to the OpenNeuro output BIDS dataset")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print actions without writing files")
    args = parser.parse_args()

    dst = Path(args.dst).expanduser().resolve()
    if not dst.is_dir():
        raise SystemExit(f"Output dataset not found: {dst}")

    print(f"Output  : {dst}")
    print(f"Dry-run : {args.dry_run}\n")
    generate_events(dst, args.dry_run)


if __name__ == "__main__":
    main()
