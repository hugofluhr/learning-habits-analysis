#!/usr/bin/env python3
"""
Build a clean OpenNeuro-ready copy of the learning-habits BIDS dataset.

What this script does:
- Copies all subject folders (all sessions/modalities) to a new output directory
- Anatomicals: renames *_T1w_defaced.nii → *_T1w.nii.gz; skips non-defaced T1w
- NIfTIs: gzips every *.nii → *.nii.gz on the way out
- fmap sidecars: rewrites IntendedFor .nii references to .nii.gz
- bold.json sidecars: fixes non-BIDS keys (SoftwareVersion → SoftwareVersions,
  drops the stray "task" key)
- Drops *_physio.log files (raw Philips logs, not BIDS physio format)
- Excludes the derivatives/ directory (phase 1: raw data only)
- Writes a corrected dataset_description.json, participants.tsv, CHANGES, and README

Usage:
    python prepare_openneuro.py --src /path/to/bids_dataset --dst /path/to/output [--dry-run]
"""

import argparse
import gzip
import json
import re
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Metadata — edit before running.
# DatasetDOI is omitted; OpenNeuro mints it after upload.
# ---------------------------------------------------------------------------
METADATA_PATCH = {
    "Name": "learning-habits",
    "BIDSVersion": "1.8.0",
    "DatasetType": "raw",
    "License": "CC0",
    "Authors": ["Hugo Fluhr", "Viktor Timokhov", "Stephan Nebe"],
    "Funding": [
        "Swiss National Science Foundation grant 207613",
        "Swiss National Science Foundation grant 215002",
    ],
    "ReferencesAndLinks": [],
}

README_TEXT = """\
# Learning Habits

This dataset contains MRI data from 70 participants collected for the Learning Habits study,
led by Hugo Fluhr, Viktor Timokhov, and Stephan Nebe at the Zurich Center for Neuroeconomics,
University of Zurich.

## Study overview

Participants performed a binary choice task during fMRI scanning.
On each trial, participants were shown a pair of stimuli and chose one. They then received feedback about both stimuli (how many points each was worth) during the learning phase. The task was designed to probe the formation of habitual choice behavior across learning.

## MRI acquisition

Scanner: Philips 3T
- Anatomical: T1-weighted (defaced for anonymisation)
- Functional: 3 runs per participant (2 learning runs, 1 test run); TR = 2.33 s, TE = 30 ms,
  40 slices, 3 mm isotropic, 0.5mm gap
- Fieldmaps: phase-magnitude pairs for each functional run

## Dataset structure

- 70 participants, 1 session each (ses-1)
- Tasks: task-learning (run-1, run-2), task-test (run-3)

## Funding

Swiss National Science Foundation grants 207613 and 215002.
"""

CHANGES_TEXT = """\
1.0.0 2026-06-25
  - Initial release of the raw BIDS dataset (70 participants, phase 1)
"""

# Subjects excluded from sharing due to acquisition problems.
EXCLUDED_SUBJECTS = {"sub-11", "sub-39", "sub-66"}

# Matches the non-defaced T1w: ends with _T1w.nii (no _defaced before .nii)
_RAW_T1W = re.compile(r"_T1w\.nii$")
# Matches the defaced T1w
_DEFACED_T1W = re.compile(r"_T1w_defaced\.nii$")


def should_skip(path: Path) -> bool:
    # Non-defaced T1w (anonymisation) and raw Philips physio logs (not BIDS physio).
    return _RAW_T1W.search(path.name) is not None or path.name.endswith("_physio.log")


def output_name(path: Path) -> str:
    name = path.name
    if _DEFACED_T1W.search(name):
        name = _DEFACED_T1W.sub("_T1w.nii", name)
    # NIfTIs are gzipped on the way out.
    if name.endswith(".nii"):
        name += ".gz"
    return name


def patch_bold_json(src: Path, dst: Path) -> None:
    """Write a *_bold.json sidecar with non-BIDS values fixed:
    - SoftwareVersion -> SoftwareVersions
    - the stray 'task' key removed
    - MRAcquisitionType 'MS' -> '2D' (BIDS only allows 2D/3D; multi-slice EPI is 2D)"""
    with src.open() as fh:
        d = json.load(fh)
    if "SoftwareVersion" in d:
        d["SoftwareVersions"] = d.pop("SoftwareVersion")
    d.pop("task", None)
    if d.get("MRAcquisitionType") == "MS":
        d["MRAcquisitionType"] = "2D"
    with dst.open("w") as fh:
        json.dump(d, fh)


def patch_fmap_json(src: Path, dst: Path) -> None:
    """Write an fmap sidecar, rewriting IntendedFor .nii references to .nii.gz
    so they keep pointing at the (now gzipped) bold images."""
    with src.open() as fh:
        d = json.load(fh)
    if "IntendedFor" in d:
        val = d["IntendedFor"]
        one = isinstance(val, str)
        lst = [val] if one else list(val)
        new = [s[:-4] + ".nii.gz" if isinstance(s, str) and s.endswith(".nii") else s
               for s in lst]
        d["IntendedFor"] = new[0] if one else new
    with dst.open("w") as fh:
        json.dump(d, fh)


def gzip_copy(src: Path, dst: Path) -> None:
    """Compress a NIfTI: src is *.nii, dst is *.nii.gz."""
    with src.open("rb") as fi, gzip.open(dst, "wb") as fo:
        shutil.copyfileobj(fi, fo)


def write_text(dst: Path, content: str, label: str, dry_run: bool) -> None:
    print(f"  write {label}")
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(content)


def prepare(src: Path, dst: Path, dry_run: bool) -> None:
    if not dry_run:
        dst.mkdir(parents=True, exist_ok=True)

    # --- Top-level files ---
    print("[top-level]")
    # .bidsignore — copy it minus the *physio.log line (those logs are dropped).
    bidsignore = src / ".bidsignore"
    if bidsignore.exists():
        lines = [ln for ln in bidsignore.read_text().splitlines()
                 if ln.strip() != "*physio.log"]
        content = "\n".join(lines) + ("\n" if lines else "")
        write_text(dst / ".bidsignore", content, ".bidsignore", dry_run)

    write_text(dst / "README", README_TEXT, "README", dry_run)
    write_text(dst / "CHANGES", CHANGES_TEXT, "CHANGES", dry_run)

    # dataset_description.json — build from scratch using METADATA_PATCH
    desc_str = json.dumps(METADATA_PATCH, indent=2)
    write_text(dst / "dataset_description.json", desc_str, "dataset_description.json", dry_run)

    # participants.tsv — participant_id only (excluded subjects omitted)
    subjects = sorted(
        p.name for p in src.iterdir()
        if p.is_dir() and p.name.startswith("sub-") and p.name not in EXCLUDED_SUBJECTS
    )
    participants_tsv = "participant_id\n" + "\n".join(subjects) + "\n"
    write_text(dst / "participants.tsv", participants_tsv, "participants.tsv", dry_run)

    # --- Subjects ---
    print(f"\n[subjects]  {len(subjects)} included  ({len(EXCLUDED_SUBJECTS)} excluded: {', '.join(sorted(EXCLUDED_SUBJECTS))})")
    n_copied = n_skipped = n_renamed = 0

    for sub_name in subjects:
        if sub_name in EXCLUDED_SUBJECTS:
            print(f"\n  {sub_name}  [excluded]")
            continue
        sub = src / sub_name
        print(f"\n  {sub_name}")
        for session in sorted(sub.iterdir()):
            if not session.is_dir():
                continue
            for modality in sorted(session.iterdir()):
                if not modality.is_dir():
                    continue
                for src_file in sorted(modality.iterdir()):
                    rel = src_file.relative_to(src)
                    if should_skip(src_file):
                        print(f"    skip  {src_file.name}")
                        n_skipped += 1
                        continue
                    out_name = output_name(src_file)
                    dst_file = dst / rel.parent / out_name
                    if out_name != src_file.name:
                        print(f"    rename  {src_file.name}  →  {out_name}")
                        n_renamed += 1
                    else:
                        print(f"    copy  {src_file.name}")
                        n_copied += 1
                    if not dry_run:
                        dst_file.parent.mkdir(parents=True, exist_ok=True)
                        if src_file.name.endswith("_bold.json"):
                            patch_bold_json(src_file, dst_file)
                        elif modality.name == "fmap" and src_file.suffix == ".json":
                            patch_fmap_json(src_file, dst_file)
                        elif src_file.suffix == ".nii":
                            gzip_copy(src_file, dst_file)
                        else:
                            shutil.copy2(src_file, dst_file)

    print(f"\n[done]  copied={n_copied}  renamed={n_renamed}  skipped={n_skipped}")
    if dry_run:
        print("[dry-run] no files were written")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="Path to source BIDS dataset")
    parser.add_argument("--dst", required=True, help="Path to output directory (will be created)")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without writing")
    args = parser.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()

    if not src.is_dir():
        raise SystemExit(f"Source not found: {src}")
    if dst.exists() and not args.dry_run:
        raise SystemExit(f"Output directory already exists: {dst}\nDelete it first or choose a new path.")

    print(f"Source : {src}")
    print(f"Output : {dst}")
    print(f"Dry-run: {args.dry_run}\n")
    prepare(src, dst, args.dry_run)


if __name__ == "__main__":
    main()
