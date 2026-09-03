#!/usr/bin/env python3
"""
RSA searchlight — group-level cluster-corrected inference, one term at a time.

Companion to run_rsa_searchlight.py (per-subject spheres) and
rsa_searchlight_results.ipynb (voxelwise FDR). This script adds the
cluster-forming-threshold + cluster-level-FWE inference Hugo used in SPM
(0.001 cluster-forming, cluster-level FWE), reimplemented via sign-flip
permutation instead of SPM's parametric Random Field Theory correction.

Why not RFT/parametric cluster-level FWE directly on these maps: RFT assumes
the map's spatial autocorrelation is Gaussian and roughly stationary, with an
FWHM estimated from the residuals. Searchlight maps get their spatial
correlation from *overlapping spheres* (radius 6mm here) instead of image
smoothing — a structurally different, non-stationary correlation (it varies
near the brain edge and wherever the mask shape changes local sphere size).
Feeding that into a stationary-RFT cluster-size null is not well justified
(see e.g. Salimi-Khorshidi, Smith & Nichols 2009 on nonstationarity
correction, and Li, Nickerson & Nichols 2016 comparing non-stationarity-
corrected cluster tests against TFCE). nilearn's
`non_parametric_inference` sidesteps this: it builds the cluster-size (and
cluster-mass, and TFCE) null directly from sign-flip permutations of the
actual per-subject maps, so no smoothness model is assumed.

Local smoke test (2026-09-03, n=58, interaction_value_freq term, restricted
to voxels finite across all subjects, mask ~77.5k voxels): n_perm=200 took
~11s on 8 cores. Extrapolating, n_perm=10000 (the default here) is ~10 min
per term with cluster-size/mass thresholding, and TFCE (n_perm=50 took ~14.5s)
is ~45-50 min per term. Both comfortably fit in an interactive session or a
short batch job — this does not need to be run overnight.

Usage
-----
python multivariate/run_rsa_group_stats.py \\
    --input-dir /path/to/derivatives/rsa_searchlight \\
    --term interaction_value_freq \\
    --output-dir /path/to/derivatives/rsa_searchlight_group_stats \\
    --n-perm 10000 --threshold 0.001 --n-jobs 8

Pass --term all to loop over every term (5 main-effect betas + the
interaction) in one invocation.

Outputs (per term)
-------------------
<output-dir>/<term>/
    <term>_mask.nii.gz            — voxels finite (and non-zero) across all subjects
    <term>_t.nii.gz               — group t-map (same as the notebook's t_img)
    <term>_logp_max_t.nii.gz      — voxelwise FWE (max-stat) -log10(p)
    <term>_size.nii.gz            — per-voxel cluster size at the cluster-forming threshold
    <term>_logp_max_size.nii.gz   — cluster-size FWE -log10(p)   <- the SPM-analog map
    <term>_mass.nii.gz            — per-voxel cluster mass
    <term>_logp_max_mass.nii.gz   — cluster-mass FWE -log10(p)   <- usually more powerful than size
    <term>_tfce.nii.gz            — TFCE statistic (only if --tfce)
    <term>_logp_max_tfce.nii.gz   — TFCE FWE -log10(p)           (only if --tfce)
    <term>_params.json            — run parameters + subject list + runtime, for provenance
    rsa_group_stats_<term>.log
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.second_level import non_parametric_inference
from nilearn.image import new_img_like

MODEL_TERMS = ["category", "value", "frequency", "second_stim_value", "choice_rate"]
INTERACTION_TERM = "interaction_value_freq"
ALL_TERMS = MODEL_TERMS + [INTERACTION_TERM]


def _paths_for_term(input_dir, term):
    """Glob per-subject searchlight maps for one term, sorted by subject id."""
    if term == INTERACTION_TERM:
        pattern = f"sub-*/sub-*_rsa_searchlight_{INTERACTION_TERM}.nii.gz"
    else:
        pattern = f"sub-*/sub-*_rsa_searchlight_beta_{term}.nii.gz"
    paths = sorted(input_dir.glob(pattern))
    subjects = [p.parent.name.replace("sub-", "") for p in paths]
    return paths, subjects


def _build_group_mask(paths):
    """Voxels finite (not NaN) and non-zero in every subject's map.

    Mirrors the mask construction in rsa_searchlight_results.ipynb (cell 5 /
    cell 20): a per-sphere regression can fail (insufficient distinct
    |Delta value| classes, degenerate design) and is written as NaN, and the
    interaction term is additionally NaN wherever a subject lacks pairs in
    either the same- or different-frequency subset. Restricting to voxels
    valid in all subjects keeps the group test's degrees of freedom uniform
    across the map instead of silently dropping subjects per voxel.
    """
    ref_img = nib.load(paths[0])
    mask = np.ones(ref_img.shape, dtype=bool)
    for p in paths:
        d = nib.load(p).get_fdata()
        mask &= np.isfinite(d) & (d != 0)
    return new_img_like(ref_img, mask.astype(np.int32)), ref_img.affine, ref_img.header


def run_term(term, input_dir, output_dir, threshold, n_perm, two_sided,
            tfce, n_jobs, seed, overwrite):
    term_output = output_dir / term
    done_marker = term_output / f"{term}_params.json"
    if done_marker.exists() and not overwrite:
        logging.info(f"{term}: outputs exist, skipping (pass --overwrite to rerun)")
        return
    term_output.mkdir(parents=True, exist_ok=True)

    paths, subjects = _paths_for_term(input_dir, term)
    if len(paths) == 0:
        logging.warning(f"{term}: no searchlight maps found under {input_dir}, skipping")
        return
    logging.info(f"{term}: {len(paths)} subject maps found")

    mask_img, affine, header = _build_group_mask(paths)
    n_mask = int(mask_img.get_fdata().sum())
    logging.info(f"{term}: group mask has {n_mask:,} voxels "
                f"(finite and non-zero in all {len(paths)} subjects)")
    mask_img.to_filename(str(term_output / f"{term}_mask.nii.gz"))

    design = pd.DataFrame({"intercept": np.ones(len(paths))})

    logging.info(f"{term}: running non_parametric_inference "
                f"(threshold={threshold}, n_perm={n_perm}, two_sided={two_sided}, "
                f"tfce={tfce}, n_jobs={n_jobs})")
    t0 = time.time()
    result = non_parametric_inference(
        second_level_input=[str(p) for p in paths],
        design_matrix=design,
        mask=mask_img,
        threshold=threshold,
        n_perm=n_perm,
        two_sided_test=two_sided,
        tfce=tfce,
        n_jobs=n_jobs,
        random_state=seed,
    )
    elapsed = time.time() - t0
    logging.info(f"{term}: done in {elapsed:.1f}s")

    for key, img in result.items():
        out_path = term_output / f"{term}_{key}.nii.gz"
        img.to_filename(str(out_path))
        logging.info(f"{term}: saved {out_path.name}")

    # Quick summary at the FWE thresholds an SPM user would recognise (p<0.05)
    for key in ("logp_max_size", "logp_max_mass", "logp_max_tfce"):
        if key in result:
            logp = result[key].get_fdata()
            n_sig = int((logp > -np.log10(0.05)).sum())
            logging.info(f"{term}: {key} -> {n_sig:,} voxels survive cluster-FWE p<0.05")

    params = {
        "term": term,
        "input_dir": str(input_dir),
        "n_subjects": len(paths),
        "subjects": subjects,
        "threshold": threshold,
        "n_perm": n_perm,
        "two_sided_test": two_sided,
        "tfce": tfce,
        "n_jobs": n_jobs,
        "random_state": seed,
        "n_mask_voxels": n_mask,
        "elapsed_seconds": elapsed,
        "nilearn_version": __import__("nilearn").__version__,
    }
    with open(done_marker, "w") as f:
        json.dump(params, f, indent=2)
    logging.info(f"{term}: saved {done_marker.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Group-level cluster-FWE inference (sign-flip permutation) "
                   "for RSA searchlight term maps.")
    parser.add_argument("--input-dir", required=True,
                        help="Directory of per-subject searchlight outputs "
                            "(run_rsa_searchlight.py's --output-dir)")
    parser.add_argument("--term", required=True,
                        choices=ALL_TERMS + ["all"],
                        help="Which term to test, or 'all' for every term")
    parser.add_argument("--output-dir", required=True,
                        help="Output root directory")
    parser.add_argument("--threshold", type=float, default=0.001,
                        help="Cluster-forming threshold, as an uncorrected "
                            "p-value (default: 0.001, matching the SPM setup)")
    parser.add_argument("--n-perm", type=int, default=10000,
                        help="Number of sign-flip permutations (default: 10000)")
    parser.add_argument("--one-sided", action="store_true",
                        help="Use a one-sided test (default: two-sided)")
    parser.add_argument("--tfce", action="store_true",
                        help="Also compute TFCE (slower: ~50min/term at "
                            "n_perm=10000 vs ~10min/term for cluster-size/mass "
                            "alone, per the 2026-09-03 local smoke test)")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel jobs for the permutation test (default: 1)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random state for the permutations (default: 0, "
                            "for reproducibility)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    terms = ALL_TERMS if args.term == "all" else [args.term]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / f"rsa_group_stats_{args.term}.log"),
        ],
    )

    for term in terms:
        run_term(
            term=term,
            input_dir=input_dir,
            output_dir=output_dir,
            threshold=args.threshold,
            n_perm=args.n_perm,
            two_sided=not args.one_sided,
            tfce=args.tfce,
            n_jobs=args.n_jobs,
            seed=args.seed,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
