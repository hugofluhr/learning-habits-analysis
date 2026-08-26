# Session log — moving to RSA: what the stimulus design allows, and an ROI crossnobis pipeline

**Date:** 2026-08-26
**Companion notes:**
[2026-08-11_glmsingle-conditions-and-reward-decoding.md](2026-08-11_glmsingle-conditions-and-reward-decoding.md)
(the identity-confound derivation this session builds on),
[2026-08-13_feedback-glmsingle-and-cue-redundancy.md](2026-08-13_feedback-glmsingle-and-cue-redundancy.md)

Started as "review the multivariate stuff and the session notes, then plan an RSA". The
planning turned up a design fact nobody had checked (finding 1), which changed what the RSA
should actually test, so the session ended up shipping the pipeline rather than just a plan.

---

## Findings

### 1. The image → value mapping is counterbalanced — 12 distinct assignments across 62 subjects

This is the fact that makes RSA defensible where the reward *classification* framing was not.
The 2026-08-11 note established that value is a deterministic function of identity; the open
worry was that a value effect could therefore just be a fixed idiosyncratic visual similarity
between two particular images. It cannot: those images carry different values in different
subjects. Derivation: `multivariate/rsa_design_checks.ipynb` §2 (runs off `bbt.csv` alone, no
fMRI data needed; every gate is an assert, so running all cells is the check).

### 2. But values {1, 5} always land on the `figure` category — 62/62 subjects

A hard design constraint, not a tendency. "Extreme value" is therefore perfectly confounded
with figure-vs-rest at the group level, and no amount of regression on the full 8-stimulus set
can separate them. Values {2, 3, 4} rotate freely across face/hand/house (37–47 subjects each).
**Consequence:** every RSA readout is computed twice, on all 8 stimuli and on the 6-stimulus
**non-figure subset**, and the non-figure version is the primary one. All 3 same-value pairs
are cross-category and all 3 already live in that subset. Same notebook, §3.

### 3. Frequency opposes the value hypothesis in the targeted contrast

All 3 same-value pairs carry the maximal |Δfreq| = 2, versus a mean of 0.998 for the other
cross-category pairs. So the same-value contrast is *conservative*: if frequency is
represented, it pushes that contrast negative. Fixed for every subject: corr(value, frequency)
= −0.346, corr(category, value) = −0.179. corr(category, frequency) is the only intercorrelation
that varies across subjects (−0.232 … 0.310), which is what identifies those two terms
separately at the group level. Same notebook, §4 and §5.

### 4. The local crossnobis is bit-exact against rsatoolbox and unbiased under the null

`run_rsa_roi.py` implements the crossnobis kernel itself because
`rsatoolbox.rdm.calc_rdm(..., method='crossnobis')` materialises a dense
`n_voxels × n_voxels` identity when given no noise precision — ~20 GB at whole-brain scale.
Validation: max |dev| = 1.1e-16 vs rsatoolbox 0.2.0 on unbalanced folds; mean −0.00001 over 400
null sims with 50.7% of cells negative; recovers a true squared distance of 4.0 as 3.993.
Derivation: `multivariate/crossnobis_validation.ipynb` (all synthetic, ~15 s, every section
ends in an assert). The unbiasedness result is what licenses testing group coefficients
against 0 rather than a permutation distribution. The same notebook shows the naive
non-cross-validated distance sitting at +0.044 on identical signal-free data — ~3,200x
larger — and climbing with noise level while crossnobis stays flat.

### 5. rsatoolbox must be pinned at 0.2.0, not 0.3.2

0.3.2 imports `sklearn.utils.validation.validate_data`, which needs scikit-learn ≥ 1.6; the env
pins scikit-learn 1.4 and bumping it would put every existing decoding result at risk. 0.2.0
works against numpy 1.26.4 + sklearn 1.4.2 and has a byte-identical crossnobis implementation
(checked against 0.3.2's source).

---

## Code shipped

| What | State |
|---|---|
| `multivariate/run_rsa_roi.py` — per-subject ROI crossnobis RSA | branch `rsa-roi` (`ab1b25c`), **unmerged, unpushed** |
| `multivariate/submit_rsa_roi.sh` — SLURM submitter | branch `rsa-roi` (`ab1b25c`) |
| `multivariate/rsa_roi_results.ipynb` — group notebook, **unrun** | branch `rsa-roi` (`ab1b25c`) |
| `environment.yml` — `rsatoolbox==0.2.0` added | branch `rsa-roi` (`ab1b25c`) |
| `multivariate/rsa_design_checks.ipynb` — findings 1–3, **executed** | branch `rsa-roi` |
| `multivariate/crossnobis_validation.ipynb` — finding 4, **executed** | branch `rsa-roi` |

Design decisions worth remembering:

- **Conditions are the 8 stimulus identities**, and no identity regressor is needed — every
  off-diagonal cell is a different-identity cell, so the identity model *is* the intercept.
- **Univariate noise normalisation** (per-voxel residual SD around the stimulus × run cell
  means), estimated once on all trials and reused for every scope, so per-run RDMs stay on a
  common scale. A full noise covariance is not estimable (328 trials ≪ 65k voxels).
- **No run-demeaning of features**, unlike the decoding scripts: a run-constant additive offset
  cancels inside the crossnobis kernel, since it enters both patterns of the difference
  `m_i − m_j` identically.
- **Per-run RDMs use interleaved within-run halves** (trials alternated within each stimulus ×
  run cell). Justified by the 2026-08-13 finding that adjacent-trial beta correlation is only
  0.05 for type-D betas. `--within-run-split blocked` is the alternative.

---

## Data produced

**None that survives.** A 10-subject smoke test was run on `uzh.vm` under `~/rsa_smoke/`
(scratch dir, separate venv, the VM's repo checkout untouched). Mid-session Hugo instructed
that the VM is not to be used at all, so those outputs were not retrieved and should be treated
as gone. The numbers seen there were plausible — category loaded positively in visual cortex
and fusiform, vmPFC was flat, and the shuffled-label control collapsed every effect to ~0 —
but they are **n=10 smoke-test numbers, not results**, and nothing in the repo depends on them.

Nothing has been submitted to the cluster. The cluster conda env does **not** yet have
rsatoolbox (only needed for `--validate-against-rsatoolbox`, which the submitter does not pass).

---

## Git state at session end

Branch `rsa-roi`, one commit ahead of `main` (`ab1b25c`), **not pushed**. `main` at `447b714`.
`multivariate/presentation.md` + `presentation_assets/` remain untracked, as before.

---

## Open threads

1. **Rebuild the cluster env** (`bash multivariate/build_env.sh`) to pick up
   `rsatoolbox==0.2.0`, then submit `bash multivariate/submit_rsa_roi.sh` for all 59 subjects.
2. **Run the shuffled control** into a separate tree:
   `SHUFFLE_SEED=1 OUTPUT_DIR=.../derivatives/rsa_shuffled bash multivariate/submit_rsa_roi.sh`
   (the submitter refuses to run unless the output dir looks like a shuffled tree).
3. **Fill in the findings summary** in `rsa_roi_results.ipynb` — it is written and syntax-checked
   but has never been executed against real outputs.
4. **Per-run RDMs are fragile at low SNR.** `crossnobis_validation.ipynb` §4 shows the two
   within-run split modes agree at r = 1.000 on strong signal but only 0.539 on pure noise.
   Any marginal learning-dynamics result must be reproduced under `--within-run-split blocked`
   before it is believed.
5. **Decide what to do about the `presentation.md` RSA slides**, which propose variance-partitioned
   / commonality RSA. The shipped pipeline does plain multiple regression instead; commonality is
   deferred until we see whether the coefficients are ambiguous.
6. **Searchlight RSA is deliberately out of scope** for this pass — revisit only if the ROI
   results warrant it.
7. **The `--validate-against-rsatoolbox` flag has never run on the cluster**, only locally and on
   the (now off-limits) VM. It needs rsatoolbox in the cluster env to work.
