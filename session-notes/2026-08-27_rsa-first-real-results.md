# Session log — RSA pipeline runs for real: no value coding, but frequency is robustly represented

**Date:** 2026-08-27
**Companion note:** [2026-08-26_rsa-design-and-roi-pipeline.md](2026-08-26_rsa-design-and-roi-pipeline.md)
— the pipeline, design checks, and crossnobis validation this session actually ran.

Started as git cleanup (merging a stray worktree branch, tracking the presentation deck)
and finishing the RSA pipeline's remaining steps (env rebuild, dry-run check, cluster
submission). Ended with the first real result: the array job succeeded cleanly on the
first try, and a sanity-check comparison against the shuffled control turned up a
genuine, unplanned finding — not a bug — that reframes what this RSA pass shows.

---

## Findings

### 1. `sub-46` confirmed still missing from the BBT — real n is 58, not 59

`run_rsa_roi.py --dry-run` (added this session) swept all of `participants_mvpa.tsv`
against `bbt.csv`: 58 pass, `sub-46` fails on BBT presence, the same gap flagged during
the qvalue work and never resolved. No other subject is malformed. Derivation:
`run_rsa_roi.py check_bbt_only()`, run via the `--dry-run` flag.

### 2. No value-coding signal survives controlling for category and frequency

β(value), non-figure subset, pooled, n=58: non-significant everywhere except vmPFC
(−0.080, p=0.030) — small, negative, and not distinguishable from the shuffled control's
own ~1-in-20 false-positive rate (fusiform β(value) hits p=0.026 there too). Real null,
not underpowered: finding 3 shows the same pipeline detects a same-sized effect in the
same ROIs. Derivation: `rsa_roi_results.ipynb` §3 / Findings.

### 3. Frequency (the habit manipulation) is robustly represented — the headline result

β(frequency): wholebrain +0.220 (p<0.001), visual cortex +0.351 (p<0.001), fusiform
+0.378 (p<0.001), all n=58; shuffled control ~0 everywhere. This RSA pass shows
exposure-frequency coding, not value coding. Derivation: `rsa_roi_results.ipynb` §3 / §6.

### 4. The raw same-value contrast is finding 3 leaking through, not a second value effect

Same-value pairs carry the maximal |Δfrequency| = 2.0 by design (established in the
2026-08-26 note), so the negative raw contrast (−0.202 to −0.637, p<0.011 in
wholebrain/visual/fusiform) is exactly what a real frequency effect produces under zero
true value coding — confirmed by comparing against the frequency-partialled regression
(finding 2, which shows no such thing). The regression is the number to trust; the raw
contrast is not. Derivation: `rsa_roi_results.ipynb` §4 / Findings.

### 5. Full 8-stimulus set reproduces the predicted confound exactly

`subset='all'`: category/value/frequency all strongly significant and near-identical in
magnitude (β≈0.15–0.28, p<0.001, wholebrain/visual/fusiform) — because on that set
they're all proxies for the same figure-vs-rest split, as `rsa_design_checks.ipynb`
predicted from the BBT alone before any beta was touched. Confirms the non-figure subset
was the right primary readout. Derivation: `rsa_roi_results.ipynb` §3.

### 6. Frequency effect survives the amplitude-confound control — it is pattern geometry

The repetition-suppression alternative (high-frequency stimuli responding globally weaker,
making same-frequency pairs closer with no shared geometry) is ruled out twice: under
`--remove-mean` (job 5331516) β(frequency) *grows* slightly (visual cortex +0.351→+0.390,
fusiform +0.378→+0.425, both p<0.001), and the direct amplitude~frequency test is
negligible (visual cortex r=−0.066, p=0.004 — a weak adaptation signature exists but far
too small to drive the RDM effect). Derivation: `rsa_roi_results.ipynb` §7.

### 7. Learning-dynamics sign flip in `test` — failed robustness, discarded

Same-value contrast in fusiform: −0.63/−0.59 (learning1/learning2, both p<0.001) flipped
to +0.19 (p<0.01) in `test` under the interleaved split — but collapses under
`--within-run-split blocked` (job 5331517): fusiform +0.074 ns, visual cortex +0.109→−0.041.
Exactly the low-SNR fragility `crossnobis_validation.ipynb` §4 predicted. The learning-run
negative contrasts DO reproduce across both splits (fusiform −0.60/−0.53 blocked, p<0.001)
— consistent with finding 5's frequency-leak reading. Derivation: `rsa_roi_results.ipynb` §7.

---

## Code shipped

| What | State |
|---|---|
| `--dry-run` flag on `run_rsa_roi.py` (BBT-only precondition check) | branch `rsa-roi` (`03e2d6d`) |
| `--remove-mean` flag + per-condition amplitude CSV on `run_rsa_roi.py`; `REMOVE_MEAN=1` in submitter | branch `rsa-roi` (`b00d116`), pushed |
| `rsa_roi_results.ipynb`, executed against real n=58 data + §7 controls | branch `rsa-roi`, committed with this note |
| Merged `worktree-cluster-only-defaults` → `main` (ff, `cfcc073`) | `main`, pushed |
| Tracked `multivariate/presentation.md` + assets on `main` | `main` (`08b3701`), pushed |
| Merged `main` → `rsa-roi` (`e869e8a`) | `rsa-roi`, pushed |
| Deleted branch/worktree `worktree-cluster-only-defaults` | done |

`CLAUDE.md` (via the merged branch) now formally documents the cluster-only compute
policy — VM is off-limits, correct cluster paths, `srun`/`sbatch` requirement — matching
what was already being enforced ad hoc.

---

## Data produced (cluster)

- **Env rebuild** (job `5320542`): `rsatoolbox==0.2.0` installed into the `learning-habits`
  conda env. Confirmed working via `--validate-against-rsatoolbox` on `sub-01`
  (max |dev| = 8.24e-18 against real ROI data).
- **Real run** (job `5331366`): 58 subjects, `derivatives/rsa/`. All 58 produced
  `results.csv` (80 rows each), submitted list = produced list exactly, zero tracebacks.
- **Shuffled control** (job `5331395`, `SHUFFLE_SEED=1`): 58 subjects,
  `derivatives/rsa_shuffled/`. Same verification, `shuffle_seed=1` confirmed on all 4,640
  rows.
- **Remove-mean control** (job `5331516`): 58 subjects, `derivatives/rsa_remove_mean/`,
  including per-condition amplitude CSVs.
- **Blocked-split rerun** (job `5331517`): 58 subjects, `derivatives/rsa_blocked/`.
- All four synced locally to
  `dev_sample/bids_dataset/derivatives/{rsa,rsa_shuffled,rsa_remove_mean,rsa_blocked}/` (results.csv + RDM .npy +
  model_rdms.npz only — not the source betas) via `rsync` from `uzh.cluster.cmd`.

---

## Git state at session end

Branch `rsa-roi`, pushed through `b00d116` (the `--remove-mean` addition); the executed
`rsa_roi_results.ipynb` (with §7 and the final findings) and this note are committed on
top of it. `main` is at `08b3701` (cluster-only-defaults + presentation deck), merged
into `rsa-roi` at `e869e8a`.

---

## Open threads

1. **Trial-level repetition-suppression analysis**: the weak adaptation signature (visual
   cortex amplitude~frequency r=−0.066) motivates a proper pass — per-trial amplitude vs.
   cumulative exposure and lag since last presentation. The per-condition amplitudes are
   already saved; trial-level needs one more pass over the betas.
2. **vmPFC's marginal β(value) (p≈0.03, stable across variants)** and **striatum's
   marginal β(frequency) (p≈0.02, ditto)**: singles among ~20 uncorrected comparisons.
   FDR across the 5×4 table before treating either as more than suggestive.
3. **Frequency vs. value separability**: r=−0.346 by design. A dedicated commonality-
   analysis pass would make "frequency, not value" airtight rather than inferred from two
   adjacent regression coefficients.
4. **Searchlight RSA** remains explicitly out of scope for this pass.
