# Session log — ROI mask coverage inspection before expanding RSA

**Date:** 2026-08-28
**Companion note:** [2026-08-27_rsa-first-real-results.md](2026-08-27_rsa-first-real-results.md)

Started with a review of which ROIs the RSA pipeline currently uses (4: visualcortex,
fusiform, vmpfc, striatum) and which are available but unused (6 more in
`masks/MNI152NLin2009cAsym/` including the Guida 2022 habit meta-analysis mask). The
question of whether to add all ROIs or switch to searchlight RSA prompted building a
proper coverage notebook first.

---

## Findings

### 1. All 11 ROI masks survive brain mask intersection — no empties, minimal variability

Every mask × subject pair has nonzero voxels; CV < 0.02 for all ROIs. The Guida habit
mask (657 voxels) and putamen_AAL (500) are both larger than vmpfc (119) and striatum
(128) already in the RSA pipeline. Derivation: `inspect_roi_brain_mask_intersection.ipynb`
§2–§5.

### 2. Mask sizes after intersection span two orders of magnitude

visualcortex (7,599) > motor_HMAT (6,186) > premotor_HMAT (4,335) > parietal_AAL (2,859)
> motor_M1only_HMAT (1,851) > habit_Guida2022 (657) > putamen_AAL (500) > fusiform (318)
> striatum (128) > vmpfc (119) > glm_chosen_peaks_spheres6mm (58). No mask is too small
to use; the question is which are theoretically motivated. Derivation: same notebook §3.

---

## Code shipped

| What | State |
|---|---|
| `multivariate/inspect_roi_brain_mask_intersection.ipynb` — executed on cluster (59 subjects), copied back with outputs | uncommitted, staged |

---

## Data produced (cluster)

- Notebook executed via `srun --mem=8G` on cluster; all outputs (summary table, per-subject
  bar charts, CV table, box plots, Dice overlap matrix, glass brain) embedded in the
  notebook. No separate data files produced.

---

## Git state at session end

Branch `rsa-roi`, local only — notebook not yet committed. Remote is at `a9a42ec`.

---

## Open threads

1. **Inspect Dice overlap matrix and glass brain** in the notebook to check redundancy
   (especially putamen_AAL vs striatum_bartra2013, motor masks vs each other).
2. **Decide which ROIs to add to the RSA pipeline** — habit_Guida2022 and putamen_AAL are
   the strongest candidates; motor/premotor/parietal need theoretical justification.
3. **Searchlight RSA** deferred pending ROI expansion results (see companion note thread 5).
