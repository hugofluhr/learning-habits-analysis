# Session log — cluster-level FWE for RSA searchlight; clarified the two ROI methods

**Date:** 2026-09-03
**Companion notes:**
[2026-08-31_rsa-searchlight-and-interaction.md](2026-08-31_rsa-searchlight-and-interaction.md)
(the searchlight maps and FDR results this session adds cluster-FWE to),
[2026-09-02_rsa-value-literature-search.md](2026-09-02_rsa-value-literature-search.md)

Started from Hugo's concern that fusiform was only ever meant as a category-
decoding ROI, and a question about where the value×frequency interaction is
*actually* strongest. Answered directly from the existing whole-brain
searchlight cluster tables (no new code needed there), then — prompted by
"can we do the SPM p<0.001-cluster-forming + cluster-level-FWE approach here
too" — built a permutation-based cluster-FWE script, since the parametric
(SPM/RFT) route is not well justified for searchlight maps. Also explained,
on request, the difference between the two "ROI" methods already in the repo
(`rsa_roi_results.ipynb`'s direct ROI-pattern RSA vs.
`rsa_searchlight_results.ipynb`'s searchlight-then-average-within-ROI).

---

## Findings

### 1. Whole-brain searchlight already answers "is fusiform just an artifact of the ROI list" — no

`rsa_searchlight_results.ipynb` §2/§4/§9 cluster tables are unbiased by ROI
choice (whole-brain, no masking beforehand). Value: 0 FDR-significant voxels
*anywhere*, confirming the ROI-level null isn't a consequence of which ROIs
were chosen. Frequency and the interaction both independently converge on
bilateral occipitotemporal/fusiform-VC territory, which is genuine
convergence, not circularity. Derivation: existing notebook cells 8/20
(unchanged this session, just re-read).

### 2. Two distinct "ROI" methods exist in this pipeline — not interchangeable

`rsa_roi_results.ipynb`: one crossnobis RDM per ROI, built from the ROI's
full joint voxel pattern (the standard/textbook ROI-RSA design).
`rsa_searchlight_results.ipynb`'s ROI numbers: the mean of many independent
6mm-sphere searchlight betas that happen to fall inside the ROI mask — a
different statistical operation, used here only as a secondary convergence
check against method 1 (which is why the two report different magnitudes for
the same effect — e.g. fusiform interaction +1.357 vs +0.647 — but agree in
sign and which ROIs are significant). No new code; explained from existing
cells 9-13 vs. 21-22 of `rsa_searchlight_results.ipynb`.

### 3. SPM's parametric cluster-level FWE (RFT) is not well-justified for searchlight maps — used permutation instead

Searchlight maps get spatial correlation from **overlapping spheres**, not
image smoothing — a non-stationary correlation structure that a
stationary-Gaussian RFT cluster-size null isn't built for (general
nonstationarity concern: Salimi-Khorshidi, Smith & Nichols, 2009; Li,
Nickerson & Nichols, 2016, comparing non-stationarity-corrected cluster
tests vs. TFCE). Built `multivariate/run_rsa_group_stats.py`, using
`nilearn.glm.second_level.non_parametric_inference` (sign-flip permutation)
to reproduce the same two-stage logic (cluster-forming threshold 0.001,
then cluster-level FWE) without assuming a smoothness model. Local smoke
test + full run (n=58, n_perm=10000, threshold=0.001, two-sided,
`value`/`frequency`/`interaction_value_freq`): ~85s/term on 8 cores — cheap,
not a batch job. Derivation: `run_rsa_group_stats.py` module docstring +
`rsa_searchlight_results.ipynb` §12 (new section, executed).

### 4. Cluster-FWE confirms both main results at a slightly more conservative extent; value stays null under every correction

Frequency: 2,570 cluster-mass-FWE voxels vs. 2,945 FDR. Interaction: 3,075
vs. 3,643 FDR. Value: 0 under FDR, cluster-size FWE, and cluster-mass FWE —
the null is not an artifact of the correction method. Every peak from the
FDR cluster tables (14 for frequency, 10 for the interaction) survives
cluster-FWE at the same coordinates. Cluster-mass was slightly more powerful
than cluster-size for both terms, as generally expected. Derivation:
`rsa_searchlight_results.ipynb` §12, executed via
`jupyter nbconvert --execute --inplace` on 2026-09-03.

### 5. Interaction's cluster-FWE peak table surfaces two small foci not previously highlighted

Beyond the two dominant bilateral occipitotemporal clusters (peaks
(33,−84,10) t=12.5 and (−24,−48,−12) t=10.8), two small clusters:
(45,3,34), ~630mm³ (right precentral/IFG opercular territory, eyeballed) and
(−39,−57,13), ~600mm³ (left angular gyrus/lateral occipital junction,
eyeballed). Neither is atlas-verified — same open item as the frequency
map's small clusters from 2026-08-31. Derivation: `rsa_searchlight_results.ipynb`
§12 cluster table.

---

## Code shipped

| File | Change | Git state |
|------|--------|-----------|
| `multivariate/run_rsa_group_stats.py` | New: permutation cluster-FWE (size/mass/TFCE) for any RSA searchlight term, one term at a time or `--term all` | uncommitted, this checkpoint |
| `multivariate/submit_rsa_group_stats.sh` | New: SLURM array-by-term submitter (not needed this session — ran locally instead, kept for future/heavier reruns e.g. `--tfce`) | uncommitted, this checkpoint |
| `multivariate/rsa_searchlight_results.ipynb` | New §12: cluster-FWE vs FDR comparison table + peak tables + findings, executed | uncommitted, this checkpoint |

## Data produced

Local only, not synced anywhere else: `~/phd_local/data/LearningHabits/derivatives/rsa_searchlight_group_stats/{value,frequency,interaction_value_freq}/` — t-map, logp_max_{t,size,mass} maps, mask, and `<term>_params.json` (full run parameters + subject list + runtime) for each of the 3 terms. `--tfce` was not run (would be ~45-50 min/term per the script docstring's extrapolation, not needed to answer this session's question).

## Git state

Branch `rsa-roi` (unchanged from prior sessions — still not merged to `main`). This checkpoint's 3 files are staged, not committed.

---

## Open threads

1. **Atlas-label all cluster peaks** (frequency's 14 + interaction's 10) instead of eyeballing MNI coordinates — the actual, most concrete carry-over from 2026-08-31 open thread #2, now with a cluster-FWE-confirmed peak list to label.
2. **Run `category`, `second_stim_value`, `choice_rate` through `run_rsa_group_stats.py --term all`** for completeness — only the 3 headline terms were run this session.
3. Consider running `--tfce` on `frequency` and `interaction_value_freq` for a third cluster-inference method, given it's cheap enough (~45-50 min/term) to not need the cluster.
4. `rsa-roi` branch still not merged to `main`.
