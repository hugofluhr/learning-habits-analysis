# Session log — cluster-level FWE, atlas/functional localization, and a visual-confound audit for RSA searchlight

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

### 6. Atlas localization confirms fusiform/occipital convergence and corrects one prior eyeballed label

Built `utils/atlas.py` (`label_coordinates`: Harvard-Oxford cortical+subcortical
maxprob-thr25-2mm primary, AAL SPM12 cross-check; both atlases already cached
locally from `notebooks/roi/ROI_masks.ipynb`, no network fetch needed —
`fetch_atlas_aal()` without `version='SPM12'` fails offline, confirmed).
Applied to §12's frequency (14 peaks) and interaction (10 peaks) cluster-FWE
tables. Every peak in both dominant bilateral clusters lands in Harvard-Oxford
"Temporal Occipital Fusiform Cortex"/"Occipital Fusiform Gyrus"/"Lateral
Occipital Cortex", matching AAL `Fusiform_L/R`/`Occipital_Mid/Inf_L/R` at 0mm
— formal confirmation, not eyeballing, that the fusiform/VC result is real
anatomical convergence independent of the ROI list. **Caught one wrong label
from §10**: (42, 12, −18.5) was called "right orbitofrontal" — atlas says
Temporal Pole (`Temporal_Pole_Sup_R`), not OFC; §10's prose needs correcting.
Also resolved the two interaction small-cluster labels: (45,3,34) is
Precentral Gyrus (not IFG, as previously guessed by proximity), (−39,−57,13)
is Angular Gyrus. Derivation: `rsa_searchlight_results.ipynb` §13, executed.

### 7. Neurosynth functional decoding (existing tool, not new) surfaces two notable functional profiles

Reused `notebooks/roi/nimare_coordinates.ipynb` (Neurosynth v7 term
association, already built + data cached locally) on the same 7 frequency +
4 interaction primary peaks. Two standouts beyond the anatomical labels:
**frequency c2 (R fusiform, 42,−54,−18.5) decodes as literally the fusiform
face area** (`face` z=23.1, `fusiform`, `ffa`) — notable since face is one
of the four stimulus categories; **frequency c5 (Subcallosal cortex,
−3,9,−1) decodes as classic reward/valuation territory** (`striatum`
z=19.4, `reward`, `ventral striatum`, `nucleus accumbens`) despite only
reaching significance for **frequency**, not value, in the searchlight —
flagged as a candidate follow-up (check whether this voxel's β(value)
trends the same direction). Also: the interaction's two large clusters are
**not functionally homogeneous** despite both being anatomically
"fusiform" — c1 (R) decodes as generic visual/occipitotemporal, c2 (L)
decodes as parahippocampal/scene-selective (`parahippocampal`, `scenes`,
`place`), closer to PPA than FFA. Derivation:
`notebooks/roi/nimare_coordinates.ipynb`, new section, executed.

### 8. Visual-category-confound audit, prompted by finding 7's FFA/PPA decoding — four checks, no evidence of a group-level confound

Hugo asked whether the frequency effect is just visual category (face/house
selectivity) leaking through. Checked four ways, all using data already on
disk (no cluster rerun):
- **VIF was stim1-category only** — confirmed from `run_rsa_roi.py`'s
  `model_rdms['category']` (never included a stim2-category term); the
  2026-08-30 VIF check doesn't speak to a second-stimulus-category leak at
  all. Correction to an earlier answer in this same conversation.
- **Signed group-level corr(category, frequency)**: face +0.02, hand −0.09,
  house +0.07, all p>0.13 — no population-level directional tilt (would
  need one for a coarse category control to leave a systematic group bias).
  Derivation: `rsa_design_checks.ipynb` §6 (new, executed).
- **Image→frequency rotation**: 8 distinct assignments across 62 subjects
  (mirrors §2's 12 value assignments) — no single image is "the high-
  frequency face" across the population. Same §6.
- **Partial corr(frequency, second_stim_category | second_stim_value)**:
  ≈0 for all 3 categories, all p>0.16 — `second_stim_category` (Hugo's
  proposed addition) would be near-redundant given `second_stim_value`
  already in the model; frequency is manipulated via value-based pairing,
  not category-based. Derivation: `rsa_design_checks.ipynb` §7 (new,
  executed). **Reasoning verdict: theoretically sound mechanism, empirically
  low-value here — not implemented.**

### 9. Literature check: reward/value history modulating visual cortex is an established, real phenomenon — not just a confound story

Verified via Scite (not memory): Antono, Dang & Auksztulewicz (2023,
preprint) show via multivariate fMRI that previously reward-associated
cues enhance target representation in early visual areas; sits within the
broader value-driven-attention literature (Anderson, 2017). So a
choice-frequency (reward-history) effect in category-selective visual
cortex is also what a genuine, literature-supported effect looks like —
doesn't resolve the confound question either way, but the prior shouldn't
default to "artifact."

### 10. Targeted frequency-label permutation test — the sharp version of the confound check, and it's decisive

The existing shuffled-label control (`run_rsa_roi.py`'s `shuffle_seed`) is
blunter than it looks: it scrambles neural-pattern-to-identity
correspondence entirely, destroying category/value/frequency structure
simultaneously — a pipeline sanity check, not a targeted test. Built the
targeted version: hold the real crossnobis RDM and real category/value/
second_stim_value/choice_rate fixed, permute only the ±1 frequency label
across the 6 non-figure stimuli (all C(6,3)=20 balanced relabelings
enumerable per subject), draw one relabeling per subject per iteration to
build a group-level null (50000 draws). **Wholebrain/VC/fusiform: perm
p=0.00002 — the real label assignment is far outside the null of any
balanced, category-correlated relabeling of the same real stimuli.**
Striatum/parietal marginal (p=0.024/0.034); vmPFC/habit/putamen/premotor
null. Derivation: `rsa_roi_results.ipynb` §9 (new, executed).

---

## Code shipped

| File | Change | Git state |
|------|--------|-----------|
| `multivariate/run_rsa_group_stats.py` | New: permutation cluster-FWE (size/mass/TFCE) for any RSA searchlight term, one term at a time or `--term all` | `main` (`b34d11a`), pushed-pending |
| `multivariate/submit_rsa_group_stats.sh` | New: SLURM array-by-term submitter (not needed this session — ran locally instead, kept for future/heavier reruns e.g. `--tfce`) | `main` (`b34d11a`) |
| `multivariate/rsa_searchlight_results.ipynb` | New §12 (cluster-FWE vs FDR) + §13 (atlas localization), both executed | `main` (`b34d11a`, `117e3d6`) |
| `utils/atlas.py` | New: `label_coordinates()`, MNI coordinate -> Harvard-Oxford/AAL label lookup, offline-cached atlases | `main` (`117e3d6`) |
| `notebooks/roi/nimare_coordinates.ipynb` | New section: Neurosynth decoding of the RSA searchlight primary peaks, executed | `main` (`1150298`) |
| `multivariate/rsa_design_checks.ipynb` | New §6 (image→frequency rotation + signed category/frequency correlation) and §7 (second_stim_category partial correlation), both executed | this checkpoint, uncommitted |
| `multivariate/rsa_roi_results.ipynb` | New §9: targeted frequency-label permutation test, executed | this checkpoint, uncommitted |

## Data produced

Local only, not synced anywhere else: `~/phd_local/data/LearningHabits/derivatives/rsa_searchlight_group_stats/{value,frequency,interaction_value_freq}/` — t-map, logp_max_{t,size,mass} maps, mask, and `<term>_params.json` (full run parameters + subject list + runtime) for each of the 3 terms. `--tfce` was not run (would be ~45-50 min/term per the script docstring's extrapolation, not needed to answer this session's question).

## Git state

Working directly on `main` this session (not `rsa-roi` — the branch mentioned
in prior companion notes belongs to earlier sessions; this session's git
status at start showed `main` already checked out with the §12 changes
pending). Three commits done mid-session (`b34d11a` §12 cluster-FWE,
`117e3d6` §13 atlas localization + `utils/atlas.py`, `1150298` Neurosynth
decoding). This checkpoint's `rsa_design_checks.ipynb` §6-7,
`rsa_roi_results.ipynb` §9, and this note's edits are staged, not yet
committed.

---

## Open threads

1. **Correct §10 finding 2's "right orbitofrontal" label to Temporal Pole**
   in `rsa_searchlight_results.ipynb` — §13 finding 2 caught this but §10's
   prose itself wasn't edited in place (kept as a historical record of the
   original eyeballed pass, with §13 as the correction layer instead).
   Worth deciding whether to fix §10 directly or leave the correction
   pointer as-is.
2. **Persist `labeled_tables` to disk** (e.g. CSV under `rsa_searchlight_group_stats/`) rather than leaving it as in-notebook-only kernel state — anyone rerunning the notebook reproduces it, but it's not currently exported for reuse elsewhere (e.g. the presentation deck).
3. **Run `category`, `second_stim_value`, `choice_rate` through `run_rsa_group_stats.py --term all`** for completeness — only the 3 headline terms were run this session.
4. Consider running `--tfce` on `frequency` and `interaction_value_freq` for a third cluster-inference method, given it's cheap enough (~45-50 min/term) to not need the cluster.
5. **Follow up on finding 7's subcallosal/ventral-striatum candidate**: check whether β(value) at that specific voxel/small neighborhood trends positive, even though it didn't survive the whole-brain value map — a small, targeted test (not another whole-brain correction pass).
6. **The §9 permutation test currently only covers the ROI-level analysis** (`rsa_roi_results.ipynb`). Extending it to the searchlight (per-voxel or at least at the peak coordinates from §12/§13) would close the confound question at the resolution the FFA/PPA decoding finding was actually made at — not done this session, would need `run_rsa_searchlight.py`-level access to per-subject sphere data, more involved than the ROI version.
7. The visual-confound audit (findings 8-10) doesn't rule out every possible finer-grained visual confound (e.g. low-level pixel statistics unrelated to category) — only category-driven and second-stimulus-pairing-driven ones. Not flagged as urgent given how decisively §9's permutation test came out.
