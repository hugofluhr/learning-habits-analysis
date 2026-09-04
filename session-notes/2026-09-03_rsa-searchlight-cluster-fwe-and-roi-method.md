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
Checking Hugo's hypothesis that frequency should build up gradually across
runs instead surfaced the opposite pattern (strong from run 1, collapsing
in `test`) and — while investigating whether that was frequency-specific —
an outright terminology error carried across several session notes and
notebooks: the "β(value)≈0" framing was wrong (it's small and
significantly negative), which then got corrected everywhere it appeared.

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

### 12. β(frequency) does NOT grow across runs as hypothesized — strong from run 1, collapses in `test`

Hugo's hypothesis was that choice-frequency accumulates from 0 over the
session, so β(frequency) should be smaller in `learning1` than later runs.
**Opposite pattern**: already strong in `learning1` (VC +0.212\*\*\*, fusiform
+0.364\*\*\*), not smaller than `learning2` (VC +0.180\*\*\*, fusiform
+0.278\*\*\*), then **collapses toward 0 in `test`** (VC −0.018 ns, fusiform
−0.054 ns; both significantly below `learning1`, paired p<0.0001).
**Replicates under the blocked within-run split** (test: VC +0.105\* weak,
fusiform +0.023 ns — still significantly below `learning1`) — not the
interleaved-split fragility flagged in 2026-08-27's note. Derivation:
`rsa_roi_results.ipynb` §5b (new, executed) + blocked-split check (new,
executed).

### 13. The same test-phase collapse also hits β(value) — and reveals it was never actually ≈0

Checking whether this is frequency-specific: **value shows the identical**
**qualitative pattern.** Pooled β(value) (non-figure, objective model) is
small and **significantly negative** (fusiform β=−0.118 p=0.0007, VC
β=−0.080 p=0.023) — not ≈0 as several session notes and this notebook's own
Findings had described it. Per-run: fusiform −0.175\*\*\* (learning1),
−0.194\*\*\* (learning2), **+0.120 ns (test)** — a sign flip. VC parallels
this. The pooled negative effect is explained by the already-established
interaction (different-frequency pairs, 9/subject, negative slope;
same-frequency pairs, 6/subject, positive slope — the larger group
dominates the non-partitioned regression) but the run-by-run breakdown
shows this negative effect is itself driven entirely by the two learning
runs and vanishes/flips in `test`. **Terminology corrected across
`rsa_roi_results.ipynb`, `rsa_searchlight_results.ipynb`,
`presentation.md`, and `session-notes/2026-08-27_rsa-first-real-results.md`
this session** — "β(value)≈0" replaced with the actual signed, significant
number everywhere it appeared. Derivation: `rsa_roi_results.ipynb`
Addendum (new, executed).

### 14. Category (reward-independent) does NOT show the same test-phase pattern — argues against a pure generic-noise account

Decisive check for whether findings 12-13 are "test phase is just noisier"
vs. something specific to reward-history-linked variables: β(category) is
already weak everywhere (consistent with prior findings) and stays flat
across all three runs (VC/fusiform test-vs-learning1: p=0.88/0.64) — no
directional shift, only a mild SEM increase (~15-20%) consistent with
somewhat more noise in `test`, nowhere near enough to explain frequency's
collapse or value's sign flip. Read as: the effect is not simply "test is
noisier for everything" — frequency and value specifically depend on the
active-learning/feedback context in a way category doesn't. Derivation:
`rsa_roi_results.ipynb`, new cell (unnamed, right after §5b).

**Design correction from Hugo, changes how finding 14 should be read:**
`test` is not "no manipulation" — it presents *all* pairwise stimulus
combinations, including same-value pairs that never occur in
`learning1`/`learning2` by design. This is the paradigm's habit-diagnostic
phase (does choice-frequency bias choice when value can't explain a
preference), and Hugo confirms the behavioural habit effect **is** present
there, small but robust. So "behavioural extinction of the frequency
split" (the second candidate mechanism floated below) is ruled out as
stated — behavior doesn't extinguish. The live question becomes why a
robust *behavioral* habit effect in `test` has no corresponding *neural*
signature there, given the same-frequency-signature is strong in
`learning1`/`learning2`.

### 15. Ruled out: learning-scope collinearity between frequency and second_stim_value does not explain the collapse

Hugo's mechanism: the pairing manipulation structurally couples frequency
and second_stim_value *specifically during learning* (by construction), and
the 2026-08-30 VIF check (run on *pooled* scope, which mixes in `test`'s
unbiased all-pairs design) could have diluted/masked much worse
learning-only collinearity — making the learning-phase β(frequency)
estimates less separable/trustworthy than the headline diagnostic
suggested. **Checked directly, per scope: not supported.** VIF(frequency)
is low (~1.2-1.6) and essentially identical across `learning1` (median
1.26), `learning2` (1.34), `test` (1.32), and `pooled` (1.24); raw
corr(frequency, second_stim_value) is modest (~−0.11) in all scopes. If
anything `test` has the one elevated-VIF outlier subject (max 7.72), not
learning. This specific confound channel doesn't explain why learning is
strong and test is null. Derivation: `rsa_roi_results.ipynb`, new cell
(unnamed, right after the category negative control).

---

## Code shipped

| File | Change | Git state |
|------|--------|-----------|
| `multivariate/run_rsa_group_stats.py` | New: permutation cluster-FWE (size/mass/TFCE) for any RSA searchlight term, one term at a time or `--term all` | `main` (`b34d11a`), pushed-pending |
| `multivariate/submit_rsa_group_stats.sh` | New: SLURM array-by-term submitter (not needed this session — ran locally instead, kept for future/heavier reruns e.g. `--tfce`) | `main` (`b34d11a`) |
| `multivariate/rsa_searchlight_results.ipynb` | New §12 (cluster-FWE vs FDR) + §13 (atlas localization), both executed | `main` (`b34d11a`, `117e3d6`) |
| `utils/atlas.py` | New: `label_coordinates()`, MNI coordinate -> Harvard-Oxford/AAL label lookup, offline-cached atlases | `main` (`117e3d6`) |
| `notebooks/roi/nimare_coordinates.ipynb` | New section: Neurosynth decoding of the RSA searchlight primary peaks, executed | `main` (`1150298`) |
| `multivariate/rsa_design_checks.ipynb` | New §6 (image→frequency rotation + signed category/frequency correlation) and §7 (second_stim_category partial correlation), both executed | `main` (`8a71250`) |
| `multivariate/rsa_roi_results.ipynb` | New §9 (targeted frequency-label permutation test), §5b (frequency per-run dynamics + blocked-split robustness check), category per-run control, per-scope VIF/collinearity check (ruled out), and an Addendum correcting the pooled β(value) terminology — all executed | `8a71250` (§9) + this checkpoint (§5b, category control, VIF check, Addendum) |
| `multivariate/rsa_searchlight_results.ipynb` | §9/§10 prose corrected to match the pooled β(value) terminology fix (no new analysis) | this checkpoint, uncommitted |
| `multivariate/presentation.md` | "What the interaction means?" slide + speaker notes corrected to the actual pooled β(value) numbers; removed a stray leftover "Not sure how to interpret this" line | this checkpoint, uncommitted |
| `session-notes/2026-08-27_rsa-first-real-results.md` | Added a dated correction pointer to finding 5 (kept the original preliminary text as historical record, per checkpoint convention) | this checkpoint, uncommitted |
| `multivariate/rsa_design_checks.ipynb` | New §8/§8b: learning/test pairing-structure design facts (finding 16) | `main` (`f060db4`) |
| `multivariate/run_beta_crossrun_reliability.py`, `submit_beta_crossrun_reliability.sh` | New: finishes `glmsingle_qc.ipynb` §8 (finding 17) | `main` (`f060db4`) |
| `multivariate/aggregate_crossrun_reliability.py` | New: aggregation script for the above | `main` (`653c017`) |
| `multivariate/run_rsa_partner_context.py`, `submit_rsa_partner_context.sh` | New: partner-context/category-leak crossnobis test, `--scope {learning,test}` (findings 18-19) | `main` (`3f087ef`) |
| `multivariate/rsa_partner_context_results.ipynb` | New: group results notebook for findings 18-19, executed | this checkpoint |
| `multivariate/run_rsa_roi.py` | Added `--symmetric` flag: `profile_dist_rdm()` helper + s2_category/s2_frequency/s2_identity predictors (finding 20) | `stim2-contamination-tests` (`b43d27c`) |
| `multivariate/check_symmetric_vif.py` | New: pre-flight VIF check for the symmetric model, BBT-only | `stim2-contamination-tests` (`b43d27c`) |
| `multivariate/submit_rsa_roi.sh` | Added `SYMMETRIC=1` env var with separate-output-tree guardrail | `stim2-contamination-tests` (`b43d27c`) |
| `multivariate/run_stim2_decoding.py`, `submit_stim2_decoding.sh` | New: stim-2 category decoding test (finding 21, results pending) | `stim2-contamination-tests` (`b43d27c`) |
| `utils/data.py` | Moved `load_string_target_from_bbt` here from `run_rsa_partner_context.py` (avoids a circular import with `run_rsa_roi.py`) | `stim2-contamination-tests` (`b43d27c`) |
| `multivariate/rsa_roi_results.ipynb` | New §10 (symmetric-model comparison, finding 20) + "9." addition to top-level Findings, both executed | this checkpoint |
| `multivariate/stim2_decoding_results.ipynb` | New: group results notebook for finding 21, built but not yet executed (job 5495879 still running) | this checkpoint |

## Data produced

Local only, not synced anywhere else: `~/phd_local/data/LearningHabits/derivatives/rsa_searchlight_group_stats/{value,frequency,interaction_value_freq}/` — t-map, logp_max_{t,size,mass} maps, mask, and `<term>_params.json` (full run parameters + subject list + runtime) for each of the 3 terms. `--tfce` was not run (would be ~45-50 min/term per the script docstring's extrapolation, not needed to answer this session's question).

Cluster: `shares-hare/ds-learning-habits/derivatives/glmsingle_qc/sub-*/*_crossrun_reliability.csv` (job 5484108, all 59 subjects) and `shares-hare/ds-learning-habits/derivatives/rsa_partner_context/sub-*/` (jobs 5484596/5484846/5484905/5484906, all 59 subjects minus `sub-46`) — both mirrored locally to `~/phd_local/data/LearningHabits/dev_sample/bids_dataset/derivatives/{glmsingle_qc,rsa_partner_context}/` for notebook use.

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
8. **What actually causes the `test`-phase collapse (findings 12-14)?** The "behavioural extinction of the frequency split" account is ruled out (finding 14's correction) — the behavioural habit effect is real and robust in `test`'s same-value trials, per Hugo. The "learning-scope collinearity" account is also ruled out (finding 15). **Still unexplained**: a robust *behavioral* habit effect in `test` with no corresponding *neural* signature there, while `learning1`/`learning2` show a strong neural signature. Candidate directions not yet tried: (a) a genuinely feedback/prediction-error-locked neural mechanism (behavior can be habit-driven without the *representational geometry* this RSA measures being the thing that drives it); (b) check the same-value trials specifically within `test` (the diagnostic subset) rather than the whole-`test` RDM used so far — the current `test`-scope RSA pools all trial types together, diluting exactly the trials where the behavioral effect is cleanest.
9. **Propagate the same "does X collapse in test" check to the value×frequency interaction** — findings 12-14 only checked the two main effects (frequency, value) and the negative-control (category); the interaction itself (§9 of `rsa_searchlight_results.ipynb`, the headline searchlight finding) hasn't been checked for the same run-by-run pattern yet.

### 16-19. RSA partner-context pollution: betas carry information about the co-present stimulus, not just the target stimulus (started from open thread 8, ended up bigger than that)

Full derivation and executed figures: `multivariate/rsa_partner_context_results.ipynb`
(new). Code: `multivariate/run_rsa_partner_context.py` + `submit_rsa_partner_context.sh`
(new; module docstring has the complete design-decision trail).

### 16. Pairing structure: `learning` and `test` sample completely different sets of stimulus pairs — a new, untested candidate mechanism for the test-phase collapse (open thread 8)

Directly inspected `bbt.csv`'s pairing structure (`left_stim`/`right_stim`/`block`), since
`run_rsa_roi.py` conditions are per-stimulus means over trials (`cond_idx` from
`stim_name`), so what those trials *are* depends on pairing. **`learning1`/`learning2`:
every subject samples only 8 of the 28 possible pairs** (a fixed adjacent-value chain —
each stimulus has 1-2 fixed partners for the whole phase), each pair repeated either 6 or
18 times, split exactly 4-vs-4 pairs/subject/block, no exceptions across all 62 subjects
— the choice-frequency label *is* this repetition ratio. **`test`: every subject samples
all 28/28 pairs**, 25 shown 4x each and exactly the 3 same-value ("habit-diagnostic")
pairs shown 12x, again 62/62 subjects with zero exceptions. So each stimulus's RSA
condition mean is built from a nearly homogeneous 1-2-partner trial pool in `learning`
but a heterogeneous ~7-partner pool in `test` — a within-condition context-diversity
change between phases, distinct from the mean-level confound accounts already ruled out
(findings 8-10, 15). Not yet tested against neural data — a candidate mechanism, not a
result. Derivation: `rsa_design_checks.ipynb` §8 (new, executed).

### 17. Cross-run per-stimulus pattern reliability does NOT drop from `learning` to `test` — rules out the "test is just noisier overall" version of finding 16

Direct neural check, prompted by Hugo asking whether this was already covered by
`glmsingle_qc.ipynb` §8 (it wasn't — written but never run to completion or
persisted anywhere, locally or on the cluster: no `qc_group_summary.csv` existed).
Finished it as `run_beta_crossrun_reliability.py` (SLURM job 5484108, all 59
subjects, <1 min): per-stimulus Pearson r between mean beta patterns across each
pair of runs. **Wholebrain**: `learning`↔`test` reliability (0.331) is actually
*higher* than `learning1`↔`learning2` (0.276, paired t=-2.10, p=0.04).
**Visualcortex**: no difference (0.705 vs 0.709, p=0.78). **Fusiform**:
`test` marginally *lower* (0.797 vs 0.821) but the effect is tiny (Δ=0.024,
p=0.007) — nowhere near large enough to explain the RSA collapse from
strongly-significant to null. **Consequence**: finding 16's pairing-structure
account can't be "test patterns are broadly noisier" — overall per-stimulus
pattern reliability is essentially preserved. Whatever finding 16 predicts must
be specific to the frequency/value-relevant subspace of the pattern, not the
whole pattern (consistent with finding 14: category doesn't collapse either) —
sharpens rather than kills the hypothesis, and is the reason a coarse
reliability check isn't sufficient; a partner-conditioned test targeted at that
subspace is still needed. Derivation: `run_beta_crossrun_reliability.py` +
`submit_beta_crossrun_reliability.sh` (new, run on cluster), aggregated locally
via `aggregate_crossrun_reliability.py` (new, committed).

### 18. Targeted neural test: `learning`'s dominant partner leaves a large residual signature, comparable in size to the identity signal itself

Hugo's sharper reframing of finding 16: it's not just fewer contexts in
`learning` vs `test`, it's that every stimulus has one *dominant* partner
(~75% of its own trials — the choice-frequency label's 6:18 ratio), so its
condition mean is heavily weighted toward "this stimulus, with its majority
partner." Built `run_rsa_roi.py`-style crossnobis test per stimulus: dominant-
vs-minor-partner-trials distance, holding identity fixed (SLURM jobs 5484596/
5484905, n=58). **Confirmed decisively**: `partner_distance` +0.030 (wholebrain)
to +0.064 (fusiform), all p<1e-20 — comparable in magnitude to the ordinary
between-stimulus identity-discrimination signal (`betweenstim_distance`
+0.009 to +0.073 in the same units/subjects). A dry run against `bbt.csv` before
touching the cluster caught a design subtlety first: GLMsingle betas are locked
to FIRST-stimulus onset only, so a stimulus's beta condition only sees the
subset of its pair-level appearances where it happened to be shown first —
realized dominant/minor counts range 1-11 (mean 6.0, sd 1.8) across 496
subject×stimulus cells, not a clean 18:6 — handled via a per-stimulus
independent test, an interleaved pooled-block 2-fold split, and a minimum-
trial-count gate (≥3). Derivation: `multivariate/run_rsa_partner_context.py`
(new) + `rsa_partner_context_results.ipynb` §1 (new, executed).

### 19. Same test in `test`: the naive prediction failed, and the failure revealed a bigger, more general confound — partner CATEGORY leaks into the beta

Naive prediction: `test` has no dominant partner, so if finding 18 is pure
repetition-context pollution, this distance should shrink there. **It didn't —
it's significantly LARGER** (+0.057 to +0.081, all p<1e-17; paired vs `learning`,
p from 0.015 to 1e-10, all 3 masks). Traced to a construct mismatch, not a
failed hypothesis: `test`'s only analog to a dominant partner is the same-value
("habit-diagnostic") partner — a qualitatively different decision problem
(tied value, no objectively correct choice), not a repetition-frequency analog.
A first attempt at a construct-matched null (arbitrary partner-id split among
the 6 different-value partners) backfired instructively: it came back the
LARGEST number yet (~0.095-0.097) because an id-sorted split has no control
over the CATEGORY composition of its two groups. Replaced with the direct,
honest version — **hold first-stimulus identity fixed and split by the
partner's CATEGORY** — and this is the headline result: `category_distance`
in `test` is +0.12 to +0.17 (all p<1e-27), roughly **2x finding 19's own
value-tie-specific effect and the largest, most significant number in the
whole investigation**. (In `learning`, category_distance mostly reproduces
finding 18, since only 2 partners exist per stimulus there — not new
information.) **Confirms Hugo's original suspicion directly**: GLMsingle cue
betas carry substantial information about the co-present stimulus — including
at minimum its category — independent of the frequency/value/repetition
structure this pipeline is meant to measure. **Not yet established**: whether
this explains the specific β(frequency)/β(value) RSA results (`rsa_roi_results.
ipynb`) — that needs a confound-regression test (open thread, below), not
attempted this session. Derivation: `run_rsa_partner_context.py`
(`find_category_split`/`category_distance`, new) + `rsa_partner_context_
results.ipynb` §2-3 (new, executed), SLURM jobs 5484846/5484905/5484906.

## Open threads (continued, findings 16-19)

10. ~~**Build the confound-regression test**~~ → **DONE** (new session,
    2026-09-03, `rsa_roi_results.ipynb` §10 — finding 20 below). Both β(value)
    and β(frequency) survive; findings 18-19 do NOT explain the headline RSA
    results.
11. Finding 19's category-leak result exists in BOTH `learning` and `test`
    (strong in both), so it doesn't by itself explain why the frequency/value
    RSA *collapses specifically in `test`* (open thread 8, findings 12-17) — if
    anything it's a confound present throughout, not one that changes between
    phases. **Now moot as an account of the collapse** — finding 20 shows the
    leakage doesn't drive β(frequency)/β(value) at all, so it isn't a candidate
    explanation for why they collapse in `test` either.
12. Finding 19's category-leak test used only each stimulus's two most
    numerous second-stimulus-category groups (up to 4 categories exist) — a
    full multi-way category discriminability measure (not just top-2) was not
    built.

## 20-21. Two independent tests of whether the stim-2 contamination (findings 18-19) explains the headline RSA findings — it doesn't

Plan: `/Users/hugofluhr/.claude/plans/sparkling-rolling-sparkle.md`. Branch
`stim2-contamination-tests`, commit `b43d27c`. Two tests per the plan:

### 20. Symmetric RSA regression (`run_rsa_roi.py --symmetric`) — β(value)/β(frequency) survive unchanged, stim-2 leakage is a separate additive signal

Added `s2_category`/`s2_frequency`/`s2_identity` predictors to the RDM
regression (new `profile_dist_rdm()` — stim-2 properties are per-condition
proportion PROFILES since each stim-1 condition pools multiple different
partners, unlike stim-1's per-condition scalars). Pre-flight VIF check
(`check_symmetric_vif.py`, BBT-only, no betas) found plausible VIFs (median
1.5-4) before submitting. Cluster job 5495880, n=58, ~1 min.

**β(frequency) survives essentially unchanged, even strengthens slightly**:
fusiform +0.373→+0.383 (both p<0.001), visual cortex +0.337→+0.358, wholebrain
+0.203→+0.272 — all still ***. **β(value) survives and strengthens (more
negative)**: fusiform −0.118→−0.194, visual cortex −0.080→−0.124 (both
p<0.001/<0.01). **s2_frequency is itself a real, independent, FDR-significant
predictor** in the same visual/fusiform territory (fusiform −0.214***, visual
cortex −0.223***, wholebrain −0.172**) — confirms the partner-context leakage
from findings 18-19 reaches the RDM-regression level, but it acts as a
*separate additive* signal, not an inflator/explainer of the headline effects.
s2_category weaker (significant only in fusiform, −0.199, FDR q=0.039);
s2_identity null everywhere after FDR. Derivation: `rsa_roi_results.ipynb`
§10 (new, executed) + its "9." addition to the top-level Findings summary.

### 21. Stim-2 category decoding (`run_stim2_decoding.py`) — existence-proof test, cluster job 5495879 still running at session-note-update time

4-class (face/hand/house/figure) LinearSVC, LOGO-CV, chance=0.25, with a
`run_s1cat_demeaned` control variant to rule out stim-1-pattern leakage as the
explanation. Results notebook (`stim2_decoding_results.ipynb`) built and ready
to execute; not yet run — job was at 8/59 subjects when this note was last
updated. **Not yet a finding** — see open thread 13 below.

## 22. Early/late within-run split: β(frequency) is already at full strength in the first half of `learning1` and does NOT grow with exposure — argues against a reinforcement-accumulation account

Direct test of Hugo's objection to finding 12 (flat β(frequency) learning1≈learning2,
collapse in test): if the effect reflects genuine habit accumulation from repeated
reward-linked choice, it should be weaker early and grow with exposure, including
*within* a run. Built `run_rsa_learning_dynamics.py` (new; splits each of
learning1/learning2 at the chronological median into early/late halves, each with
its own interleaved-CV crossnobis RDM; thin per-stimulus counts ~3-9/half handled
with a minimum-fold-count gate, same pattern as `run_rsa_partner_context.py`).
Cluster job 5518508, n=56 usable after the gate.

**β(frequency) is already significant in the FIRST HALF of `learning1`** — fusiform
+0.244\*\*\* (~first 12-24 trials of the whole experiment) — **and does not grow**:
fusiform +0.244\*\*\*→+0.186\*\*(1-late)→+0.252\*\*\*(2-early)→+0.169\*\*(2-late); visual
cortex +0.153\*\*→+0.096\*→+0.078(ns)→+0.072(ns), actually *weakening* over exposure —
the opposite of what habit accumulation predicts. β(value) shows the same flat
pattern. Reconciles with finding 8a (graded H-value already failed to beat the flat
design label) and finding 16 (pairing assignment fixed for the whole learning
phase, not revealed gradually): the geometry looks like it tracks something
established quickly (perceptual/associative registration of the fixed pairing
structure), not something built by repeated reinforcement. Doesn't resolve *why*
it then collapses in `test` (open thread 8 still open) but rules out "test
collapses because habit strength hadn't finished building" — it was already at
full early-`learning1` strength well before `test`. Derivation:
`rsa_roi_results.ipynb` §11 (new, executed).

## 23. Methodological detour: `stim2_decode`'s slowness was NOT a convergence problem — corrected mid-session, real bottleneck still under investigation

`stim2_decoding`'s first cluster attempt (job 5495879, NPROC=8) showed 0/59 subjects
complete after 15+ min; `.err` log had `ConvergenceWarning`s, initially (wrongly)
diagnosed as the cause and "fixed" via `tol=1e-3` + bumping NPROC to 24. A follow-up
isolated diagnostic (single dedicated CPU, sub-01, wholebrain) overturned that: timing
was IDENTICAL (~6s) across `max_iter` 200→2000, meaning the fit converges cleanly well
under 200 iterations — no real convergence problem. Checking history confirmed
`ConvergenceWarning` is pervasive in `run_decoding.py`'s and `run_frequency_decoding.py`'s
existing logs too (up to 409 warnings in one `frequency_decoding` job) without causing
comparable slowness there — a red herring. NPROC reverted to 8 (the established
convention; 24 concurrent heavy fits per node likely worsens contention, not helps).
Real bottleneck most likely node-level contention, not yet confirmed — a small
8-subject real-batch test (job 5518511) is running to establish actual throughput
before committing all 59 subjects again.

## Open threads (continued, findings 20-23)

13. **Run `stim2_decoding_results.ipynb` once the full 59-subject `stim2_decode` job
    completes** (superseded job IDs: 5495879 cancelled, 5518511 is an 8-subject
    throughput test only — resubmit all 59 once §23's throughput question is
    settled), sync `derivatives/stim2_decoding/` locally, execute the notebook, and
    fold the actual numbers into finding 21. Existence-proof framing predicts raw
    accuracy well above chance in visual/fusiform masks; the decisive part is
    whether `s1cat_demeaned` accuracy stays well above chance too.
14. **Diagnose the actual stim2_decode throughput bottleneck** (finding 23) — the
    8-subject test (job 5518511) will show real per-subject wall time; if still
    much slower than the ~30-60s/subject the isolated diagnostic implies, the next
    candidate is node-level contention (shared-node xargs-P fan-out vs. e.g. an
    array job spreading subjects across nodes) rather than anything in the code.
15. Commit the `stim2-contamination-tests` branch work still staged/uncommitted
    at session end (§10/§11 additions to `rsa_roi_results.ipynb`,
    `stim2_decoding_results.ipynb`, `run_rsa_learning_dynamics.py` +
    `submit_rsa_learning_dynamics.sh`, this note) and consider whether to merge
    into `main` or open a PR, once findings 14/21 above are resolved.

## 24. Stress-tested §9's permutation null against a design-constant confound it could have missed — survives, ruling out one more rival account

Prompted by "what the hell can the frequency effect actually be" after finding 22:
checked whether §9's permutation null (all C(6,3)=20 3-vs-3 relabelings) could be
secretly biased by a within-category EXEMPLAR-DISCRIMINABILITY constant — if every
subject's true label always put one +1/one -1 per category, the true label would
always get full credit for pure visual exemplar discriminability (present from
trial 1, no learning needed) while random permutations sometimes lose that
within-category contrast, producing significance for a reason having nothing to do
with reward/choice.

**Checked directly: not a constant.** Only 16/58 subjects have the literal
one-per-category split (counterbalancing rotation varies it, like the
image-frequency-assignment check). On exactly this n=16 subset — where the
exemplar-discriminability account could in principle explain everything — built the
sharpest possible test: a null restricted to the 8 category-preserving relabelings
(7 impostors), which share the identical within-category term and differ only in
cross-category structure. **Still rejects decisively: wholebrain p=0.0025, visual
cortex p=0.00006, fusiform p=0.00002.** A genuine attempt to break the strongest
existing evidence for "real frequency information," using the most conservative
null constructible from the data — it survived. Derivation: `rsa_roi_results.ipynb`
§9b (new, executed).

### Current best-supported account for the frequency effect (as of this session)

Ruled out: stim-2 leakage (finding 20), generic test-noise/category-general artifact
(finding 14, 17), learning-scope collinearity (finding 15), design-constant exemplar
discriminability (finding 24), gradual habit/reinforcement accumulation (finding 22,
8a). Leading remaining candidate: a **fast, non-accumulating tagging of each
stimulus by its structural role** in the pairing schedule (usually-chosen vs.
usually-passed-over), consistent with established value-driven-attention literature
(finding 9) — forms within a handful of trials, doesn't need or benefit from
repetition, sits specifically in visual/fusiform cortex (attentional prioritization
territory, not vmPFC/striatum reward-magnitude territory), and disappears in `test`
because the discriminating role-context (a dominant partner) is absent there. Still
unexplained: why the neural signature vanishes rather than merely weakens in `test`,
given the behavioral habit effect Hugo confirms is still present there (open thread 8).
