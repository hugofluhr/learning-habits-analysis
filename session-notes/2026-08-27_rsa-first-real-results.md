# Session log — RSA pipeline runs for real: choice-frequency geometry established, value coding reopened

**Date:** 2026-08-27
**Companion note:** [2026-08-26_rsa-design-and-roi-pipeline.md](2026-08-26_rsa-design-and-roi-pipeline.md)
— the pipeline, design checks, and crossnobis validation this session actually ran.

Started as git cleanup (merging a stray worktree branch, tracking the presentation deck)
and finishing the RSA pipeline's remaining steps (env rebuild, dry-run check, cluster
submission). The first real run surfaced a headline choice-frequency effect and a null
for value — then a terminology correction from Hugo (the "frequency" manipulation is
*choice* frequency, not presentation frequency) triggered a deeper follow-up that
reopened the value-coding question via an unplanned but robust-looking interaction.
**That interaction is preliminary and explicitly not validated yet — see Open thread 1.**

---

## Findings

### 1. `sub-46` confirmed still missing from the BBT — real n is 58, not 59

`run_rsa_roi.py --dry-run` swept all of `participants_mvpa.tsv` against `bbt.csv`: 58
pass, `sub-46` fails on BBT presence, the same gap flagged during the qvalue work and
never resolved. No other subject is malformed. Derivation: `run_rsa_roi.py
check_bbt_only()`, run via `--dry-run`.

### 2. The "frequency" manipulation is choice frequency, not presentation frequency

Corrected mid-session after Hugo flagged it: all stimuli are shown equally often. First-
stim presentation counts are identical at 84 per stimulus for the ±1 label (zero
variance) — not just non-significantly different, exactly equal. What differs is how
often the subject *chose* the stimulus (38.3 vs 44.0 mean chosen-count, r=+0.224,
p=8×10⁻⁸), manipulated by selectively pairing it with higher-/lower-valued alternatives.
Every subsequent finding below uses "choice-frequency"; earlier same-day language calling
this "exposure" or invoking repetition suppression was wrong. Derivation:
`rsa_design_checks.ipynb` §1b (new cell this session).

### 3. Choice-frequency is robustly represented — the best-established result

β(frequency): wholebrain +0.220, visual cortex +0.351, fusiform +0.378 (all p<0.001,
n=58); shuffled control ~0 everywhere. Survives the amplitude-confound control: grows
slightly under `--remove-mean` (visual cortex +0.390, fusiform +0.425, both p<0.001,
job 5331516/rerun), and the direct amplitude~choice-frequency correlation is negligible
(visual cortex r=−0.066, p=0.004) — ruling out a global choice-related gain artifact, not
(as first framed) repetition suppression, which never applied given equal presentations.
Derivation: `rsa_roi_results.ipynb` §3, §7.

### 4. Added a graded choice-kernel (H-value) model variant — inconclusive, not disconfirming

`model='ck'` swaps the categorical ±1 label for the graded per-subject H-value
(`first_stim_value_ck`, `|ΔH|`) in the same regressor slot. Outcome: `|ΔH|` gives a
sign-flipped, worse-fitting result than the label (paired r-comparison p<0.0001 in
wholebrain/visual/fusiform) — but a diagnostic shows `|ΔH|` barely reconstructs the
label's own pairwise structure (corr(|Δlabel|,|ΔH|) ≈ 0, highly variable across subjects)
despite per-stimulus H correlating positively with the label (r≈+0.32). Tried a
median-split "H-bin" as a fix for what looked like a magnitude-noise problem — doesn't
help (same ≈0 pairwise reconstruction despite 67% item-level label agreement), so the
noise is relational, not a scale artifact binning could fix. Read as "this
operationalisation of H doesn't work," not as evidence against habit-strength coding.
Derivation: `rsa_roi_results.ipynb` §8a.

### 5. ⚠️ PRELIMINARY — value predicts distance, with opposite signs depending on choice-frequency match

**Terminology correction (2026-09-03):** "β(value)≈0" below is imprecise — the pooled
joint-model value effect is actually small and **significantly negative**, not zero
(ROI-level, non-figure, pooled: fusiform β=−0.118 p=0.0007, visual cortex β=−0.080
p=0.023). It's negative rather than exactly zero because different-frequency pairs
(9/subject, negative slope) outnumber same-frequency pairs (6/subject, positive slope) in
the non-partitioned regression. See
`session-notes/2026-09-03_rsa-searchlight-cluster-fwe-and-roi-method.md`.

Restricting to non-figure pairs that share the same choice-frequency label (frequency
held exactly constant, not regressed): value predicts distance **positively**
(wholebrain +0.426\*\*\*, visual cortex +0.651\*\*\*, fusiform +0.869\*\*\*, n=58, §8b).
Restricting instead to *different*-frequency pairs: value predicts distance
**negatively** (−0.239\* to −0.579\*\*\*, §8c.2). The interaction (same-minus-different)
is large and highly significant (+0.665 to +1.357, all p<0.001, wholebrain/visual/
fusiform). This resolves why the joint regression (`category+value+frequency`) finds
β(value)≈0 — two opposite-signed effects averaging out — and rules out simple
omitted-variable bias as the explanation: dropping frequency from the regression entirely
pushes β(value) *further negative* (visual cortex −0.149\*\*\*, fusiform −0.143\*\*\*,
§8c.1), not toward finding 5's positive same-frequency sign. **This is computed once, not
checked against the shuffled/blocked/remove-mean controls already on disk, and not
multiple-comparison corrected. Do not treat "no value coding" as settled, and do not cite
these interaction numbers outside `rsa_roi_results.ipynb` §8c until validated — top open
thread below.** Derivation: `rsa_roi_results.ipynb` §8b, §8c.

### 6. Sanity check reproduces the design-checks prediction exactly

On the full 8-stimulus set (`subset='all'`), category/value/frequency are all strongly
significant and near-identical in magnitude (β≈0.15–0.28, p<0.001, wholebrain/visual/
fusiform) — because there they're all proxies for the same figure-vs-rest split, as
`rsa_design_checks.ipynb` predicted from the BBT alone before any beta was touched.
Confirms the non-figure subset is the right primary readout for everything above.
Derivation: `rsa_roi_results.ipynb` §3.

### 7. The raw same-value contrast reads differently now that finding 5 exists

Same-value pairs carry the maximal |Δchoice-frequency| = 2.0 by design, so finding 3's
effect mechanically drags the non-partialled same-value contrast negative
(−0.202 to −0.637, p<0.011, wholebrain/visual/fusiform) regardless of any value effect.
Previously stated as "trust the regression instead" — given finding 5, neither number is
settled until the §8c validation pass runs. Derivation: `rsa_roi_results.ipynb` §4.

### 8. Learning-dynamics sign flip in `test` — failed robustness, discarded

Same-value contrast in fusiform: −0.63/−0.59 (learning1/learning2, both p<0.001) flipped
to +0.19 (p<0.01) in `test` under the interleaved split — collapses under
`--within-run-split blocked` (job 5331517/rerun): fusiform +0.074 ns, visual cortex
+0.109→−0.041. Exactly the low-SNR fragility `crossnobis_validation.ipynb` §4 predicted
for per-run RDMs. The learning-run negative contrasts DO reproduce across both splits
(fusiform −0.60/−0.53 blocked, p<0.001). Not re-examined in light of finding 5; the
per-run value×frequency interaction is untested. Derivation: `rsa_roi_results.ipynb` §5, §7.

### 9. Shuffled-label control is clean for everything checked so far

5 masks × 4 terms: real vs. shuffled diverges only where findings 3/6/7 predict; shuffled
shows ~2/20 nominal hits, consistent with uncorrected α=0.05 chance. **Not yet run on the
finding-5 interaction specifically.** Derivation: `rsa_roi_results.ipynb` §6.

---

## Code shipped

| What | State |
|---|---|
| `--dry-run` flag on `run_rsa_roi.py` (BBT-only precondition check) | branch `rsa-roi` (`03e2d6d`), pushed |
| `--remove-mean` flag + per-condition amplitude CSV | branch `rsa-roi` (`b00d116`), pushed |
| `rsa_roi_results.ipynb`, first executed pass (n=58, §1–§7) | branch `rsa-roi` (`43583c9`), pushed |
| Choice-frequency terminology correction + `ck` (H-value) model variant on `run_rsa_roi.py`; §1b presentation-balance check on `rsa_design_checks.ipynb` | branch `rsa-roi` (`b15b1eb`), pushed |
| `rsa_roi_results.ipynb` §8a/§8b/§8c + full terminology sweep + Findings rewrite | branch `rsa-roi`, **this checkpoint** |
| Merged `worktree-cluster-only-defaults` → `main` (ff, `cfcc073`) | `main`, pushed |
| Tracked `multivariate/presentation.md` + assets on `main` | `main` (`08b3701`), pushed |
| Merged `main` → `rsa-roi` (`e869e8a`) | `rsa-roi`, pushed |
| Deleted branch/worktree `worktree-cluster-only-defaults` | done |

`CLAUDE.md` (via the merged branch) now formally documents the cluster-only compute
policy — VM off-limits, correct cluster paths, `srun`/`sbatch` requirement.

---

## Data produced (cluster)

- **Env rebuild** (job `5320542`): `rsatoolbox==0.2.0` installed. Confirmed via
  `--validate-against-rsatoolbox` on `sub-01` (max |dev| = 8.24e-18 against real ROI data).
- **First pass, `model` ∈ {objective, rl}** (jobs `5331366` real / `5331395` shuffled /
  `5331516` remove-mean / `5331517` blocked): 58/58 each, 80 rows/subject.
- **Rerun with `ck` variant added** (jobs `5332740` real / `5332741` shuffled /
  `5332742` remove-mean / `5332743` blocked, `OVERWRITE=1`): 58/58 each, 120 rows/subject,
  `model` ∈ {objective, rl, ck} confirmed in every tree, `ck_value_*` keys confirmed in
  the model_rdms npz.
- All four trees synced to
  `dev_sample/bids_dataset/derivatives/{rsa,rsa_shuffled,rsa_remove_mean,rsa_blocked}/`
  (results.csv + RDM .npy + model_rdms.npz + amplitude.csv only — not the source betas)
  via `rsync` from `uzh.cluster.cmd`.

---

## Git state at session end

Branch `rsa-roi`, pushed through `b15b1eb`; this checkpoint's notebook/session-note
changes are staged on top, not yet committed (commit happens right after this note).
`main` is at `08b3701` (cluster-only-defaults + presentation deck), merged into `rsa-roi`
at `e869e8a`.

---

## Open threads — in priority order

1. **Validate finding 5 before trusting it at all.** Rerun the same-frequency vs
   different-frequency value-slope comparison (§8c.2) against `derivatives/rsa_shuffled`
   (both subsets should collapse to ~0 if the interaction is an artifact),
   `derivatives/rsa_blocked` (does the interaction reproduce under the alternative
   within-run split, extending to the per-run scopes too), and
   `derivatives/rsa_remove_mean` (rule out an amplitude-driven account of the
   interaction specifically, not just of the main choice-frequency effect). FDR-correct
   across the ~20 terms already in the regression table plus the new interaction tests.
   **This is the next session's main task**, not a quick addition.
2. If finding 5 survives validation, propagate the revision: this notebook's Findings,
   `project_rsa_roi.md` in memory, and any external summary — none should say "no value
   coding" as settled until then.
3. vmPFC β(value) (p≈0.03, stable across variants) and striatum β(frequency) (p≈0.02):
   both singles among many uncorrected comparisons even before finding 5's additions.
4. §8a's H-value proxy: consider whether a better-fitting CK model (checked against
   behavioural log-likelihood) or a different summary than pooled-mean H would help,
   before concluding graded habit strength can't be tested via RSA at all.
5. Searchlight RSA out of scope unless the validated results warrant it.
