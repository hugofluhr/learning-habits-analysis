# Session log — RSA confound controls, §8c interaction validated, FDR + VIF

**Date:** 2026-08-30
**Companion note:** [2026-08-27_rsa-first-real-results.md](2026-08-27_rsa-first-real-results.md)

Started as a methodological audit of the RSA choice-frequency result (β≈+0.35 in visual
cortex). Three parallel reviewers identified the **beta-window confound** as the top
concern: GLMsingle betas capture the entire trial (first stim → second stim → response in
~1.44s, all within one TR). Implemented two confound-control RDMs, FDR correction, and a
VIF diagnostic. **The frequency effect survives all controls.**

---

## Findings

### 1. Frequency survives second-stimulus and choice-rate confound controls

After adding `second_stim_value` and `choice_rate` RDMs to the regression (5 predictors
total), the core frequency betas barely move:
- Visual cortex: β=+0.337, p<0.001, FDR q<0.001
- Fusiform: β=+0.373, p<0.001, FDR q<0.001
- Whole brain: β=+0.203, p<0.001, FDR q<0.001

Neither confound predictor absorbs the frequency effect. Derivation:
`rsa_roi_results.ipynb` §3 bar chart + §3a FDR table.

### 2. Second-stim-value is independently significant only in fusiform

Fusiform β(second_stim_value) = +0.159, p=0.0002, FDR q=0.002. No other ROI shows a
significant second-stim effect. This means fusiform geometry partly reflects the
paired-alternative's value — but the frequency effect there (β=+0.373) is more than
double and independent. Derivation: `rsa_roi_results.ipynb` §3a FDR table.

### 3. Choice_rate is significant only in visual cortex — with opposite sign

Visual cortex β(choice_rate) = −0.120, p=0.007, FDR q=0.047. The negative sign means
stimuli with *more different* choice rates are *more similar* in neural pattern — the
opposite of what would be needed to explain the frequency effect. Derivation: same FDR
table.

### 4. Model matrix is well-conditioned (VIF < 5 for all predictors)

Condition number κ: median=2.7, max=4.1 across 58 subjects. VIF all below 4.35 (median
≤2.0). Adding 2 confound predictors (5 total for 15 data points) does not destabilise the
regression. Derivation: `rsa_roi_results.ipynb` §1b VIF diagnostic cell.

### 5. FDR correction: 8 of 54 tests survive at q=0.05

From 9 ROIs × (5 model terms + 1 contrast) = 54 tests. Survivors: frequency in VC,
fusiform, whole brain; value in fusiform; second_stim_value in fusiform; choice_rate in
VC; contrast_value in VC and fusiform. Marginal effects (striatum frequency p=0.013,
vmPFC value p=0.015) do not survive. Derivation: `rsa_roi_results.ipynb` §3a.

### 6. §8c value × frequency interaction VALIDATED

The interaction from the previous session (value predicts distance positively within
same-frequency pairs, negatively within different-frequency pairs) survives all three
controls:
- **Shuffled**: collapses to ~0 (VC: orig=+1.23 vs shuf=+0.08, paired-t p<0.0001;
  fusiform: +1.36 vs −0.12, p<0.0001)
- **Blocked**: reproduces identically (same numbers — expected since blocked only
  changes within-run fold splitting, not the pooled-scope RDM)
- **Remove-mean**: slightly strengthens (fusiform +1.36 → +1.38)

Derivation: `rsa_roi_results.ipynb` §8c-validation cell.

---

## Code shipped

| File | Change | Git state |
|------|--------|-----------|
| `multivariate/run_rsa_roi.py` | Added `second_stim_value` and `choice_rate` confound-control RDMs | merged to rsa-roi (618a6c1) |
| `multivariate/rsa_roi_results.ipynb` | §1b VIF, §3a FDR, 5-term bar charts, §8c-validation, Findings rewrite | merged to rsa-roi (248dcbb) |

## Data produced

- 4 RSA variant jobs (rsa, rsa_shuffled, rsa_remove_mean, rsa_blocked) re-run on the
  cluster with the 5-term regression. 58/58 subjects complete per tree. Results synced
  locally to `derivatives/rsa*/`.
- 4D singleton mask fix committed (0c27a60) and verified on cluster.

## Git state

Branch: `rsa-roi`, pushed to origin. Local and remote agree at `248dcbb`.

---

## Open threads

1. ~~§8c interaction validation~~ → **DONE** (finding 6 above).
2. **FDR-correct the §8c interaction** jointly with the §3a tests — currently the
   interaction is tested via paired t-test but not BH-corrected in the same family.
3. **Choice_rate negative effect in VC** — survives FDR (q=0.047); worth interpreting.
4. **vmPFC and striatum effects** — do not survive FDR. Targeted hypotheses needed.
5. **Per-run §8c interaction** — untested (learning dynamics of the value×frequency effect).
6. **Branch not merged to main.**
