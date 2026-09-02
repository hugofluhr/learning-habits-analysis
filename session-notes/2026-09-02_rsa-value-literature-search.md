# Session log — literature search: RSA on value in binary-choice tasks

**Date:** 2026-09-02
**Companion notes:**
[2026-08-27_rsa-first-real-results.md](2026-08-27_rsa-first-real-results.md),
[2026-08-30_rsa-confound-controls-and-validation.md](2026-08-30_rsa-confound-controls-and-validation.md)
— the value×frequency interaction this search was trying to contextualize.

Started with Hugo asking whether dropping the frequency RDM from the RSA model would be
valid, given surprise at the null value effect (answered from existing notebook findings,
no new code — see §8c.1 in `rsa_roi_results.ipynb`, already checkpointed). Moved into a
literature search via the Scite MCP server for prior RSA work on value in binary-choice
tasks, to check whether the value×frequency sign-flip has a precedent or framing in the
literature. Pure literature review — no computation, nothing to rescue into code.

---

## Findings

### 1. No prior paper found doing RSA-regression-with-frequency-as-co-regressor on binary value choice

Four targeted Scite searches (`representational similarity analysis subjective value
binary choice fMRI`; `multivoxel pattern representational similarity reward value
orbitofrontal cortex choice`; `neural representational geometry economic decision making
value distance code`; `representational similarity analysis value-based decision making
two-alternative forced choice human fMRI ventromedial prefrontal`) surfaced no paper using
RSA/RDM regression with an explicit choice-history/frequency confound term for value in a
paired-stimulus binary-choice design. Most direct hits are monkey electrophysiology
(population-geometry framing, not RDM-based) or human fMRI value studies that are
univariate/parametric rather than RSA. This may be a genuine gap in the literature rather
than a search-coverage failure, but only 4 queries were run — not exhaustive. Full hit
lists (10 results/query, titles+DOIs+authors extracted via `jq`/`python3 json.load`) are
in the tool-result files under this session's `tool-results/` cache, not persisted in the
repo.

### 2. Closest human-fMRI multivariate/value matches

- Yan et al. (2016), *Multivariate Neural Representations of Value during Reward
  Anticipation and Consummation in the Human OFC*, Sci Rep,
  [10.1038/srep29079](https://doi.org/10.1038/srep29079) — multivariate value pattern in
  human OFC, closest methodological match found.
- Lee, Yu, Lerman et al. (2021), *Subjective value, not a gridlike code, describes neural
  activity in ventromedial prefrontal cortex*, NeuroImage,
  [10.1016/j.neuroimage.2021.118159](https://doi.org/10.1016/j.neuroimage.2021.118159) —
  direct model-comparison design (value vs. a competing geometric code), structurally
  similar to our value-vs-frequency competition.
- Veselič et al. (2025), *A cognitive map for value-guided choice in vmPFC*, Cell,
  [10.1016/j.cell.2025.03.038](https://doi.org/10.1016/j.cell.2025.03.038), and Orloff et
  al. (2026, preprint), *A cognitive map of subjective value space for human risky choice*,
  [10.64898/2026.05.19.726239](https://doi.org/10.64898/2026.05.19.726239) — both frame
  value as a 2D relational/positional code rather than a single scalar; potential reframing
  for the value×frequency interaction (finding 5 in the 2026-08-27 note) as a 2D geometry
  rather than two competing 1D regressors.

### 3. Mechanistic candidate for the sign-flip: divisive normalization by choice history

Louie & Glimcher (2012, [10.1111/j.1749-6632.2012.06496.x](https://doi.org/10.1111/j.1749-6632.2012.06496.x))
and Louie, Khaw & Glimcher (2013, [10.1073/pnas.1217854110](https://doi.org/10.1073/pnas.1217854110))
give a normalization account of value coding relative to recent choice/reward context —
offered as a candidate explanation for *why* choice-frequency would sign-flip a value
effect (validated interaction, finding 6 in the 2026-08-30 note), not yet checked against
our data in any way.

---

## Code shipped

None — literature-only session, no notebook/script changes.

---

## Git state at session end

Branch `main` (from the top-level `git status` at session start), clean before this note.
This checkpoint stages only the new session note.

---

## Open threads

1. **Confirm the literature gap with a broader search** before claiming novelty in any
   write-up — only 4 Scite queries run, not a systematic review (e.g. add "choice history
   confound," "repetition suppression value fMRI," "reinforcement learning frequency
   multivariate pattern").
2. **Consider reframing the value×frequency interaction (2026-08-27 finding 5 / 2026-08-30
   finding 6) as a 2D relational/positional code**, following Veselič (2025) / Orloff
   (2026), rather than as two competing 1D RDM regressors — would need a new model-RDM
   construction, not just relabeling.
3. **Check the normalization account (Louie & Glimcher)** against the existing data: does
   the sign of the value effect track a normalization-by-recent-choice-rate prediction, or
   is that just a plausible-sounding mechanism with no specific testable prediction here?
