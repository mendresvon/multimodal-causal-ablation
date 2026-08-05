# Comprehensive Quadruple-Check Audit Report
## Section VI-D: Neuron-Level Causal Validation of Modality Competition

**Date:** 2026-08-06  
**Ground Truth:** Prof. KC Lan's 3-Week Experiment Protocol (`experiment_protocol[From Prof KC Lan].docx`)  
**Artifacts Reviewed:** All 39 notebook cells, 12 result files, 2 ADRs, 15 journal entries, `e2e.py` model architecture  
**Verdict:** 🔴 **NOT PUBLICATION-READY** — 7 critical/high issues remain unresolved that would cause peer-review rejection

---

## Executive Summary

The notebook has undergone significant remediation since the original Aug 4 master audit (16 issues). Several critical bugs were fixed (signed ratio formula B.1 eliminated, L1 probe weights adopted, full-dataset extraction implemented, Day 13 cosine similarity added, Phase D expanded to all 6 classes). However, deep scrutiny reveals **7 remaining issues** — including 3 that individually could sink a peer review. The most damaging: the causal ablation results show functionally zero effect sizes across every condition, and the paper's interpretation of this as "multimodal redundancy" rests on a methodology that never tested the dominant modality's own neurons.

---

## PART 1: Protocol Compliance Checklist

| Protocol Step | Status | Detail |
|:---|:---:|:---|
| **Day 0.1:** Confirm both checkpoints | ✅ | Base + fine-tuned loaded and verified |
| **Day 0.2:** Identify dominant modality via SHAP | ⚠️ | **See Issue #1** — Dominant modality changed mid-experiment |
| **Day 0.3:** Pick layer to hook | ⚠️ | **See Issue #2** — Ablating Audio FFN but probing Audio FFN too |
| **Day 0.4:** Confirm feature dimensionality | ✅ | 64-d Audio FFN confirmed |
| **Day 1-2:** Extract activations over full RML | ✅ | 720 samples extracted (Cell 16, `base_full_1152.npy`) |
| **Day 3:** L1-penalized probes, rank by \|coef\| | ✅ | Fixed in Cell 30 — ranks by `np.abs(probe.coef_[0])` |
| **Day 4:** Sanity-check probes (AUC) | ✅ | Base 0.85, FT 0.84 Mean AUC — well above 0.65 |
| **Day 5:** Top-5 table with weights | ✅ | `phase_b_top5_probe_weights.csv` has indices + L1 weights |
| **Day 6-7:** Mean-ablation (not zero) | ⚠️ | **See Issue #3** — Mean computed from training set, but applied to test features from wrong model |
| **Day 8:** Ablate base model, k={1,3,5,10} | ⚠️ | **See Issue #4** — Evaluated on test set only (144 samples), not full eval set |
| **Day 9:** Repeat for fine-tuned model | ⚠️ | Same issue as Day 8 |
| **Day 10:** Buffer/compile | ✅ | CSVs and JSONs saved |
| **Day 11-12:** Selectivity vectors | ✅ | Used L1 probe weights consistently |
| **Day 13:** Cosine similarity | ✅ | Implemented in Cell 36, `phase_d_cosine_similarity.csv` |
| **Day 14:** Ablation transfer (cross-ablation) | ⚠️ | **See Issue #5** — R denominator is zero for all classes |

---

## PART 2: Critical & High Issues

### Issue #1 (CRITICAL): Dominant Modality Contradiction — SHAP Says Audio, But Early Phases Targeted Text

**The fundamental confusion at the heart of this experiment.**

The experiment history reveals a tortuous path:
1. **July 31 (Phase A):** SHAP was computed using `np.abs(np.sum(shap[:, start:end], axis=1))` — summing first, then taking absolute value. Result: **Text dominant at ~70%**.
2. **Aug 4 audit:** Discovered this was wrong, identified 16 issues, started remediation.
3. **Aug 5 session:** Switched target to **Audio (64-d)**. Correctly re-sliced `base_acts_tier1 = base_full_1152[:, 1088:1152]`.
4. **Current result files:** `phase_a_dominant_modality_verdict.csv` now reports **Audio dominant** (Base: 50.48%, FT: 58.51%).

**But the Phase A SHAP aggregation code (Cell 10) still uses the flawed formula:**
```python
per_sample = np.abs(np.sum(sv[:, start:end], axis=1))
```

This computes `|sum(SHAP)|` — which measures the **net signed attribution** of a modality. The correct metric for *contribution magnitude* is `sum(|SHAP|)`:
```python
per_sample = np.sum(np.abs(sv[:, start:end]), axis=1)
```

**Why this matters:** With `|sum(SHAP)|`, SHAP values within a modality cancel each other out. A modality where half the features have positive SHAP and half have negative SHAP (common in high-dimensional representations like Text's 1024-d) would appear to contribute nearly zero, even if each individual feature is highly important. Audio's 64-d representation suffers less cancellation simply because it has fewer features. The current aggregation method **systematically biases toward lower-dimensional modalities**.

**Impact on results:** The dominant modality verdict (Audio vs Text) may be an artifact of the aggregation formula. If the correct `sum(|SHAP|)` formula restores Text as dominant, then the entire experiment targeted the wrong modality branch — and the near-zero ablation drops would be trivially explained (you were ablating the secondary modality, not the dominant one).

> [!CAUTION]
> This single issue potentially invalidates the entire experimental pipeline from Phase A onward. A reviewer who checks the SHAP aggregation formula will catch this immediately.

---

### Issue #2 (CRITICAL): Phase C Evaluates Test Set Only (144 Samples) While Computing Mean from Train Set (518 Samples) — Creating an Inconsistent Evaluation Regime

**What the protocol says (Day 8):**
> *"Ablate them, run the full eval set through the patched model, record per-class accuracy"*

**What the code does (Cell 34, `run_sweep`):**
```python
base_df, base_sweep = run_sweep(
    base_model,
    base_1152[576:720],       # <-- Only test set (144 samples)
    base_acts_tier1,           # <-- Full 720 samples for mean computation
    labels_tier1[576:720],     # <-- Only test set labels
    base_top_neurons,
    'base',
)
```

And inside `run_sweep`:
```python
dataset_mean_acts = np.mean(acts_tier1[0:518], axis=0)  # Train set mean
```

**The problem is threefold:**

1. **Only 144 evaluation samples.** With 6 classes, that's ~24 samples per class. A single prediction flip = 4.17% accuracy change. The minimum detectable effect size is enormous. The protocol says *"run the full eval set"* — which in context means all 720 samples.

2. **The mean clamp vector is computed from training set indices `[0:518]` of the Audio Tier 1 activations.** This is correct per the protocol (*"computed once, upfront, over the training set"*). ✅

3. **But the evaluation itself only uses the test set.** The protocol intends for the accuracy evaluation to be on a large, representative sample. With 24 samples per class, statistical power is negligible.

**Impact:** Every ablation drop in the results is either 0.00% or 4.17% (1 sample). This granularity is too coarse to detect real causal effects. A reviewer would correctly note that the experiment is severely underpowered.

**The counterargument (from journal 2026-08-05-session2)** was that evaluating on training samples is invalid because the model has memorized them. This is a legitimate concern — but the solution is not to evaluate on only 144 samples. The correct approach is:
- Evaluate on all 720 samples but **report train and test accuracy separately**
- Or perform k-fold cross-validation of the ablation evaluation
- Or at minimum, acknowledge the severe statistical limitation

---

### Issue #3 (CRITICAL): Mean Clamp Vector Uses Base Model's Training Mean for Fine-Tuned Model Ablation

**Cell 34 (`run_sweep`):**
```python
dataset_mean_acts = np.mean(acts_tier1[0:518], axis=0)
```

When calling `run_sweep` for the fine-tuned model:
```python
ft_df, ft_sweep = run_sweep(
    ft_model,
    ft_1152[576:720],
    ft_acts_tier1,          # <-- This is FT activations
    labels_tier1[576:720],
    ft_top_neurons,
    'finetuned',
)
```

Inside `run_sweep`, `acts_tier1` refers to the FT's Tier 1 activations. So `dataset_mean_acts = np.mean(ft_acts_tier1[0:518], axis=0)` — this correctly uses the FT model's own training-set mean. ✅

**But in Phase D (Cell 39):**
```python
ft_dataset_mean_acts = np.mean(ft_acts_tier1[0:518], axis=0)
```

This computes the FT model's training-set mean and uses it when ablating base model neurons inside the FT model. **This is correct** — when you ablate inside the FT model, you should use the FT model's distribution-preserving mean. ✅

**Verdict:** The mean clamp vectors are correctly computed from the respective model's training set. This issue from the original audit has been resolved.

---

### Issue #4 (HIGH): The Ablation Results Show Zero Effect — Is This a Genuine Finding or a Power/Methodology Failure?

**The central empirical question.** The results show:

| Model | Classes with 0.00% target drop | Only non-zero |
|:---|:---|:---|
| Base | 6/6 classes = 0.00% | None |
| Fine-tuned | 5/6 classes = 0.00% | Anger: 4.17% (1 sample) |

**The journal (2026-08-06) interprets this as:**
> *"high SHAP attribution and high linear discriminability do NOT guarantee causal fragility under discrete ablation in multimodal transformers. Modality redundancy acts as a powerful error-correction mechanism."*

**This interpretation may be correct, but it's not adequately supported by the experimental design.** Here's why:

**A) Statistical power problem:**
- With ~24 test samples per class, the experiment can only detect effects that flip at least 1 out of 24 predictions (4.17%).
- A genuine causal effect of 2-3% would be invisible at this sample size.
- The confidence interval around 0.00% with n=24 is approximately ±8% (exact binomial).
- A reviewer would note that failing to detect an effect is not the same as proving absence of effect.

**B) k=5 out of 64 may be too few neurons:**
- Ablating 5 out of 64 neurons removes only 7.8% of the representation dimensions.
- The model has 3 modalities (Text 1024-d, Video 64-d, Audio 64-d) feeding into a learned weighted fusion.
- Even if all 5 ablated Audio neurons were perfectly class-selective, the remaining 59 Audio neurons + 1024 Text neurons + 64 Video neurons provide massive redundancy.
- The k=10 sweep also shows near-zero drops, which partially addresses this — but k=10 is still only 15.6% of dimensions.

**C) The "correct" interpretation requires a control experiment:**
- To claim "multimodal redundancy causes the near-zero drops," you need to show that ablating a *larger* fraction of Audio neurons (e.g., k=32 or k=64) *does* cause drops.
- Without this control, a reviewer could equally argue: "The Audio neurons simply aren't important, regardless of what SHAP says."

> [!WARNING]
> The paper's discussion cannot claim "multimodal redundancy" without a dose-response control (ablating larger k values or full modality knockout).

---

### Issue #5 (HIGH): Transfer Retention Ratio R is Undefined (0/0) for All Classes

**From `phase_d_transfer_retention.csv`:**

| Class | base_target_drop | ft_target_drop | R |
|:---|:---|:---|:---|
| anger | 0.0 | 4.17 | 0.0 |
| disgust | 0.0 | 0.0 | 0.0 |
| fear | 0.0 | 0.0 | 0.0 |
| happiness | 0.0 | 0.0 | 0.0 |
| sadness | 0.0 | 0.0 | 0.0 |
| surprise | 0.0 | 0.0 | 0.0 |

**The problem:** R = ft_drop / base_drop. When base_drop = 0.0, R is mathematically undefined (0/0). The code handles this by setting R = 0.0, which then triggers the "Substrate Dispersion" classification (R < 0.30). But this classification is **meaningless** — you can't compute a retention ratio when the numerator is undefined.

The protocol (Day 14) says:
> *"If fine-tuned accuracy on class c also drops a lot, the neurons are still functionally load-bearing after fine-tuning (preserved). If the drop is small — even though those neurons mattered a lot in the base model — the function has moved elsewhere."*

The protocol **presupposes** that the base model's neurons DO matter ("even though those neurons mattered a lot in the base model"). When base_drop = 0.0, the precondition of the analysis is violated — the neurons don't matter in the base model either. You cannot classify a substrate outcome when the base ablation produced no effect.

**Impact:** The entire Phase D Transfer Retention taxonomy (Preservation/Reassignment/Dispersion) is inapplicable. All 6 "Substrate Dispersion" labels in the results are a forced classification from an undefined ratio. A reviewer will immediately see that `R = 0/0 → 0.0` is not a legitimate result.

---

### Issue #6 (HIGH): Cosine Similarity Results Contradict the Ablation Results

**From `phase_d_cosine_similarity.csv`:**

| Class | Cosine Similarity | Interpretation |
|:---|:---|:---|
| anger | 0.9048 | High Sharpening |
| disgust | 0.7491 | High Sharpening |
| fear | 0.9832 | High Sharpening |
| happiness | 0.7089 | High Sharpening |
| sadness | 0.9570 | High Sharpening |
| surprise | 0.9394 | High Sharpening |

The cosine similarities are uniformly high (0.71–0.98), which per the protocol means:
> *"High similarity → fine-tuning sharpened the same existing representation"*

This suggests **Substrate Preservation** — the selectivity vectors barely rotated. But the Phase D ablation transfer says **Substrate Dispersion** for 5/6 classes. These two results directly contradict each other.

The protocol anticipates this:
> *"Day 14 should be your headline result if the two disagree."*

But the protocol's guidance assumes Day 14 produces meaningful results (non-zero base drops). Since the ablation drops are all zero, the "headline result" is vacuous. You can't use it to override Day 13.

**For the paper:** This contradiction needs explicit discussion. The honest interpretation is: "The selectivity vectors are geometrically preserved (high cosine similarity), but our ablation methodology lacked sufficient statistical power to detect functional preservation at the neuron level."

---

### Issue #7 (HIGH): Phase B Probes Use Full 720-Sample Dataset But Phase C Evaluates Only Test Set — Train/Test Contamination in Neuron Ranking

**Cell 27 (Phase B probes):**
```python
base_probes, base_valid = validate_probe_signal(
    base_acts_tier1, labels_tier1, 'base'
)
```

Here `base_acts_tier1` has 720 samples (full dataset). The L1 probes are fitted on all 720 samples (with 5-fold CV for AUC evaluation, then a final `pipe.fit(acts, y_binary)` on all 720 samples to retain coefficients).

**Cell 34 (Phase C ablation evaluation):**
```python
base_df, base_sweep = run_sweep(
    base_model, base_1152[576:720], base_acts_tier1,
    labels_tier1[576:720], base_top_neurons, 'base',
)
```

The neuron rankings (`base_top_neurons`) were derived from L1 probes fitted on **all 720 samples including the 144 test samples** that are now being used for evaluation. This means the neuron selection was informed by the test set — a subtle form of data leakage.

**The protocol's position:**
> *"train+test combined is fine here — you're not training anything, just observing"*

This is true for **activation extraction** (Day 1-2). But for **probe fitting** (Day 3), the protocol doesn't explicitly address this. The probes ARE being trained. Using all 720 samples to fit the probe, then evaluating ablation accuracy on a subset of those same 720 samples, means the neuron rankings are optimistically biased toward neurons that separate the test set.

**Severity:** In practice, with L1 regularization and 720 samples across 64 dimensions, this leakage is likely small. But it's a methodological impurity that a rigorous reviewer would flag.

**Fix:** Fit probes on training set only (indices 0:518), then evaluate ablation on test set (576:720). The validation set (518:576) can be used for probe hyperparameter tuning if needed.

---

## PART 3: Issues From Original Audit — Resolution Status

| Original Issue | Status | Notes |
|:---|:---:|:---|
| A.1: 144 vs 720 samples | ⚠️ Partial | Extraction done (720), but evaluation still uses 144 test |
| A.2: Activation ratio vs L1 weights | ✅ Fixed | Cell 30 uses `np.abs(probe.coef_[0])` |
| A.3: Day 13 cosine similarity | ✅ Fixed | Cell 36 implemented |
| A.4: Mean clamp from test set | ✅ Fixed | Now uses `acts_tier1[0:518]` (training set) |
| A.5: Missing probe weight values | ✅ Fixed | `phase_b_top5_probe_weights.csv` has weights |
| A.6: 1024-d vs 64-d dimensionality | ✅ Resolved | Switched to Audio 64-d |
| B.1: Signed ratio formula bug | ✅ Fixed | Replaced with L1 probe weight ranking |
| B.2: Phase D gate filter | ✅ Fixed | All 6 classes evaluated |
| B.3: Pre-processing leakage | ✅ Fixed | `Pipeline` used in Cell 27 |
| B.4: Discarded probe weights | ✅ Fixed | `pipe.fit()` called after CV |
| B.5: Selectivity edge case | ✅ Fixed | `target_drop > 0` guard added |
| B.6: Epsilon division artifact | ✅ Fixed | Clean ratio formatting |
| B.7: CSV path mismatch | ✅ Fixed | Both individual + combined CSVs saved |
| C.1: Deterministic seed missing | ✅ Fixed | `set_deterministic_seed(seed=0)` in all Phase cells |
| C.2: Missing markdown cells | ✅ Fixed | All code cells have markdown headers |
| C.3: Ephemeral dependencies | ⚠️ Open | Still requires manual `!pip install` |

---

## PART 4: Code-Level Verification (Line-by-Line)

### Cell 10 (SHAP Aggregation) — 🔴 BUG
```python
per_sample = np.abs(np.sum(sv[:, start:end], axis=1))
#                   ^^^^^^^^^^^^^^^^^^^^^^^^
# This sums SIGNED SHAP values first, THEN takes abs.
# Correct: np.sum(np.abs(sv[:, start:end]), axis=1)
```

### Cell 18 (Activation Slicing) — ✅ CORRECT
```python
base_acts_tier1 = base_full_1152[:, 1088:1152]  # Audio 64-d
ft_acts_tier1   = ft_full_1152[:, 1088:1152]     # Audio 64-d
```
Correctly targets Audio branch.

### Cell 27 (Probe Pipeline) — ✅ CORRECT
Uses `Pipeline([('scaler', StandardScaler()), ('probe', LogisticRegression(...))])` with `cross_val_predict`, then `pipe.fit(acts, y_binary)` for persistent coefficients. No pre-processing leakage.

### Cell 30 (Neuron Ranking) — ✅ CORRECT
```python
coefs = np.abs(probe.coef_[0])
ranked_indices = np.argsort(coefs)[::-1]
```
Ranks by absolute L1 probe weight magnitude as protocol requires.

### Cell 32 (Proxy Inference) — ✅ CORRECT
```python
specs[:, n] = dataset_mean_acts[n]  # Ablates Audio neurons
```
Correctly clamps Audio (`specs`) neurons, not Text.

Architecture verification: `e2e.py` L96-97 shows `text_cls = self.T(text, get_cls=True)` → `self.t_out(text_cls)`. L121-123 shows `specs = self.a_transformer(specs, spec_lens, get_cls=True)` → `self.a_out(specs)`. The proxy inference in Cell 32 replicates this exactly.

### Cell 34 (Sweep Evaluation) — ⚠️ ISSUE
Evaluates only `[576:720]` (test set). Protocol says "full eval set."

### Cell 36 (Cosine Similarity) — ✅ CORRECT
```python
selectivity[d] = mean(act[d] | c) - mean(act[d] | ~c)
sim = 1.0 - cosine(base_sel, ft_sel)
```
Uses full 720-sample activations for selectivity vectors. Correct.

### Cell 39 (Transfer Retention) — ⚠️ ISSUE
```python
R = round(ft_drop / base_drop, 2) if base_drop > 0 else 0.00
```
Sets R=0.0 when base_drop=0, which forces "Substrate Dispersion." Should be flagged as "Undefined" or "N/A."

---

## PART 5: Statistical Sanity Check of Results

### Phase A (SHAP Attribution)
- **Base:** Audio 50.48%, Text 43.66%, Video 5.86%
- **FT:** Audio 58.51%, Text 36.15%, Video 5.34%
- Audio leads by 6.81% (base) and 22.36% (FT) — above 5% threshold → Audio dominant.
- ⚠️ **BUT** these percentages were computed with the `|sum(SHAP)|` formula. With `sum(|SHAP|)`, the results would likely differ significantly (Text has 16x more features, so more cancellation under signed summation).

### Phase B (Probe AUCs)
- **Base:** Mean AUC 0.85 (range: 0.75–0.94). All classes > 0.55. ✅ PASS.
- **FT:** Mean AUC 0.84 (range: 0.71–0.93). All classes > 0.55. ✅ PASS.
- These are strong AUCs for a 64-d linear probe — the Audio branch genuinely carries class-discriminative signal.

### Phase C (Ablation Drops)
- **Base:** ALL classes = 0.00% target drop. No causal selectivity.
- **FT:** Only Anger = 4.17% drop (1 sample flip out of 24).
- **k=10 sweep also shows near-zero drops** — ruling out "k too small" as the sole explanation.
- The k=10 base surprise class shows the interesting pattern: [0.0, 4.17%, -4.17%, 0.0, 4.17%, 0.0] — which is just noise (1 sample flip in multiple classes).

### Phase D (Cosine Similarity)
- All 6 classes show high cosine similarity (0.71–0.98) between base and FT selectivity vectors.
- This means the Audio branch's class representations barely rotated during fine-tuning.
- This is consistent with the probe AUCs being similar (0.85 vs 0.84).

### Phase D (Transfer Retention)
- R = 0/0 → 0.0 for all classes. Substrate Dispersion by default. Scientifically void.

### Internal Consistency Check
The results tell a **coherent but incomplete** story:
1. Audio carries linear signal (strong probe AUCs).
2. Audio selectivity vectors are preserved through fine-tuning (high cosine similarity).
3. But ablating 5 Audio neurons doesn't affect predictions (zero drops).

Interpretation options:
- **(A) The model is genuinely robust** to sparse Audio neuron ablation because multimodal redundancy compensates. This is plausible but requires a dose-response control.
- **(B) The SHAP aggregation is wrong** and Audio isn't actually dominant. If Text is dominant and you're ablating Audio, near-zero drops are trivially expected.
- **(C) The evaluation set is too small** (n=24 per class) to detect real effects.

**All three explanations are simultaneously possible.** The paper needs to rule out (B) and (C) before claiming (A).

---

## PART 6: Prioritized Remediation Plan

| Priority | Issue | Action | Effort | Impact |
|:---|:---|:---|:---|:---|
| **P0** | SHAP aggregation formula | Change `\|sum(SHAP)\|` to `sum(\|SHAP\|)` in Cell 10. Recompute dominant modality verdict. If Text becomes dominant, the entire experiment must be re-evaluated. | 10 min | Potentially invalidates everything |
| **P1** | Test-only evaluation (144 samples) | Evaluate ablation on full 720 samples, reporting train and test accuracy separately. | 30 min | Increases statistical power 5x |
| **P2** | Dose-response control | Add k=16, k=32, k=48, k=64 to the sweep to show whether *any* ablation level causes drops. | 30 min | Required to support "redundancy" claim |
| **P3** | Undefined R taxonomy | When base_drop = 0, report R as "N/A" instead of 0.0. Modify substrate outcome logic to output "Indeterminate (Base Effect Undetected)" | 15 min | Honest reporting |
| **P4** | Probe-fit data leakage | Re-fit probes on training set only (indices 0:518), re-rank neurons | 30 min | Methodological purity |
| **P5** | Cosine vs ablation contradiction | Add explicit discussion paragraph addressing the Day 13/Day 14 disagreement | 15 min | Reviewer anticipation |
| **P6** | Full-modality knockout control | Ablate ALL 64 Audio neurons as a sanity check. If the model still predicts correctly, this proves the Audio branch is dispensable. | 15 min | Definitive test |

---

## PART 7: What a Peer Reviewer Would Say

> **Reviewer 1 (Methodology):**
> "The authors claim to identify the dominant modality via aggregated SHAP attribution, but their aggregation formula `|sum(SHAP)|` allows signed cancellation within modalities, systematically favoring lower-dimensional branches. The correct formula is `sum(|SHAP|)`. This methodological error may have led the authors to target the wrong modality. Additionally, evaluating causal ablation on only 24 samples per class provides negligible statistical power. The Transfer Retention Ratio is undefined (0/0) for all classes, making the substrate taxonomy classifications meaningless. Major revision required."

> **Reviewer 2 (Experimental Design):**
> "The ablation results show zero effect across all conditions, but the authors interpret this as evidence of 'multimodal redundancy' without providing dose-response evidence (what happens at k=32 or k=64?). A full-modality knockout (ablate all 64 Audio neurons) would be far more informative. Without this control, the null result is uninterpretable — it could equally reflect targeting the wrong modality, insufficient statistical power, or genuine redundancy. The claim is not adequately supported."

> **Reviewer 3 (Internal Consistency):**
> "The Day 13 cosine similarity analysis shows uniformly high similarity (0.71-0.98) suggesting Substrate Preservation, while the Day 14 ablation transfer claims Substrate Dispersion for 5/6 classes. The authors acknowledge this contradiction but resolve it by privileging the ablation result, which itself is based on undefined (0/0) ratios. The paper should honestly report that the ablation methodology lacked the power to distinguish between its three substrate outcomes at the given sample size."

---

## Summary Verdict

The experiment has a **sound conceptual framework** and the proxy inference mechanism is mathematically verified. The remediation since Aug 4 fixed many code bugs. However, three fundamental issues remain:

1. **The SHAP aggregation formula may point to the wrong modality** (P0 — must be checked immediately)
2. **The evaluation has negligible statistical power** (P1+P2 — 24 samples per class, no dose-response)
3. **The Transfer Retention taxonomy is vacuous** when base drops are zero (P3)

Until these are resolved, the experiment produces valid but uninterpretable null results that cannot support the claims needed for Section VI-D.
