# Phase D Methodology Audit

**Date:** 2026-08-04
**Auditor:** Antigravity (Claude Opus 4.6)
**Ground Truth:** Prof. KC Lan's 3-week experiment protocol
**Status:** IN PROGRESS — Proxy inference verified ✅; 4 issues remain

---

## 1. Executive Summary

This audit compares our actual notebook execution (Phases A–D in `multimodal-causal-ablation.ipynb`) against the step-by-step experiment protocol provided by Prof. KC Lan. Five issues were identified, ranging from critical (potentially invalidating results) to minor (missing complementary evidence). A prioritized remediation plan is included at the end.

---

## 2. What We Executed Correctly

| Protocol Step | Our Phase | Status | Notes |
|:---|:---|:---|:---|
| Day 0.1: Confirm both checkpoints | Setup | ✅ | Base + fine-tuned loaded and verified |
| Day 0.2: Identify dominant modality via SHAP | Phase A | ✅ | Text dominant (~70%), data-driven via aggregated SHAP |
| Day 0.3: Pick the layer to hook | Phase B | ✅ | Confirmed Text branch has no FFN; Tier 1 = ALBERT CLS (1024-d) |
| Day 0.4: Confirm feature dimensionality | Phase B | ✅ | 1024-d verified against code (`e2e.py`) |
| Day 3: L1-penalized logistic probes | Phase B | ✅ | Fit with `liblinear`, L1 penalty, balanced class weights |
| Day 4: Sanity-check probes via AUC | Phase B | ✅ | Base Mean AUC 0.74, FT Mean AUC 0.68 — both pass 0.65 threshold |
| Day 5: Top-5 neuron table | Phase C | ✅ | Produced per-class top-5 neuron indices for both models |
| Day 6-7: Mean-ablation (not zero) | Phase C | ✅ | Dataset-mean clamping, exactly as protocol specifies |
| Day 8: Ablate on base model, k={1,3,5,10} | Phase C | ✅ | Full sweep saved to `results/phase_c_base_ablation_sweep.json` |
| Day 9: Repeat for fine-tuned model | Phase C | ✅ | Same procedure, FT's own rankings; saved to `results/phase_c_finetuned_ablation_sweep.json` |
| Day 14: Ablation transfer ("the stronger test") | Phase D | ✅ | Base top-5 neurons ablated inside FT model; result saved to `results/phase_d_transfer_retention.csv` |

---

## 3. Issues Identified

### Issue 1 (CRITICAL): Sample Size — Only 144 Test Samples Used

**Protocol says (Day 1-2):**
> *"run a forward pass over the full RML dataset (train+test combined is fine here — you're not training anything, just observing)"*

**What we did:**
We extracted activations exclusively from the 144 test samples cached in the SHAP pickles (`base_shap.pkl`, `finetuned_shap.pkl`). The protocol explicitly says to use train+test combined because you're not training anything — you're just computing selectivity statistics.

**Why this matters:**
With 144 samples across 6 classes, that's ~24 samples per class. A single sample flipping its prediction = 4.17% accuracy change. Our entire Phase D conclusion (Substrate Dispersion) rests on the `surprise` class showing a 4.17% drop (literally 1 sample) and every other class showing 0.00%. This granularity is too coarse to draw confident causal conclusions.

**Remediation:**
Re-run activation extraction over the full RML dataset (train+val+test splits) by performing a fresh forward pass through both models on Colab. Save the full activation cache to `checkpoints/activations/`. Then re-run the selectivity ranking and ablation sweeps with the larger sample.

**Files affected:** Phase B (Section 4-5), Phase C (Sections 6-8), Phase D (Section 9)

---

### Issue 2 (HIGH): Neuron Ranking Used Activation Ratios, Not Probe Weights

**Protocol says (Day 3):**
> *"Rank the 64 dimensions by absolute weight magnitude for each class."*

**Protocol says (Day 11-12):**
> *"You can reuse the Day 3 probe weights instead if you prefer"*

**What we did:**
In Phase C (Section 6), we ranked neurons using a raw activation ratio:
```
selectivity_ratio = mean(activation[d] | label=c) / mean(activation[d] | label!=c)
```

The protocol says to rank neurons by the **absolute L1 probe weight magnitudes** — which directly tells you which neurons the linear classifier relies on to detect each emotion. These two metrics can diverge significantly: a neuron might have a high activation ratio but carry no useful linear signal (e.g., if variance is enormous), or vice versa.

**Remediation:**
After fitting the L1 probes (which we already do in Phase B, Section 5.2), extract the `.coef_` vectors from the fitted `LogisticRegression` objects. Rank neurons by `|coef_[c, d]|` for each class `c`. Use these rankings for the ablation sweep instead of the activation ratio.

**Files affected:** Phase C (Section 6), which cascades to Sections 7-8 and Phase D Section 9.

---

### Issue 3 (HIGH): Day 13 Cosine Similarity Entirely Skipped

**Protocol says (Day 13):**
> *"For each class, compute the cosine similarity between the base model's selectivity vector and the fine-tuned model's selectivity vector."*
> *"High similarity -> fine-tuning sharpened the same existing representation"*
> *"Low similarity -> fine-tuning moved the class's representation to different neurons"*

**What we did:**
We skipped this step entirely and went straight to Day 14 (ablation transfer).

**Why this matters:**
The protocol frames Day 13 and Day 14 as complementary evidence. Day 13 provides a geometric/representational view (did the selectivity vector rotate?), while Day 14 provides a causal/functional view (are the same neurons still load-bearing?). A reviewer will expect both. The protocol explicitly says Day 14 "should be your headline result if the two disagree" — implying both should be present for the comparison to be made.

**Remediation:**
For each class `c`, compute a selectivity vector:
```
selectivity[d] = mean(activation[d] | label=c) - mean(activation[d] | label!=c)
```
for both base and fine-tuned models. Then compute `cosine_similarity(base_selectivity[c], ft_selectivity[c])` for each class. Produce a 6-row table (one per emotion class) with the cosine similarity value.

**Files affected:** New notebook section (Section 9.5 or equivalent), new CSV artifact `results/phase_d_cosine_similarity.csv`.

---

### Issue 4 (RESOLVED ✅): Proxy Inference Is Mathematically Identical

**Concern:**
In Phase C (Section 7), we bypassed the actual model forward pass. Instead, we took the cached 1152-d SHAP `test_feature` vectors, split them into Text (0:1024), Video (1024:1088), Audio (1088:1152), and fed those directly into `model.t_out()`, `model.v_out()`, `model.a_out()`, and `model.weighted_fusion()`.

**Audit Result (verified via line-by-line trace of `e2e.py`):**

Proxy inference is **100% mathematically valid**. The audit confirmed:

1. **Text branch:** `text_cls` (1024-d ALBERT CLS token) is passed **directly** into `self.t_out` (`nn.Linear(1024, 4)`). No LayerNorm, Dropout, or any transformation in between.
2. **Video branch:** `faces` (64-d Transformer CLS output) is passed **directly** into `self.v_out` (`nn.Linear(64, 4)`). No intermediate layers.
3. **Audio branch:** `specs` (64-d Transformer CLS output) is passed **directly** into `self.a_out` (`nn.Linear(64, 4)`). No intermediate layers.
4. **Fusion:** `weighted_fusion` is `nn.Linear(3, 1, bias=False)` — a pure linear combination of the three modality logits with no non-linearities. The MLP alternative in the code is commented out.
5. **SHAP `test_feature` concatenation:** `torch.cat([text_cls, faces, specs], dim=1)` produces the 1152-d vector, which is the **exact** input to the three classification heads.

**Conclusion:** Clamping neuron indices in the cached 1152-d feature and running through the classification heads produces **bit-for-bit identical** results to a full forward pass with those neurons clamped via hooks. No remediation needed.

---

### Issue 5 (LOW): 1024-d vs 64-d Dimensionality Concern

**Context:**
The protocol was written assuming the dominant modality would be Audio (64-d FFN output). Our SHAP data drove us to Text (1024-d ALBERT CLS). This is scientifically defensible — you follow the data, not the prior assumption.

**Why this matters:**
The change in dimensionality affects the statistical power of the selectivity ranking. Searching for 5 selective neurons out of 1024 is a much harder needle-in-a-haystack problem than 5 out of 64. A reviewer could argue that the Substrate Dispersion finding is partly an artifact of the higher dimensionality making sparse selection statistically unlikely.

**Remediation:**
- Acknowledge this in the paper's discussion section.
- Optionally: run a quick sanity ablation on the 64-d Audio branch (dims 1088-1151) as a robustness check. If Audio also shows dispersion at 64-d, the dimensionality argument is neutralized.

---

## 4. Prioritized Remediation Plan

Execute in this order (later steps depend on earlier ones):

| Priority | Task | Depends On | Estimated Effort | Status |
|:---|:---|:---|:---|:---|
| ~~P0~~ | ~~**Verify proxy inference** against `e2e.py` architecture~~ | — | ~~30 min~~ | ✅ RESOLVED |
| P1 | **Re-extract activations over full RML dataset** (train+val+test) | — | 1–2 hours (Colab GPU) | TODO |
| P2 | **Re-rank neurons using L1 probe weights** (`\|coef_\|`) | P1 | 30 min | TODO |
| P3 | **Re-run Phase C ablation sweep** with corrected rankings + larger sample | P1, P2 | 1 hour | TODO |
| P4 | **Compute Day 13 cosine similarity** | P1 | 15 min | TODO |
| P5 | **Re-run Phase D** with corrected data | P3, P4 | 15 min | TODO |
| P6 | **Optional: 64-d Audio sanity check** | P1 | 30 min | TODO |

**Total estimated effort:** ~4–5 hours of Colab GPU time across 1-2 sessions.

---

## 5. Files Reference

| Artifact | Path |
|:---|:---|
| Main notebook | `multimodal-causal-ablation.ipynb` |
| Phase C base sweep | `results/phase_c_base_ablation_sweep.json` |
| Phase C FT sweep | `results/phase_c_finetuned_ablation_sweep.json` |
| Phase C base k=5 | `results/phase_c_base_ablation_k5.csv` |
| Phase C FT k=5 | `results/phase_c_finetuned_ablation_k5.csv` |
| Phase D transfer retention | `results/phase_d_transfer_retention.csv` |
| Model architecture | `Model/Dig-Data_Model-Main/src/models/e2e.py` |
| ADR 0001 (methodology) | `docs/adr/0001-causal-validation-methodology.md` |
| Prof. KC Lan's protocol | (provided verbally / chat, not yet saved to repo) |

---

## 6. Open Questions

1. **If proxy inference is invalid**, do we need to implement forward hooks for the ablation? This would require restructuring Sections 7-8 significantly.
2. **If full-dataset re-extraction changes the selectivity rankings**, how do we handle the discrepancy with existing Phase C results? Answer: the new results supersede; old artifacts should be archived or deleted.
3. **Should we save Prof. KC Lan's protocol verbatim** to `docs/` as a reference document?
