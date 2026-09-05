# Spec: Direct Causal Neuron Screening Protocol (Future Work)

**Status:** Proposed specification for future work / next iteration.  
**Companion Theoretical Analysis:** [`docs/research-notes/neuron-causal-screening-methodology.md`](research-notes/neuron-causal-screening-methodology.md) (comprehensive assessment by Codex `gpt-5.6-sol`).

---

## 1. Context & Motivation

In Section VI-D of the paper (*Modality Competition in Multimodal Emotion Recognition*), the primary protocol selected "top class-selective neurons" using L1-regularized logistic regression probes on 64-dimensional audio CLS activations (Approach A).

However, an empirical audit ([Table X](file:///Users/breznev/Library/CloudStorage/GoogleDrive-mendresvon@gmail.com/My%20Drive/multimodal-causal-ablation/results/table_x_probe_rank_validation.csv)) revealed a fundamental disconnect:
- In **12 of 12** model-by-class pairs, probe-selected neurons caused significantly less causal damage than damage-selected neurons (achieving only 16% to 73% of measured damage).
- The Spearman rank correlation between probe weight magnitude and causal ablation damage was **negative or zero in 11 of 12** pairs.

An external linear probe identifies coordinates that linearly separate emotion classes (decodability), but does not guarantee that the host model causally depends on those coordinates. Direct single-neuron causal ablation screening (Approach B) eliminates this proxy and aligns the candidate selection criterion directly with the downstream causal evaluation.

---

## 2. Experimental Design & Protocol

To implement Approach B in future work while remaining methodologically airtight, the protocol must satisfy six core requirements:

### Step 1: Strict Train-to-Test Split Discipline (Preventing Selection Bias)
- **Discovery Split (Train, N=518):** All single-neuron screening, selectivity scoring, and top-5 greedy subset selection must occur strictly on the training activations.
- **Confirmation Split (Test, N=144):** The selected 5-neuron set is locked before running inference on the held-out test split. Evaluating selected winners on the discovery data produces optimistically biased effect sizes (the "winner's curse").

### Step 2: Continuous Logit-Margin Metric
Instead of raw argmax accuracy (which quantizes into 4.17 pp steps on N=24 per class) or unconstrained softmax probability (which suffers from multiclass coupling and saturation), define the primary target evidence as a continuous **logit margin**:
```text
m_c(x) = z_c(x) - mean_{k != c}(z_k(x))
```
- **Target necessity:** Paired drop in target-class margin under ablation:
  `D_target(j, c) = E[m_c(x) - m_c(x; do(h_j := b_j)) | Y=c]`
- Rank by **signed loss of target evidence**. Coordinates where ablation *increases* target margin represent suppressive neurons and must be categorized separately.

### Step 3: Class Selectivity vs. Generic Disruption
Ranking solely by target-class drop risks selecting generic audio circuit breakers (coordinates required for basic acoustic energy or phonation across all emotions).
For each coordinate `j` and class `c`, compute:
1. `Target Necessity(j, c)`: drop on true class-`c` examples.
2. `Off-Target Disruption(j, c)`: macro-average drop on all non-`c` classes on their respective true examples.
3. `Selectivity Contrast`:
   ```text
   Score(j, c) = Target Necessity(j, c) - Off-Target Disruption(j, c)
   ```
Only coordinates with a positive contrast (or exceeding the 2.5x selectivity ratio) qualify as class-selective.

### Step 4: Greedy Forward Selection (Addressing Redundancy)
As proven in the architecture audit, single-coordinate effects are strictly additive in logit space:
```text
Δz_c(x, j) = α_audio * W_audio[c, j] * (h_j(x) - mean_train(h_j))
```
However, downstream probability and margin transformations saturate. Taking the top 5 individual single-neuron knockouts can select redundant coordinates that duplicate the same shift.
- **Algorithm:** 5-step greedy forward selection on the training set:
  1. Pick coordinate `j_1` that maximizes `Score(j, c)`.
  2. Given `S_1 = {j_1}`, find `j_2` from the remaining 63 coordinates that maximizes the joint score `Score(S_1 U {j_2}, c)`.
  3. Repeat until `|S| = 5`.
- **Cost:** Requires only `64 + 63 + 62 + 61 + 60 = 310` forward evaluations. On the cached linear fast path, this takes < 0.5 seconds.

### Step 5: Replacement Baseline & Sensitivity
- **Primary Baseline:** Dataset-mean replacement (`train_mean[d]`) computed strictly from training samples, preventing out-of-distribution shocks that occur with zero-clamping.
- **Robustness Checks:** Verify stability against random-resample ablation (sampling coordinates from other training utterances) and zero-ablation.

### Step 6: Transfer & Fine-Tuning Survival Test
When testing whether base-model neurons survive fine-tuning:
1. **Same-Index Persistence:** Lock the base model's top-5 coordinates and ablate them inside the fine-tuned model.
2. **Benchmark Against the Random Null:** Compare the resulting transfer retention ratio `R` against an empirical null of **2,000 random 5-coordinate subsets** (as established in ADR 0006). Never conclude preservation from `R > 1` alone, which reflects the fine-tuned model's global fragility.

---

## 3. Recommended Paper Positioning

In future write-ups, Approach A (L1 probing) and Approach B (causal screening) should be presented as **complementary lenses**:
1. **Approach A (Probing):** Reports *representational decodability*—where emotional information is linearly readable in the activation space.
2. **Approach B (Causal Screening):** Reports *functional reliance*—which coordinates the model actually depends upon for its predictions.
3. **The Dissociation Finding:** Presenting the empirical divergence between probe weights and causal damage as an explicit result demonstrates that readable features are not necessarily used features, providing a nuanced mechanistic insight into multimodal fusion.

---

## 4. References & Documentation

- [Codex Comprehensive Methodology Assessment](research-notes/neuron-causal-screening-methodology.md)
- [Table X: Probe vs. Measured Causal Damage Audit](../results/table_x_probe_rank_validation.csv)
- [ADR 0006: Random Ablation Null & Band Withdrawal](adr/0006-week3-transfer-null-and-band-withdrawal.md)
- [Section VI-D Review Briefing](section-vi-d-professor-briefing.md)
