# ADR 0006 — Week 3 Transfer Ratio: Random-Ablation Null and Withdrawal of the Substrate Bands

## Status

Accepted, 2026-08-11. Supersedes `0005-day0-dominant-modality-recompute-and-protocol-alignment.md` Decision 7, and withdraws the substrate taxonomy in `0001-causal-validation-methodology.md` Decision 4 and in `CONTEXT.md`. Weeks 0–2 are unaffected; ADR 0004's provenance and fidelity-gate decisions and ADR 0005 Decisions 1–6 and 8 all stand.

## Context

ADR 0005 Decision 7 settled a disagreement between ADR 0001 and `CONTEXT.md` over where the substrate bands should sit, and chose `CONTEXT.md`'s version because its bands are total. That resolution answered the wrong question. Both schemes band only the region `R <= 1`:

| Source | Preservation | Reassignment | Dispersion |
| :--- | :--- | :--- | :--- |
| ADR 0001 Decision 4 | R ≥ 0.70 | R < 0.30, fine-tuned drop sparse | R < 0.30, fine-tuned drop dense |
| `CONTEXT.md` | R ≥ 0.80 | 0.20 ≤ R < 0.80 | R < 0.20 |

The measured transfer ratios are 1.32, 1.39, 1.69, 2.07, 2.08 and 4.34. Every class sits above 1, so every class lands in Preservation under either scheme, and the taxonomy separates nothing. Worse, it invites the reading that `R > 1` means the base model's neurons remain load-bearing after fine-tuning — that the substrate was preserved.

That reading does not follow, because of a confound the bands cannot see. `R` is the ratio of the fine-tuned model's drop to the base model's drop under the same ablation. The fine-tuned model is simply more fragile: ablating *any* five audio-CLS dimensions hurts it more than it hurts the base model. A ratio above 1 is therefore the default state of this pair of models under any ablation whatsoever, not evidence about the particular neurons chosen. Clearing 1 is not a test. The comparison has to be against what arbitrary dimensions do.

## Decisions & Rationale

### 1. The substrate bands are withdrawn

No band, index, or normalised score is reported for Week 3. `R` is reported per class with its 95% CI, next to the raw `base_drop_pp` and `ft_drop_pp` it is built from, and next to the measured null described below. The label a class receives comes from where its CI falls relative to the null, not from a threshold chosen in advance.

`results/measurement_resolution.json` records this withdrawal under `substrate_bands`, and `results/week3_transfer_retention.csv` carries `R`, `R_ci95_lo`, `R_ci95_hi`, `R_null`, `R_null_ci95_lo`, `R_null_ci95_hi`, `R_exceeds_null`, `R_below_null` and a per-class `verdict` string.

A normalised or re-banded index must not be reintroduced later to rescue a transfer claim. If the raw ratios and the null do not support a claim, the claim is not there.

### 2. The transfer ratio is tested against a measured random-ablation null

`results/week3_random_ablation_null.json` draws 2000 random 5-dimension subsets of the 64-dimensional audio CLS (`k = 5`, `seed = 20260812`, deliberately distinct from the bootstrap seed so the null and the CI are not sharing a stream) and, for each class, computes the same ratio the real top-5 subset gets. The estimator is the ratio of the mean fine-tuned drop to the mean base drop over subsets — a ratio of means, never a mean of ratios, since individual subsets can produce near-zero denominators. The CI is a bootstrap over subsets.

| class | base drop (pp) | fine-tuned drop (pp) | R | R 95% CI | R_null | R_null 95% CI |
| :--- | ---: | ---: | ---: | :--- | ---: | :--- |
| anger | 4.3286 | 7.3154 | 1.6900 | [1.1962, 2.4108] | 1.4150 | [1.3993, 1.4311] |
| disgust | 2.3000 | 4.7920 | 2.0835 | [1.4680, 2.9032] | 2.0050 | [1.9879, 2.0232] |
| fear | 0.6387 | 2.7714 | 4.3393 | [2.6772, 9.3368] | 4.0093 | [3.8999, 4.1363] |
| happiness | 2.0433 | 2.8454 | 1.3926 | [0.7825, 2.3622] | 1.1852 | [1.1591, 1.2108] |
| sadness | 4.0641 | 5.3542 | 1.3174 | [1.0768, 1.7162] | 1.2073 | [1.1951, 1.2195] |
| surprise | 2.2882 | 4.7421 | 2.0724 | [1.5534, 2.5328] | 1.7256 | [1.7080, 1.7435] |

**Result: 0 of 6 classes have an `R` whose 95% CI clears the null, and 0 of 6 fall below it.** Every per-class verdict is `No transfer signal above random ablation`. The base model's top-5 neurons, ablated inside the fine-tuned model, do no more damage than five arbitrary audio dimensions.

Fear is the class most likely to be mistaken for a signal and is the clearest illustration of why the null was needed. Its `R` of 4.34 is by far the largest, and under either band scheme it would have been the strongest Preservation case in the table. Its null is 4.01. The ratio is large because fear's base drop is 0.6387 pp — a small denominator inflates the ratio for random subsets exactly as it does for the selected ones. Its `R_defined_fraction_of_draws` is 0.99, the only class below 1.0, which is the same fragility showing up in the bootstrap.

### 3. The write-up must state that all six point estimates sit above their null

Every class's `R` point estimate exceeds its `R_null` point estimate. No CI separates them, because the `R` intervals are wide (144 test samples, 24 per class) while the null intervals are tight (2000 subsets). "No per-class significance" is the correct and conservative conclusion and is the one this ADR adopts.

The 6-of-6 pattern is nonetheless the first thing a reader will notice, and the paper states it plainly rather than leaving it to be found: the direction is consistent across classes, the effect sizes are small relative to their uncertainty, and the study is not powered to resolve a difference of this magnitude. Whether the consistent direction reflects a real weak effect is a question for a larger evaluation set, not something this data answers.

### 4. Probability drop is the primary measure; accuracy is secondary

The test split holds 144 samples, 24 per class, so argmax per-class accuracy can only move in steps of 4.166666666666667 pp. Every Week 2 and Week 3 effect measured here is smaller than one step. Accuracy therefore cannot resolve them, and an accuracy-based table would show mostly zeros punctuated by single-step jumps that look like large effects and are not.

The primary measure is the mean true-class softmax probability drop in percentage points, with a paired class-stratified bootstrap, 2000 draws, `bootstrap_seed = 20260811`, and 95% CIs. Decision margin is reported alongside. Accuracy drops are retained in `week3_transfer_retention.csv` under `legacy_acc_*` column names so the quantisation is visible rather than hidden: anger's legacy accuracy drops are 0.0 and 8.333, two steps apart, against continuous drops of 4.33 and 7.32.

`results/measurement_resolution.json` records the split sizes, the resolution, the primary measure, and the seeds.

### 5. No epsilon is ever substituted for a denominator

Any ratio in this experiment is NaN with a stated reason when its denominator is unusable, never a small constant. For the Week 2 selectivity ratio the denominator is used only when at least one non-target class has an effect whose CI excludes zero **and** the mean absolute non-target effect clears `ratio_denominator_floor_pp = 0.1`. The floor is a chosen constant, so the sweep prints the observed distribution of denominators against it, allowing it to be calibrated on real data rather than defended in the abstract.

### 6. Top-5 overlap is tested against a hypergeometric null

Overlap between the base and fine-tuned top-5 sets is drawn from Hypergeometric(64, 5, 5) under the null, with expected overlap 0.390625 and P(X ≥ 1) = 0.343383, P(X ≥ 2) = 0.044920, P(X ≥ 3) = 0.002283. Observed: anger 3, surprise 3, fear 2, disgust 1, sadness 1, happiness 0. Three of six classes are at or below chance. An overlap of 1 is not evidence of anything and must not be reported as partial agreement.

### 7. Cosine separation passes but does not outrank the transfer result

The selectivity vectors separate cleanly: cosine similarity between the base and fine-tuned per-class selectivity vectors runs 0.9602–0.9896, all six classes are nearest their own class, and matched minus mismatched is +1.168221021057142. This is a correlational statement about the probe weights. The protocol's Week 3 question is causal, the ablation transfer answers it directly, and where the two disagree the causal measurement governs. The cosine result is reported as a supporting observation, not as transfer evidence.

Day 13b (`results/week3_rank_validation.csv`, `results/table_x_probe_rank_validation.csv`) makes the same point about the selection rule itself. Across all 12 model-class pairs, ablating the probe's top-5 dimensions does less damage than ablating the measured top-5 — `probe_over_measured` runs 0.1585 to 0.7306, every pair below 0.80 — and the Spearman correlation between probe weight and measured effect over the top-16 union runs −0.5438 to +0.0023, negative in eleven of twelve. `|probe weight|` is a way of selecting candidates to test, not a way of identifying causal neurons.

## Consequences

- Week 3 is publishable as a **negative result with a measured null behind it**. It is not a shared-substrate result and must never be written as one.
- `CONTEXT.md`'s substrate band table and ADR 0001 Decision 4 are historical. They are not deleted from those documents, which are records, but they are not implemented and no future run should reinstate them.
- Weeks 0–2 are unchanged by this ADR.
- Any future claim of transfer requires either a larger evaluation set or a different intervention, plus its own null. Re-running the same 144-sample test split will not change the answer.
