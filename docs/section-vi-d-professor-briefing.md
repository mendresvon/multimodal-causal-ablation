# Section VI-D professor briefing

Prepared 2026-08-13 from the current v3 notebook, exported results, upstream model source, training logs, protocol, ADRs, journals, and latest paper draft. This is a meeting-preparation document, not paper-ready prose.

## The 90-second answer

> Section VI-D is a preliminary, model-internal causal case study on RML. Section VI-C uses DeepSHAP, so it can say which representation is associated with predictions but not whether particular units are functionally necessary. I target the 64-dimensional post-Transformer audio CLS representation because audio has the highest attribution per coordinate in both RML models—5.21 times the runner-up in the scratch base model and 6.94 times in the fine-tuned model—even though text has greater total attribution mass because it has 1,024 coordinates.
>
> I fit one-vs-rest L1 logistic probes on 518 training utterances to select five candidate coordinates per class, then replace those coordinates at inference with their model-specific training-set means. I evaluate on 144 held-out utterances, 24 per class. Per-class accuracy has a 4.17-percentage-point quantum, so it cannot resolve these small interventions; the primary outcome is mean drop in true-class softmax probability with 2,000 paired bootstrap draws.
>
> The positive result is mixed: every one of the twelve model-by-class top-five interventions reduces target-class confidence with a nominal 95% interval excluding zero. Four of six classes meet the 2.5 selectivity ratio in the base model, and three of six in the fine-tuned model. The negative result is that the base model’s selected coordinate indices do not show detectable special alignment with the fine-tuned model beyond a random five-coordinate baseline: zero of six classes clear the implemented confidence-interval separation rule.
>
> I would not call that proof that neurons fail to transfer. The scratch base model is not the ancestor of the fine-tuned checkpoint; the comparison is cross-model coordinate alignment. The random-null comparison is a useful sensitivity control but not a direct randomization p-value, the probe weights rank readability rather than causal damage, and the analysis covers one dataset, one modality, one layer, one model seed, and a non-speaker-independent split. The safe conclusion is that selected audio CLS coordinates are causally load-bearing and sometimes class-concentrated, while no special cross-model coordinate-level signal was detected with this design.

Sources: `experiment_protocol[From Prof KC Lan].docx`, Day 0 and Weeks 1–3; `results/day0_dominant_modality.json`; `results/measurement_resolution.json`; `results/table_vii_base_ablation_k5.csv`; `results/table_viii_finetuned_ablation_k5.csv`; `results/week3_transfer_retention.csv`; `docs/provenance/RML_{origin,finetune}_training_log.txt`.

## What your task actually is

Section VI-D is not a second full transfer-learning study. It is a bounded causal-validation case study intended to test one mechanistic layer beneath Section VI-C:

1. Does the RML model’s attributed audio representation contain coordinates whose intervention changes the model’s emotion confidence?
2. Is that damage concentrated on the class used to select the coordinates?
3. Do the scratch-RML and transfer-learned RML solutions use specially aligned coordinate indices?

RML was chosen because it is small and because it is one of the paper’s puzzle cases: the pretrained and scratch base models have the same aggregate dominant-modality assignment, yet fine-tuning yields only a limited gain. The original protocol asks whether neuron-level reassignment explains that puzzle. The executed result does not supply that explanation. (`experiment_protocol[From Prof KC Lan].docx`, Goal and Week 3; `Latest_Paper_Draft.docx`, Sections VI-C–VI-D.)

The causal claim is deliberately model-internal: changing an activation causes a change in this trained model’s output. It is not a causal claim about human emotion, and it does not by itself prove that modality competition during training caused the learned representation.

## First conceptual trap: the two checkpoints are not one lineage

The “base” checkpoint was trained from scratch on RML. The “fine-tuned” checkpoint descends from a separate model pretrained on the other five datasets and then optimized on RML. The fine-tuning log explicitly records loading `1014_pretrain_no_mosei_no_RML`; the base log has no such load. (`docs/provenance/RML_origin_training_log.txt`, lines 1–12; `docs/provenance/RML_finetune_training_log.txt`, lines 1–13; protocol Day 0.)

Therefore, taking coordinate 20 in the scratch base model and coordinate 20 in the fine-tuned model is a comparison of the same architectural position in two different training lineages. It is not tracking a parameter physically carried from the base checkpoint through fine-tuning.

Defensible interpretation:

- Table IX asks whether independently learned scratch and transfer solutions converge on specially aligned raw coordinate indices.
- It does **not** establish literal “preservation through fine-tuning” of the scratch base model’s units.
- A literal preservation design would compare the five-dataset pretrained parent immediately before RML fine-tuning with its own fine-tuned descendant.
- Raw-coordinate cosine and index overlap also inherit this cross-model basis assumption; hidden coordinates are not generally permutation-invariant identifiers.

This is probably the most important design issue to raise with the professor.

## Exact model location and intervention

The paper currently calls the target an “FFN output.” The executable intervention is more precise:

```text
audio waveform
  -> audio CNN (`A`)
  -> `a_flatten`
  -> `a_transformer(..., get_cls=True)`
  -> 64-d post-Transformer audio CLS vector   <-- hook and overwrite here
  -> `a_out`: Linear(64, 6)
  -> weighted fusion with text/video logits
```

`WrappedTransformerEncoder.forward(..., get_cls=True)` returns `inputs[0]`, the final CLS coordinate vector. `MME2E.forward` passes this directly into `a_out`. The notebook registers its hook on `model.a_transformer`. This is the output of the whole audio Transformer stack, not the output of an internal Transformer FFN sublayer. (`Model/Dig-Data_Model-Main/src/models/transformer_encoder.py`, lines 22–45; `Model/Dig-Data_Model-Main/src/models/e2e.py`, lines 76–83 and 117–128; notebook cells `ec7b1411`, `a344395c`.)

Safe paper term: **64-dimensional post-Transformer audio CLS representation feeding the audio classifier head**.

“Neuron” in this section means one raw coordinate of that vector. It should not be described as a biological neuron, a guaranteed monosemantic concept, or a whole learned circuit.

## Why audio, despite text having the largest aggregate SHAP mass?

The DeepSHAP feature vector has layout `[text CLS: 1024 | video CLS: 64 | audio CLS: 64]`. (`checkpoints/README.md`, DeepSHAP pickles; `docs/provenance/upstream_SHAP_analysis.ipynb`, feature-name cell; notebook cell `e4843b29`.)

| Model | View | Text | Video | Audio | Winner |
|---|---:|---:|---:|---:|---|
| Base | aggregate `sum(|phi|)` | 0.258486 | 0.010561 | 0.084228 | text |
| Base | per-coordinate `mean(|phi|)` | 0.000252 | 0.000165 | 0.001316 | audio, 5.21x |
| Fine-tuned | aggregate `sum(|phi|)` | 0.265253 | 0.006042 | 0.114995 | text |
| Fine-tuned | per-coordinate `mean(|phi|)` | 0.000259 | 0.000094 | 0.001797 | audio, 6.94x |

Audio is the per-coordinate winner for every output class in both models and still carries 23.8%/29.8% of aggregate mass. Text’s aggregate advantage is partly the arithmetic consequence of having sixteen times as many coordinates. (`results/day0_dominant_modality.json`; ADR 0005, Decision 1.)

These are not contradictory measurements: `mean(|phi|)` equals `sum(|phi|)` divided by width. Aggregate mass answers how much the whole branch contributes; per-coordinate attribution answers which branch has the strongest average coordinate. Since VI-D intervenes on equal numbers of coordinates, the per-coordinate view is a coherent target-selection rule.

The co-author decision is how to reconcile this VI-D carve-out with Section IV-C and Table IV, which define dominant modality by aggregate contribution. The latest draft wisely scopes per-coordinate selection to VI-D, but the wording must stay explicit.

## Data, checkpoints, and fidelity

### Fixed dataset and labels

- RML splits: 518 train, 58 validation, 144 test; the test set is balanced at 24 utterances per class.
- Label order is fixed by upstream code: anger, disgust, fear, happiness, sadness, surprise.
- Labels/text come from the merged-corpus `meta_one_hot_label_six_categories.pkl`, not the incompatible RML archive `meta.pkl`.
- The current audio path uses the training-era 16 kHz files; using the shipped 44.1 kHz route reproduces neither checkpoint.

Sources: `data/README.md`; `Model/Dig-Data_Model-Main/src/datasets.py`, `getEmotionDict`; ADR 0004; `journals/2026-08-11-session1.md`; `journals/2026-08-12-session1.md`.

### Fidelity and harness gates

The cold v3 run reproduced the logged test accuracy exactly:

- Scratch base: 76.39%.
- Fine-tuned: 79.86%.

Checkpoint identity is SHA-256 checked, every sampled input path is checked, and the SHAP pickle’s audio block correlates at `r = 1.000000` with freshly extracted audio CLS activations. A full 64-coordinate clamp reduces base accuracy by 18.06 points, while the Day 0 random-five mechanical control changes it by 0.00 points. All clamped samples share the identical 64-vector, proving the hook writes through. (`checkpoints/README.md`; notebook cells `c5b90ba6`, `e4843b29`, `ec7b1411`, `a344395c`; `journals/2026-08-12-session1.md`, Findings.)

The Day 0 random-five check is a harness sanity check, not the Week 3 statistical null.

### Speaker-split caveat

Every `s1`–`s8` prefix appears in each of the train, validation, and test split files. The fixed evaluation is therefore utterance-stratified rather than subject/speaker-disjoint. Speaker-specific cues may cross splits, so results should not be sold as unseen-speaker generalization. VI-D correctly reuses the original split to preserve checkpoint fidelity; this is a limitation, not an accusation of protocol violation. (The three `Final_{train,valid,test}_split_six_categories_RML.txt` files under `Model/Dig-Data_Model-Main/data/data_split/all_single_label_six_category/with_valid/`.)

## The experiment, step by step

### 1. Extract candidate representation

For each checkpoint and split, the notebook caches the audio CLS vector plus text and video logits. Base and fine-tuned rows and labels must match exactly. Activations are `(N, 64)`. Cache provenance includes checkpoint hashes and split-ID hashes. (Notebook cell `a344395c`; `checkpoints/README.md`, activations.)

### 2. Fit sparse probes

For each model and emotion class:

1. Standardize each of the 64 training activation coordinates using `StandardScaler` fit on train only.
2. Convert the six-way label into class-versus-rest.
3. Fit `LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=2000)`.
4. Rank coordinates by absolute standardized probe coefficient; break ties by index.
5. Require at least five nonzero coefficients.

The coefficient magnitude measures sparse linear readability per standardized coordinate. It is a candidate-selection rule, not causal importance. Fine-tuned anger has exactly five nonzero coefficients, so its top five exhaust the probe’s entire sparse support. (Notebook cell `913e374b` and its output.)

Held-out test AUC is a sanity check:

| Class | Base AUC | Fine-tuned AUC |
|---|---:|---:|
| Anger | 0.902 | 0.949 |
| Disgust | 0.910 | 0.920 |
| Fear | 0.767 | 0.806 |
| Happiness | 0.849 | 0.904 |
| Sadness | 0.961 | 0.961 |
| Surprise | 0.911 | 0.921 |
| **Mean** | **0.884** | **0.910** |

No class is below 0.55. (`results/week1_probe_auc.json`; notebook cell `59dbea6b`.)

### 3. Mean-ablate the selected coordinates

For model `m`, coordinate set `S`, and test sample `i`:

```text
a_i'[d] = train_mean_m[d]  if d in S
          a_i[d]            otherwise
```

Each model uses its own training-set mean. Cross-model ablation relocates indices, not the base model’s activation values. (Notebook cells `a518c0cb`, `2230367c`.)

Mean replacement avoids making zero a privileged baseline and usually reduces marginal shift. It does **not** prove the hybrid 64-vector lies on the joint training manifold; correlations among coordinates can still be broken. Say “mean-imputation baseline,” not “guaranteed in-distribution.”

### 4. Measure a continuous effect

For sample `i` with true label `y_i`:

```text
delta p_i = 100 * [softmax(z_clean_i)[y_i] - softmax(z_ablated_i)[y_i]]
```

The class effect is the mean `delta p_i` over that class’s 24 test samples. A positive number means ablation lowers confidence assigned to the correct class. It does not necessarily flip the argmax and is not an accuracy improvement/degradation in the usual sense.

Per-class accuracy is secondary because `100 / 24 = 4.1667` percentage points is its minimum nonzero step. Every top-five probability effect is smaller than one such step. Decision-margin drop is also recorded in artifacts as a calibration-robust secondary view, but the paper tables use probability drop. (`results/measurement_resolution.json`; notebook cell `a518c0cb`; `docs/v3-downstream-statistics-audit.md`, Sections 2 and 6.1.)

Softmax probability is sensitive to logit scale/calibration. Within-model ablation contrasts are still meaningful, but cross-model probability ratios can inherit different calibration. The random baseline controls broad differential sensitivity empirically; it does not prove calibration equivalence.

### 5. Bootstrap uncertainty

The notebook forms clean-minus-ablated differences per sample before resampling, then resamples within each class. It uses 2,000 percentile-bootstrap draws with seed 20260811, and resets the same stream across conditions to use common random numbers. These are nominal per-comparison 95% intervals. (`results/measurement_resolution.json`; notebook cell `a518c0cb`.)

The current `Key Decisions.pdf` says 1,000 resamples. That is stale. The executable notebook, results metadata, latest draft, and journals all say **2,000**; 2,000 is authoritative.

### 6. Define class concentration

For target class `c`:

```text
selectivity(c) = target probability drop
                 / mean over other classes of |probability drop|
```

The binary label requires a resolvable target effect and ratio at least 2.5. The denominator is used only if at least one non-target class has an interval excluding zero and the mean absolute collateral exceeds 0.10 pp; no epsilon is substituted. (`results/measurement_resolution.json`; notebook `selectivity_verdict` in cell `a518c0cb`.)

The numerical threshold 2.5 was chosen in ADR 0001 for an accuracy-drop ratio and later inherited when the primary metric was repaired to probability drop. It is therefore an earlier chosen number, but the full probability-scale decision rule was not preregistered. The 0.10 floor and null were remedial additions after the downstream audit.

## Tables VII and VIII: what the results say

### Scratch base model

| Class | Top five | Target drop pp [nominal 95% CI] | Mean abs other drop | Ratio | Meets 2.5? |
|---|---|---:|---:|---:|---|
| Anger | 20, 3, 39, 27, 53 | 4.33 [3.12, 5.56] | 0.41 | 10.66 | Yes |
| Disgust | 62, 9, 32, 6, 36 | 2.30 [1.74, 2.84] | 0.73 | 3.14 | Yes |
| Fear | 38, 11, 55, 8, 54 | 0.64 [0.14, 1.10] | 0.58 | 1.10 | No |
| Happiness | 63, 22, 36, 58, 24 | 2.04 [1.11, 2.95] | 0.82 | 2.48 | No |
| Sadness | 54, 15, 43, 61, 60 | 4.06 [2.98, 5.15] | 0.79 | 5.14 | Yes |
| Surprise | 13, 28, 38, 42, 53 | 2.29 [1.48, 3.06] | 0.52 | 4.44 | Yes |

### Fine-tuned model, using its own probe ranking

| Class | Top five | Target drop pp [nominal 95% CI] | Mean abs other drop | Ratio | Meets 2.5? |
|---|---|---:|---:|---:|---|
| Anger | 3, 55, 54, 53, 27 | 5.77 [4.46, 7.19] | 0.54 | 10.69 | Yes |
| Disgust | 41, 11, 52, 40, 36 | 1.92 [1.29, 2.53] | 1.01 | 1.89 | No |
| Fear | 40, 29, 54, 11, 6 | 1.69 [0.77, 2.59] | 0.99 | 1.70 | No |
| Happiness | 12, 51, 62, 5, 55 | 2.86 [1.82, 3.89] | 1.43 | 2.01 | No |
| Sadness | 60, 0, 9, 32, 33 | 6.81 [4.69, 8.75] | 1.19 | 5.70 | Yes |
| Surprise | 13, 28, 17, 61, 53 | 2.86 [1.19, 4.46] | 0.79 | 3.64 | Yes |

Sources: `results/table_vii_base_ablation_k5.csv`; `results/table_viii_finetuned_ablation_k5.csv`.

The exact safe reading:

- All twelve tested sets have a positive target-confidence effect under nominal, uncorrected intervals.
- Four base and three fine-tuned rows pass the chosen concentration threshold.
- Anger and sadness pass in both models; surprise also passes in both; disgust loses concentration after fine-tuning.
- Fear has the weakest probe AUC and weakest selectivity.
- Happiness at 2.48 in the base model demonstrates threshold sensitivity.
- Fine-tuned collateral is generally larger, consistent with greater global audio sensitivity.

Do not say all top-five sets are “class-selective”; all are target-load-bearing, but only seven of twelve pass the selected concentration rule.

## The k sweep: correct the monotonicity story

The sweep uses `k = {1, 3, 5, 10, 16, 32, 48, 64}`. Target damage generally grows as more coordinates are removed, and concentration tends to decline for large `k`. That supports a distinction between small class-concentrated sets and broad branch damage.

The current draft and `Results.pdf` overstate this:

- “Monotonic for all twelve” is false. Base anger drops from 4.328570 pp at `k=5` to 4.307119 pp at `k=10`, a 0.021451 pp dip. Eleven of twelve sequences are nondecreasing over the sampled k values; base anger is nearly flat but not monotone. (`results/week2_base_ablation_sweep.json`.)
- `k=5` is not “mathematically optimal.” It was the protocol/ADR primary reporting point, and it offers a useful small-set tradeoff. No objective function was optimized over k.
- Say “generally increasing dose response with concentration tending to weaken,” not “strictly monotonic” or “optimal.”

## Probe-ranking validation: the selection rule is not a causal ranking

Day 13b ablates each coordinate individually and compares the probe-selected top five with the five largest measured single-coordinate target effects.

- In all twelve model-class pairs, probe-top-five target damage is below 80% of damage from the measured reference set.
- Ratios range from 0.159 to 0.731.
- Spearman correlation between probe coefficient magnitude and measured damage over the top-16 union is negative in eleven of twelve pairs and approximately zero in the remaining pair.

Sources: `results/table_x_probe_rank_validation.csv`; notebook cell `81cb0e0a`.

This does not make the probes useless. A sparse linear readout and downstream causal necessity answer different questions. It does mean:

- Call them **probe-selected candidate coordinates**, not “the neurons the model uses.”
- Do not claim absolute probe weight identifies causally strongest units.
- Do not call the reported selectivity ratio a proven lower bound. The damage-ranked reference maximizes measured individual target damage, not joint five-set damage; single-coordinate effects are nonadditive, and its non-target collateral was not used to establish a better selectivity ratio.

The defensible positive point is simply that even this imperfect, train-only candidate rule found sets with resolvable model-internal effects.

## Table IX: cross-model alignment and its null

For each class, the notebook:

1. Takes the scratch base model’s top-five indices.
2. Ablates those indices in the base model using the base training mean.
3. Ablates the same indices in the fine-tuned model using the fine-tuned training mean.
4. Defines `R = fine-tuned target drop / base target drop`.
5. Compares R with `R_null`, estimated from 2,000 random five-coordinate subsets shared across both models.

The null uses seed 20260812, distinct from the sample bootstrap. It reports a ratio of mean fine-tuned drop to mean base drop across subsets, avoiding unstable per-subset ratios with near-zero denominators. (`results/week3_random_ablation_null.json`; notebook cells `11392f30`, `bbc121d8`; ADR 0006.)

### Results

| Class | Base drop | FT drop under base indices | R [95% CI] | R-null [95% CI] | Implemented verdict |
|---|---:|---:|---:|---:|---|
| Anger | 4.33 | 7.32 | 1.69 [1.20, 2.41] | 1.42 [1.40, 1.43] | Does not clear rule |
| Disgust | 2.30 | 4.79 | 2.08 [1.47, 2.90] | 2.01 [1.99, 2.02] | Does not clear rule |
| Fear | 0.64 | 2.77 | 4.34 [2.68, 9.34] | 4.01 [3.90, 4.14] | Does not clear rule |
| Happiness | 2.04 | 2.85 | 1.39 [0.78, 2.36] | 1.19 [1.16, 1.21] | Does not clear rule |
| Sadness | 4.06 | 5.35 | 1.32 [1.08, 1.72] | 1.21 [1.20, 1.22] | Does not clear rule |
| Surprise | 2.29 | 4.74 | 2.07 [1.55, 2.53] | 1.73 [1.71, 1.74] | Does not clear rule |

Every R point estimate is above its null point estimate, but all R intervals overlap the corresponding null interval. The data are directional but imprecise. Fear illustrates denominator inflation: R=4.34 looks dramatic until compared with random R=4.01. Happiness’s R interval includes values below 1. (`results/week3_transfer_retention.csv`; ADR 0006, Decisions 2–3.)

### What the comparison rule is—and is not

Day 15 calls a result above null only if `R`’s lower confidence endpoint exceeds `R_null`’s upper endpoint. These intervals are obtained separately:

- R’s interval bootstraps the 24 same-class sample rows.
- R-null’s interval bootstraps the ratio of mean effects over 2,000 subsets.

This is a stringent confidence-interval separation heuristic. It is **not** a p-value, a paired interval for `R - R_null`, an equivalence test, or a randomization-tail test. R-null’s tight interval estimates the mean random-subset ratio; it is not the 95% prediction range of individual random-subset ratios.

Safe conclusion: **zero of six classes show a detected cross-model signal above the measured random-ablation baseline under the implemented rule.**

Unsafe conclusions:

- “The units definitively do not transfer.”
- “The null hypothesis is proven.”
- “The models are equivalent to random.”
- “The units were reassigned.”

A stronger follow-up would retain subset-level effects and perform a direct paired contrast or randomization-tail test. Most importantly, literal fine-tuning preservation should use the pretrained parent and its descendant.

The latest draft also says the old `R > 1` framing implied “reassignment.” That is backwards: the withdrawn bands classified the entire `R > 1` region as preservation. ADR 0006 withdrew them because every row landed there and arbitrary coordinates also yield R>1.

## Cosine and top-five overlap

The selectivity vector for class c is computed on training activations:

```text
s_c[d] = mean(a[d] | y=c) - mean(a[d] | y!=c)
```

Matched base/fine-tuned cosine similarities range from 0.960 to 0.990. Each class’s fine-tuned vector is the closest of all six to its corresponding base vector. The notebook reports one global statistic, `mean(matched cosine) - mean(mismatched cosine) = 1.168`; it is not six class-specific gaps. (`results/table_ix_cosine_and_transfer.csv`; notebook cell `78e57917`.)

Top-five overlaps are anger 3, disgust 1, fear 2, happiness 0, sadness 1, surprise 3. Under `Hypergeometric(64,5,5)`, expected overlap is 0.391; nominal tail p-values are 0.0023 for overlap 3, 0.0449 for overlap 2, and 0.343 for overlap 1. These six p-values are not multiplicity-corrected; fear’s nominal 0.0449 should not be advertised as robust across six overlap tests. (`results/week3_transfer_retention.csv`; ADR 0006, Decisions 6–7.)

Interpretation:

- The full class-contrast patterns are highly similar in the chosen raw basis.
- Exact top-ranked indices are much less stable.
- Both are correlational, raw-coordinate comparisons across separate training lineages.
- Neither overrides the direct ablation result.

“Representational geometry is preserved” is stronger than the evidence. Prefer: **class-contrast vectors are highly aligned in the raw coordinate basis**.

## Statistical and generalization limitations to volunteer

1. **Multiplicity:** Twelve target intervals, sixty non-target intervals, and six overlap p-values are nominal and uncorrected. No Holm-adjusted p-value or interval artifact exists. `docs/vi-d-paper-writeup-guidance.md` only speculated that base fear would “most plausibly” change and explicitly says a proper computation needs per-sample arrays not exported to `results/`. The latest draft’s exact Holm claim should be removed unless computed.
2. **Small evaluation:** 24 samples per class; bootstrap precision reflects this fixed empirical split, not new subjects, new model seeds, or training variability.
3. **One trained pair:** no replicate training seeds, so unit/index findings may be seed-specific.
4. **No speaker independence:** `s1`–`s8` appear in both training and testing.
5. **Selection and evaluation reuse:** probes are fit on train, which is good, but the test split is used for AUC, intervention estimates, many k values, and follow-up diagnostics.
6. **Post hoc remediation:** probability drop, denominator guards, rank validation, and the random null were added after problems in the accuracy pipeline were seen. The changes are well motivated and documented, but this is not a preregistered confirmatory study.
7. **Metric calibration:** softmax probability is continuous but calibration-sensitive; accuracy is transparent but too coarse here. Decision margin helps triangulate but is not in the main tables.
8. **Mean ablation:** coordinatewise mean replacement is not guaranteed to remain on the joint activation manifold.
9. **Raw-coordinate dependence:** “neurons” may be polysemantic and are basis-dependent; cross-model indices have no guaranteed identity.
10. **Scope:** RML only, audio only, one post-Transformer layer, six classes, one architecture.
11. **Modality selection:** audio is chosen from attribution on the same RML test representation; the result does not causally compare audio against text/video in matched k interventions.
12. **Causal scope:** the intervention establishes effects on this model’s output, not that training-time modality competition caused RML’s limited transfer gain.

## Claims ladder: what you can safely say

### Strongest supported claims

- The v3 pipeline reproduces the two checkpoints’ logged RML accuracy exactly and verifies hook write-through.
- Audio has the highest per-coordinate DeepSHAP attribution for every class in both checkpoints.
- The audio CLS representation contains strong linearly readable class signal.
- Mean-ablation of each probe-selected top-five set changes target-class true-label confidence; all twelve nominal target intervals exclude zero.
- Class concentration is heterogeneous: 4/6 base and 3/6 fine-tuned rows pass the inherited 2.5 ratio.
- No class clears the implemented cross-model random-baseline separation rule.

### Claims requiring qualification

- “Causal validation”: causal only with respect to activation intervention and model output.
- “Class-selective neurons”: only some sets pass the selected ratio; coordinates can be polysemantic.
- “Geometry preserved”: raw class-contrast vectors are aligned, under a cross-model coordinate-basis assumption.
- “No transfer”: no detected special cross-model alignment under this analysis, not proof of absence.

### Claims to avoid

- “Mathematically proven.”
- “Definitive null.”
- “k=5 is optimal.”
- “The dose response is monotonic for all twelve.”
- “Mean ablation guarantees in-distribution activations.”
- “Probe weights identify the causally strongest neurons.”
- “Selectivity is a proven lower bound.”
- “Holm correction changes only base fear.”
- “The base neurons were preserved/reassigned through fine-tuning.”

## Likely professor questions and defensible answers

### Why is this causal if the candidates came from a correlational probe?

The selection rule is correlational, but the evaluation is interventional: overwrite selected activations and measure the resulting output change. The intervention supports a causal effect of the tested set within the model. It does not retroactively make probe coefficient magnitude a causal ranking.

### Why not ablate text, since the paper says text is dominant?

Section IV-C’s aggregate question and VI-D’s equal-k coordinate question differ. Text wins total mass with 1,024 dimensions; audio wins average contribution per coordinate by 5.21x/6.94x and in all classes. Audio is therefore defensible for a tractable coordinate intervention, but the carve-out must be explicit and does not alter Table IV silently.

### What exactly did you hook?

`model.a_transformer`’s returned `(batch,64)` final audio CLS vector, immediately before `a_out`. Calling it an internal FFN output is imprecise.

### Why L1 logistic regression?

L1 creates a sparse class-versus-rest readout and an ordered candidate list. Inputs are standardized, so coefficients are comparable per standard deviation. Held-out AUC verifies readability. Rank validation shows it is not a causal-damage ranking.

### Why use absolute probe weights?

Both positive and negative coefficients carry class-discriminative signal. Ablation tests necessity regardless of coefficient sign. Absolute magnitude is the pre-existing candidate rule, not proof of direction or causality.

### Is there train-test leakage?

The scaler, probes, rankings, and replacement means use 518 train samples only; effects use the held-out 144 test samples. There is no parameter fitting on test. However, the same test split supports AUC, all ablations, k inspection, and follow-up diagnostics, so it is exploratory reuse rather than a fresh confirmatory test.

### Why probability drop instead of accuracy?

Each class has 24 test examples, so accuracy changes in 4.17-point jumps and discards confidence movement that does not flip an argmax. Probability drop uses every example continuously. Accuracy remains in `legacy_acc_*` columns for transparency.

### Is probability drop “performance”?

Not exactly. It is a change in confidence assigned to the true class. It is more sensitive than accuracy here, but depends on calibration. The paper should name it precisely.

### Why mean instead of zero?

Zero may be atypical for a learned coordinate. The training mean provides a neutral marginal baseline without leaking test statistics. It reduces one source of distribution shift but does not guarantee the full vector remains on-manifold.

### Why the mean absolute non-target effect?

Signed collateral can cancel when one non-target class improves and another worsens. Mean absolute movement captures total off-target disruption. That makes the ratio a concentration measure rather than a net-effect measure.

### Why 2.5?

It was fixed in ADR 0001 as the study’s primary concentration threshold and retained after moving from coarse accuracy to probability. It is a convention, not a natural constant; raw ratios are reported, and happiness at 2.48 makes threshold sensitivity visible.

### Why k=5?

It was the protocol and ADR’s primary reporting size: small enough to test concentration and large enough to avoid relying on one coordinate. The sweep contextualizes it. No optimization establishes k=5 as uniquely best.

### Is the k sweep monotone?

Almost, not strictly. Eleven of twelve target-drop sequences are nondecreasing over sampled k. Base anger dips 0.021 pp from k=5 to k=10. The correct statement is a generally increasing dose response.

### What is the strongest Week 2 result?

Anger: ratios 10.66 base and 10.69 fine-tuned with clear target effects. Sadness is also consistently concentrated. Fear is weakest and has the lowest AUC.

### Do all twelve sets contain class-selective neurons?

All twelve affect target confidence under nominal intervals. Only seven pass the 2.5 concentration rule. “Load-bearing” and “class-concentrated” are different claims.

### What does the rank-validation failure mean?

Probe readability is not downstream necessity. Across all twelve pairs, other coordinates selected by measured individual damage hurt the target more. It means the probe rule is not a causal importance ranking, not that the tested effects vanish.

### Why is the fast path exact?

With `mod='tav'`, final logits are a linear fusion of text, video, and audio-head logits. Only `a_cls` changes, so cached text/video logits plus the audio vector and linear head determine the ablated output. Real hooked forwards verify probability differences below `1e-3`; the maximum observed was `9.43e-06`. (`results/week2_fast_path_verification.json`; model `e2e.py`.)

### What is R supposed to mean?

R divides fine-tuned damage by base damage under the scratch base model’s indices. R>1 alone means the fine-tuned model is more damageable at those coordinates; it does not establish special index alignment because arbitrary coordinates also yield R>1.

### Why ratio of means for R-null?

Individual random subsets can have near-zero base damage, making per-subset ratios explode. Averaging drops before division provides a stable global-sensitivity reference without filtering inconvenient denominators.

### Is zero of six statistically nonsignificant?

The implemented endpoint-separation rule is not a direct p-value test. Say zero of six clear the rule and no special signal is detected. All six point estimates are directionally above null, so a weak effect remains possible with more data.

### Does high cosine prove preservation?

No. It shows high raw-basis alignment of class-contrast vectors. The models come from different lineages, exact top-five overlap is unstable, and the causal comparison does not clear its null baseline.

### Why compare the scratch base to fine-tuned if it is not the ancestor?

That comparison follows the supplied protocol and asks whether two RML solutions converge on aligned coordinate indices. It should be relabeled accordingly. For literal preservation across fine-tuning, add the pretrained-parent checkpoint.

### What explains RML’s limited gain, then?

VI-D does not establish a unit-level explanation. It shows model-internal audio dependence and no detected special raw-coordinate alignment. Any claim that reassignment caused the limited gain would go beyond the data.

### Did you correct multiple comparisons?

No. Intervals and overlap p-values are nominal. No valid Holm artifact exists, so the draft should disclose this and remove its exact counterfactual claim.

### Does the random null solve every confound?

No. It addresses the fact that the fine-tuned model is globally more sensitive to five-coordinate audio ablation. It does not solve calibration, cross-lineage coordinate identity, multiple testing, or small-sample uncertainty, and its CI-overlap decision is not a direct randomization test.

### Can this generalize to unseen speakers?

Not from the current split. All s1–s8 prefixes occur in train and test. This is a fixed-split RML model analysis, not speaker-independent validation.

### Why should reviewers trust the numbers after v1/v2 failed?

V1/v2 are explicitly retracted because they used mismatched metadata/data. V3 adds hash identity, canonical labels/text, a 16 kHz fidelity route, halting accuracy gates, path checks, SHAP-to-activation correlation, hook write-through, fast/slow agreement, and a cold reproduction in which eight of nine artifacts were byte-identical and the ninth differed only by GPU float noise. (ADR 0004; `journals/2026-08-12-session1.md`.)

## Prioritized decisions for tomorrow’s meeting

### Must settle before submission

1. **Rename Table IX’s scientific question.** Decide whether to present it as exploratory cross-model coordinate alignment or rerun the intended preservation test using the pretrained parent versus its fine-tuned descendant.
2. **Correct the hook description.** Replace “FFN output” with the exact post-Transformer audio CLS representation.
3. **Choose statistically honest transfer wording.** “No detected signal above the measured random baseline under the implemented rule,” not “definitive no transfer.”
4. **Remove uncomputed Holm claim.** Either compute adjusted inference from saved per-sample effects or disclose nominal intervals only.
5. **Correct dose-response language.** Remove “monotonic all twelve,” “optimal k=5,” and the unsupported selectivity lower-bound claim.
6. **Resolve modality terminology.** Keep VI-C aggregate dominance and explicitly scope VI-D’s per-coordinate audio target.

### Strongly recommended

7. Correct the stale 1,000-bootstrap statement in `Key Decisions.pdf` to 2,000.
8. Replace “in-distribution” guarantee with “training-mean imputation intended to reduce zero-baseline shift.”
9. Add the fixed-split/speaker-overlap and single-seed limitations.
10. Decide whether to strengthen the null analysis with a direct contrast/randomization-tail test.

### Best follow-up experiment if time permits

Obtain the five-dataset pretrained parent checkpoint that is named in the fine-tuning log. Extract its audio CLS activations on RML, align rows, fit the same train-only selection procedure, and compare that checkpoint directly with its fine-tuned descendant. This tests literal within-lineage change. If retaining the random null, export subset-level effects and use a direct paired contrast or empirical tail probability.

## One-minute closing statement

> My contribution gives Section VI-C a limited interventional check. On the fixed RML split, probe-selected audio CLS coordinates causally affect the model’s true-class confidence, and that effect is class-concentrated for several emotions. The study does not establish a neuron-level mechanism for RML’s limited transfer gain: the scratch and transfer-learned models do not show detectable special coordinate alignment beyond the random sensitivity baseline under our current rule. Because they are different training lineages, I would present that as exploratory cross-model alignment and propose a pretrained-parent versus fine-tuned-descendant comparison as the correct preservation follow-up. I would also disclose the nominal inference, single seed, speaker-overlapping split, and probe-ranking limitation.

## Source map

- Current paper text and tables: `Latest_Paper_Draft.docx`, Sections IV-A/C, V-C, VI-C/D, VIII.
- Professor’s requested protocol: `experiment_protocol[From Prof KC Lan].docx`.
- Executable implementation: `multimodal-causal-ablation-v3.ipynb`, especially cells `e4843b29`, `ec7b1411`, `a344395c`, `913e374b`, `59dbea6b`, `a518c0cb`, `7fce4fae`, `78e57917`, `81cb0e0a`, `2230367c`, `11392f30`, `bbc121d8`.
- Upstream architecture: `Model/Dig-Data_Model-Main/src/models/e2e.py`; `src/models/transformer_encoder.py`.
- Checkpoint lineages and targets: `docs/provenance/RML_origin_training_log.txt`; `RML_finetune_training_log.txt`; `checkpoints/README.md`.
- Canonical data and split: `data/README.md`; the three RML split files under `Model/Dig-Data_Model-Main/data/data_split/all_single_label_six_category/with_valid/`.
- Final numeric artifacts: `results/day0_dominant_modality.json`; `week1_probe_auc.json`; `table_vii_base_ablation_k5.csv`; `table_viii_finetuned_ablation_k5.csv`; `table_ix_cosine_and_transfer.csv`; `table_x_probe_rank_validation.csv`; `week3_random_ablation_null.json`; `week3_transfer_retention.csv`; `measurement_resolution.json`.
- Governing corrections: ADR 0004, ADR 0005, ADR 0006; `docs/v3-downstream-statistics-audit.md`; `journals/2026-08-11-session{2,3}.md`; `journals/2026-08-12-session1.md`; `journals/ERRATA.md`.

