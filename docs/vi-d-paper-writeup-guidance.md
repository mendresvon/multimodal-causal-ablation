# Section VI-D: filled tables, rigor verdict, and write-up guidance

Date: 2026-08-12. Source artifacts: `results/table_vii_base_ablation_k5.csv`,
`results/table_viii_finetuned_ablation_k5.csv`, `results/table_ix_cosine_and_transfer.csv`,
`results/table_x_probe_rank_validation.csv`, `results/measurement_resolution.json`,
`results/week3_random_ablation_null.json`, `results/week2_{base,finetuned}_ablation_sweep.json`.
Notebook: `multimodal-causal-ablation-v3.ipynb`, run end to end from cold on 2026-08-12
(journal: `journals/2026-08-12-session1.md`).

Row order below follows the paper's draft tables: Anger, Disgust, Fear, Sadness, Surprise,
Happiness. The CSVs are alphabetical; these tables were filled by class name, not by position.

---

## 1. Headline answer

Section VI-D's experiment ran to completion and produced usable, defensible results. The
paper is **not yet ready for submission**, for reasons that are all editorial rather than
experimental: four placeholder blocks in VI-D are unwritten, two table headers ask for a
quantity the measurement cannot resolve, Table IX's caption encodes a framing the data
refutes, and Section IV-C's definition of "dominant modality" does not match what VI-D
actually hooked. Every one of those is fixable in text without touching the GPU.

---

## 2. The measurement change that has to happen first

The draft Tables VII and VIII ask for **Δ accuracy**. The RML test split has 144 samples,
24 per class, so per-class argmax accuracy moves only in steps of 100/24 = **4.17 pp**. Every
effect this experiment measures is smaller than one step. Filling the tables as headed would
report the measurement floor as if it were a result.

The concrete damage, straight from the artifacts:

- **Base / anger**: accuracy drop **0.00 pp**, while the probability drop is **+4.33 pp**
  with a 95% CI of [3.12, 5.56] that excludes zero. Reported as accuracy, the strongest
  selective effect in the study reads as "no effect."
- **Base / disgust**: mean absolute non-target accuracy drop is **0.00 pp**, so the
  selectivity ratio printed from accuracy is **infinite**.

The replacement measure, already computed throughout the notebook, is the **mean drop in the
true-class softmax probability, in percentage points**, with a 95% CI from a paired,
class-stratified bootstrap (2000 draws, seed 20260811, per-sample differences formed before
resampling, common random numbers across conditions). Decision margin in logit units is
recorded alongside it. The argmax numbers survive as `legacy_acc_*` columns in every CSV and
should be kept visible in a footnote, so the switch reads as a resolution argument rather
than as hiding an inconvenient result.

**Edit sites this change touches** (not just the table headers):

1. VI-D §1 Method, the sentence "we measured the resulting change in per-class test accuracy
   relative to the unablated model."
2. The sentence immediately after it: "A causally selective feature set produces a large
   accuracy drop on its own class and a small drop elsewhere."
3. Table VII header, Table VIII header.
4. Table IX header and caption.
5. The prose paragraph introducing Tables VII/VIII (paper.txt line 125).
6. The prose paragraph introducing Table IX (paper.txt line 147).
7. Section V-C (Evaluation Metrics), which lists accuracy/precision/recall/F1 only — add one
   sentence that VI-D additionally reports true-class probability and decision margin,
   because accuracy on a 24-sample-per-class split cannot resolve single-neuron interventions.

---

## 3. Table VII — filled

**CAUSAL ABLATION SELECTIVITY, BASE RML MODEL (TOP-5 NEURONS PER CLASS)**

Suggested header: `Class Ablated (top-5 neurons)` | `Δ P(true), target class (pp) [95% CI]` |
`Mean |Δ P(true)|, other classes (pp)` | `Selectivity ratio` | `Selective?`

| Class Ablated (top-5 neurons) | Δ P(true), target (pp) [95% CI] | Mean abs Δ P(true), others (pp) | Selectivity ratio | Selective? |
|---|---|---|---|---|
| Anger (20, 3, 39, 27, 53) | +4.33 [3.12, 5.56] | 0.41 | 10.66 | Yes |
| Disgust (62, 9, 32, 6, 36) | +2.30 [1.74, 2.84] | 0.73 | 3.14 | Yes |
| Fear (38, 11, 55, 8, 54) | +0.64 [0.14, 1.10] | 0.58 | 1.10 | No |
| Sadness (54, 15, 43, 61, 60) | +4.06 [2.98, 5.15] | 0.79 | 5.14 | Yes |
| Surprise (13, 28, 38, 42, 53) | +2.29 [1.48, 3.06] | 0.52 | 4.44 | Yes |
| Happiness (63, 22, 36, 58, 24) | +2.04 [1.11, 2.95] | 0.82 | 2.48 | No |

All six target CIs exclude zero. Selectivity threshold is 2.5; happiness at 2.48 falls just
under it, which is worth one clause in the prose rather than silence.

---

## 4. Table VIII — filled

**CAUSAL ABLATION SELECTIVITY, FINE-TUNED RML MODEL (TOP-5 NEURONS PER CLASS)**

| Class Ablated (top-5 neurons) | Δ P(true), target (pp) [95% CI] | Mean abs Δ P(true), others (pp) | Selectivity ratio | Selective? |
|---|---|---|---|---|
| Anger (3, 55, 54, 53, 27) | +5.77 [4.46, 7.19] | 0.54 | 10.69 | Yes |
| Disgust (41, 11, 52, 40, 36) | +1.92 [1.29, 2.53] | 1.01 | 1.89 | No |
| Fear (40, 29, 54, 11, 6) | +1.69 [0.77, 2.59] | 0.99 | 1.70 | No |
| Sadness (60, 0, 9, 32, 33) | +6.81 [4.69, 8.75] | 1.19 | 5.70 | Yes |
| Surprise (13, 28, 17, 61, 53) | +2.86 [1.19, 4.46] | 0.79 | 3.64 | Yes |
| Happiness (12, 51, 62, 5, 55) | +2.86 [1.82, 3.89] | 1.43 | 2.01 | No |

All six target CIs exclude zero. The fine-tuned model's non-target collateral is uniformly
larger than the base model's (0.54–1.43 pp against 0.41–0.82 pp), which is the global
sensitivity difference that Table IX has to control for.

---

## 5. Table IX — filled, with a changed caption

The draft caption reads **PRESERVATION VS. REASSIGNMENT OF CLASS-SELECTIVE NEURONS ACROSS
FINE-TUNING**, and the Verdict column is drafted to receive "Preserved" or "Reassigned."
That dichotomy was withdrawn (`docs/adr/0006-week3-transfer-null-and-band-withdrawal.md`)
because it has no zero point: the fine-tuned model loses more probability than the base model
under **five arbitrary dimensions**, so a ratio above 1 is what noise produces. The correct
comparison is against a random-ablation null, not against 1.

Suggested caption: **CROSS-MODEL ABLATION OF BASE-MODEL CLASS-SELECTIVE UNITS, RML, AGAINST A
RANDOM 5-DIMENSION NULL.**

| Class | Cosine sim. (base vs FT selectivity) | Base top-5 overlap w/ FT top-5 (p) | Δ P(true) in FT, base neurons ablated (pp) | R (FT/base drop) [95% CI] | R_null [95% CI] | Verdict |
|---|---|---|---|---|---|---|
| Anger | 0.970 | 3 (p = 0.002) | +7.32 | 1.69 [1.20, 2.41] | 1.42 [1.40, 1.43] | Not above null |
| Disgust | 0.960 | 1 (p = 0.343) | +4.79 | 2.08 [1.47, 2.90] | 2.01 [1.99, 2.02] | Not above null |
| Fear | 0.966 | 2 (p = 0.045) | +2.77 | 4.34 [2.68, 9.34] | 4.01 [3.90, 4.14] | Not above null |
| Sadness | 0.982 | 1 (p = 0.343) | +5.35 | 1.32 [1.08, 1.72] | 1.21 [1.20, 1.22] | Not above null |
| Surprise | 0.972 | 3 (p = 0.002) | +4.74 | 2.07 [1.55, 2.53] | 1.73 [1.71, 1.74] | Not above null |
| Happiness | 0.990 | 0 (p = 1.000) | +2.85 | 1.39 [0.78, 2.36] | 1.19 [1.16, 1.21] | Not above null |

Notes for the caption or a footnote:

- The Δ P(true) column is a real, resolvable effect in every row — all six CIs exclude zero.
  The units still do damage in the fine-tuned model. What fails is the claim that they do
  **more** damage than five arbitrary dimensions would.
- Overlap null is Hypergeometric(64, 5, 5); expected overlap by chance is 0.39. An overlap of
  1 has p = 0.343, i.e. at chance. Three classes (anger, surprise, fear) are above chance.
- Matched-class cosine minus mismatched-class cosine is **+1.168** for every class, so the
  mismatched baseline sits near −0.2. The high cosine values are not an artifact of every
  vector pointing the same way.
- R_null is 2000 random 5-dimension subsets (seed 20260812, deliberately a different stream
  from the sample bootstrap), the same subsets in both models, estimated as a ratio of means.

---

## 6. What to write in the four placeholders

### VI-D §2, "Causal ablation results"

Report a **mixed positive**, not a null.

All twelve top-5 sets produce a target-class probability drop whose 95% CI excludes zero, so
the ranked units are causally load-bearing in both models. Selectivity is class-dependent:
four of six classes in the base model (anger, disgust, sadness, surprise) and three of six in
the fine-tuned model (anger, sadness, surprise) exceed the 2.5x threshold, with anger and
sadness selective in both. Fear is the consistent exception — 1.10 in the base model, 1.70
fine-tuned — and its probe AUC is also the lowest of the six (0.767 base, 0.806 fine-tuned),
so the ranking has the least to work with there. Happiness sits at 2.48 base and 2.01
fine-tuned, just below threshold in both.

Then state the dose-response, which is the strongest single piece of evidence in the section
and is currently unused: sweeping k over {1, 3, 5, 10, 16, 32, 48, 64}, the target-class drop
increases monotonically in k for **all twelve class-model pairs**, while the selectivity ratio
decays toward ~1 as k grows (base/anger: 8.68 at k=1, 10.66 at k=5, 1.40 at k=64). Small
probe-ranked sets are class-concentrated; large sets are generically useful. This makes k=5 a
principled operating point rather than an arbitrary protocol constant. The figure already
exists at `figures/dose_response_k_sweep.png` and should go in the paper.

Finally, the honest caveat, which is also in the artifacts
(`results/table_x_probe_rank_validation.csv`): in all twelve model-class pairs, the probe's
top-5 does **less than 80%** of the damage that the measured causally-strongest top-5 does
(ratios 0.16–0.73; Spearman rho between probe weight and measured damage over the top-16 union
is near zero or negative in eleven of twelve). The absolute-probe-weight ranking is a
correlational **selection rule**, not an identification of the causally most important units.
The good-faith reading, which should be stated: because a suboptimal set still produced
resolvable, class-concentrated damage, the selectivity estimates are a **lower bound** on what
a causally-optimal set would show.

### VI-D §3, "Preservation versus reassignment across fine-tuning"

This is where the draft's suggested mechanistic story must **not** be written. The placeholder
proposes that if the units were reassigned, this explains RML's limited gain. The data does
not support writing that, and it does not support the preservation branch either.

What to write instead — two findings that point in different directions and should be reported
as such:

1. **The representational geometry is preserved.** Matched-class cosine similarity between
   base and fine-tuned selectivity vectors is 0.960–0.990, against a mismatched-class baseline
   1.168 lower. Fine-tuning did not rotate the class structure of the audio CLS space.
2. **The specific units are not demonstrably more load-bearing after fine-tuning than
   arbitrary units are.** Ablating the base model's top-5 in the fine-tuned model produces a
   resolvable drop in every class, but the base-to-fine-tuned ratio R fails to separate from a
   random 5-dimension null in **all six classes**. The fine-tuned model is globally more
   sensitive to audio ablation, and that alone accounts for R > 1.

The conclusion to state: the preservation-versus-reassignment question, posed at the level of
individual units, **is not answerable with this measurement** on this dataset. Neither branch
is supported. RML's limited fine-tuning gain therefore does not get a unit-level mechanistic
explanation here, and the paper should say so rather than choose a branch.

### VI-D closing sentence

State that the intervention confirms the audio CLS units carry causally load-bearing,
partially class-selective information in both models, that the cross-model transfer question
returns a null against its own control, and that the validation is scoped to one dataset, one
modality, one layer, and 144 test samples. Extension to the remaining five datasets and to the
other modalities is future work.

### Section VIII, Limitation 1

Write that the causal validation **supported the correlational finding in part and complicated
it in part**: targeted ablation of probe-ranked audio units causally moves the target class in
both models, which is direct evidence that the attributed modality is load-bearing rather than
merely correlated; but the finer claim — that fine-tuning preserves or reassigns specific
class-selective units — did not separate from a random-ablation null, so the mechanism behind
RML's limited gain remains open. The remaining five datasets still rest on correlational
DeepSHAP evidence.

---

## 7. The one thing that is not just a placeholder: IV-C versus VI-D

Section IV-C defines the dominant modality as "the modality with the largest **aggregated**
SHAP contribution." Under that definition, RML's dominant modality is **text**:

| Model | View | text | video | audio |
|---|---|---|---|---|
| base | aggregate mass, sum abs phi | 0.2585 | 0.0106 | 0.0842 |
| base | per-neuron, mean abs phi | 0.000252 | 0.000165 | **0.001316** |
| fine-tuned | aggregate mass | 0.2653 | 0.0060 | 0.1150 |
| fine-tuned | per-neuron | 0.000259 | 0.000094 | **0.001797** |

VI-D hooked **audio**. As written, "we hooked the FFN output of the dominant modality (Section
VI-C)" is false under the paper's own definition. This is a coherence defect a reviewer will
find, and it is more dangerous than any statistical point in this document.

These are not two competing metrics. `mean|phi|` is `sum|phi|` divided by the modality's
dimension count; text leads on aggregate mass because it has 1024 dimensions against audio's
64, a 16:1 advantage. Audio leads per neuron by 5.2x (base) and 6.9x (fine-tuned), and is
rank-1 per neuron for **all six classes in both models**, unanimously. Audio also holds 23.8%
(base) and 29.8% (fine-tuned) of total attribution mass, so it is genuinely load-bearing and
not selected on a technicality. The two readings together reproduce the thesis's own wording,
"in the RML model, the dominant modality is text and audio."

**Minimal safe patch**: add a carve-out sentence, in IV-C or at the top of VI-D §1, stating
that the aggregate definition is used for the cross-dataset comparison in VI-C, while the
neuron-level intervention in VI-D targets the modality with the highest **per-neuron**
attribution, because a per-neuron intervention has to be targeted by a per-neuron metric and
the aggregate comparison is otherwise decided arithmetically by dimensionality. Never phrase
it as "the two metrics disagree."

**Open question for the co-authors, flagged not settled**: does the per-neuron reading change
the dominant modality for any of the other five datasets? If it does, Table IV moves, and
Table IV is the paper's central claim. The recommendation here is to scope the per-neuron
reading explicitly to VI-D and leave VI-C on the aggregate definition — but that is a
co-author decision, not one to make silently.

One related oddity worth a sentence but not a chase: Table III lists RML as **V, A** — no
native text track — which makes "text dominant by aggregate mass" independently strange.

---

## 8. Rigor verdict

### Verified sound

- **Provenance.** Both checkpoints SHA-256 verified against ADR 0004 at load time. The 16 kHz
  audio archive is size- and SHA-checked, and its WAV headers are read to confirm 16000 Hz
  rather than trusted by filename.
- **Fidelity gate.** The notebook halts unless the loaded checkpoints reproduce the logged
  training accuracies. They do, exactly: 76.39% base / 79.86% fine-tuned.
- **Data integrity.** 720 folders, every `audio.wav` and all 7377 sampled frame paths present,
  no stray files, no zero-frame samples, gate halts otherwise.
- **Falsification pair.** Full 64-dimension audio knockout costs 18.06 points; a random
  5-dimension control costs 0.00. The intervention machinery moves the model when it should
  and does not when it should not.
- **Hook correctness.** Write-through verified — under a full 64-dimension clamp all 144
  samples share one identical activation vector, max deviation 0.00e+00.
- **Fast path.** The cached-activation evaluation path is checked against real hooked forward
  passes at three operating points; max absolute probability deviation 9.43e-06 against a
  tolerance of 1e-3, and it reproduces the Day 0 fidelity gate for both models.
- **No leakage.** Probes are fit on the train split only (N=518) and scored on the held-out
  test split (N=144). The mean-ablation replacement vector is computed from the train set.
- **Bootstrap construction.** Paired (per-sample differences formed before resampling, so the
  clean/ablated correlation is retained), class-stratified, percentile CIs, common random
  numbers across conditions, fixed seed.
- **No epsilon fudge.** The selectivity ratio is computed only when at least one non-target
  class has an effect whose CI excludes zero **and** the mean absolute non-target effect
  clears 0.10 pp. Otherwise it is NaN with a stated reason. No epsilon is ever substituted
  into the denominator anywhere in the notebook.
- **A real null for the transfer claim.** 2000 random 5-dimension subsets, shared across both
  models, independent random stream, ratio-of-means estimator. Most ablation papers do not
  have this.
- **Internal consistency.** The Day 13b helper reproduces the Days 8–9 sweep exactly for all
  12 top-5 sets. Determinism is seeded at notebook start across `random`, `numpy`, and
  `torch`.
- **Self-criticism is in the artifacts, not just the prose.** The probe-ranking shortfall
  (12/12 pairs below 80%) is computed, exported, and printed by the notebook itself.

### Gaps to disclose (none require a re-run)

1. **No multiple-comparisons correction, anywhere.** Twelve target-effect CIs and sixty
   non-target CIs are all reported at nominal per-comparison 95%. Recommended handling:
   state explicitly that CIs are per-comparison and uncorrected, give the counts, and scope
   the section as exploratory single-dataset validation. Worth adding that a Holm correction
   across the twelve target effects would most plausibly flip only **fear in the base model**
   (CI [0.14, 1.10]) — a class already reported as non-selective — so the correction costs the
   conclusions nothing. Computing adjusted intervals properly would need the per-sample
   difference arrays, which are not in `results/`, so this is a documentation fix, not a run.
2. **n = 24 per class, one split, one seed per analysis.** No repeated splits, no
   seed-sensitivity check. Disclose the split size explicitly next to every CI.
3. **`selectivity_threshold = 2.5` is a chosen constant.** State that it was fixed in advance
   (ADR 0001) and report the raw ratios so a reader can apply their own line. Happiness at
   2.48 base is close enough that the threshold visibly decides a row.
4. **One layer, one modality, one dataset.** Already acknowledged in the closing sentence, but
   it should also appear in the abstract's characterization of VI-D as preliminary.

---

## 9. Concrete change list for the final paper

1. Fill Tables VII, VIII, IX from sections 3–5 above, in the paper's row order (Happiness
   last), with re-specified headers.
2. Change Table IX's caption away from "preservation vs. reassignment" and add the R_null
   column; verdict strings become "Not above null."
3. Write the four VI-D placeholders per section 6.
4. Rewrite the VI-D §1 Method sentence about "change in per-class test accuracy," and add the
   metric sentence to Section V-C.
5. Add the IV-C / VI-D dominant-modality carve-out per section 7, and raise the Table IV
   question with the co-authors.
6. Add the multiple-comparisons disclosure sentence per section 8.
7. Add a new figure: `figures/dose_response_k_sweep.png`, with a caption stating monotone
   target damage in k for all 12 class-model pairs and selectivity decaying toward 1.
8. Add a short paragraph to VI-D §1 stating the measurement-resolution argument (24 samples
   per class, 4.17 pp accuracy quantum) — this is the justification for the whole metric
   change and it should be visible, not buried in a footnote.
9. Add the probe-ranking-shortfall caveat to VI-D §2 and one clause to Limitation 1.
10. Footnote or appendix: the `legacy_acc_*` argmax numbers, so the accuracy result is
    disclosed rather than replaced silently.
11. Optional but cheap and strong: report the probe held-out AUCs (base mean 0.884, fine-tuned
    mean 0.910, no class below 0.55) in VI-D §1 as evidence the probes are readable before any
    ablation is discussed.

## 10. What was deliberately not done

- The `.docx` was not edited. It is open in Word (a `~$` lock file is present); patching it
  under a live lock risks losing the co-authors' edits.
- Nothing was committed. The working tree still holds the notebook changes, the new journal,
  and this file.
