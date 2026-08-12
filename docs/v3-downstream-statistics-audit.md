# Audit: v3 downstream statistics (Weeks 2–4)

> **Recovered 2026-08-12.** This file was destroyed with the rest of the
> 2026-08-11 session-2/3 working tree and was rebuilt by replaying the tool
> calls that wrote it, from the Claude Code transcript
> `ded06ca6-9d1e-48a9-9567-bb067b3b36c4.jsonl`. Content is byte-for-byte the
> text those calls produced; only this note is new.

**Date:** 2026-08-11
**Scope:** `multimodal-causal-ablation-v3.ipynb`, the executed outputs committed in `1d5aa39`.
**Subject:** the ablation sweep (Week 2, Days 8–9), the transfer retention table (Week 3, Day 15),
and the exported publication tables (Week 4, Days 16–18).

**Verdict:** the accuracy pipeline upstream of these cells is sound. Every derived
statistic downstream of it is an artifact. Zero of the five `causally_class_selective = True`
verdicts and zero of the six retention-table `outcome` labels survive inspection.

**Status:** the five affected cells have been rewritten (§6) and their stale outputs
cleared. Weeks 2–4 need a re-run; Weeks 0–1 are untouched. The new code has not been
executed yet, so this document still describes the last measured numbers.

The cleared outputs are not lost: the executed run they came from is committed in
`1d5aa39`, and the numbers this audit turns on are quoted in §3 above. Before re-running,
reload the notebook in the IDE — the file was rewritten on disk, and an editor still
holding the old cells in memory will overwrite them on its next save.

---

## 1. What is *not* wrong

These were checked first, because if any of them were broken the rest would not be worth
analysing:

- **The fidelity gate holds.** Base 76.39%, fine-tuned 79.86%, reproduced by the sweep's
  own unablated baseline to `< 1e-6`.
- **The fast path is exact.** All three discriminator checks in Week 2 Days 6–7 report
  `max per-class delta = 0.00e+00pp` against real hooked forward passes, including the
  full 64-dim knockout and a realistic top-5 ablation.
- **The probes are real.** Held-out test AUC is 0.8835 (base) and 0.9102 (fine-tuned),
  no class below 0.55. The audio CLS does carry a linear class signal.
- **The ablated tensor is the right tensor.** The Day 0 step 2 cross-check reports
  Pearson `r = 1.000000` between the DeepSHAP reference features and the extracted
  test `a_cls`.
- **The audio branch matters in aggregate.** Clamping all 64 dimensions to the train mean
  moves overall accuracy 76.39% → 56.25%.

The failure is entirely in how per-class effects are *measured* and then *divided*.

## 2. Root cause: the measurement has no resolution

The evaluation split is 144 samples across 6 classes — 24 per class. Per-class accuracy
is `mean(argmax(logits) == label)`, so the finest per-class effect the harness can
represent is **one sample = 4.1667pp**. There is no smaller number available to it.

This is visible in the data, not merely inferred. Every per-class delta in every exported
table is an exact multiple of 4.166667, and every non-target aggregate is a multiple of
0.833333 (= 4.166667 / 5, one sample averaged over the five non-target classes) or
1.666667 (= two such). Nothing else appears anywhere in Tables VII, VIII, or IX.

So each "target-class drop" in the published tables is a statement that **either zero or
one test utterance changed its predicted label.** Cell 25 already prints this
("read it as single-sample noise"), and it flags all 6 of 6 rows. The flag is correct.
What follows is what the notebook does with those numbers anyway.

## 3. Three defects that turn noise into confident verdicts

### 3.1 The epsilon floor manufactures selectivity (Week 2, Days 8–9)

```python
denom = max(mean_abs_non_target_drop, 1e-6)
selectivity_ratio = target_drop / denom
'causally_class_selective': bool(selectivity_ratio >= 2.5)
```

`1e-6` is intended as a divide-by-zero guard. Because the numerator is quantized to
4.1667, it instead acts as a *multiplier of 4.17 million* whenever the non-target drop is
exactly zero:

| model | class | target_drop | mean_abs_non_target | ratio | verdict |
|---|---|---|---|---|---|
| base | disgust | 4.166667 | **0.000000** | 4.166667e+06 | SELECTIVE |
| finetuned | anger | 4.166667 | **0.000000** | 4.166667e+06 | SELECTIVE |
| finetuned | fear | 4.166667 | **0.000000** | 4.166667e+06 | SELECTIVE |

Three of the five positive verdicts are this. Each says "one sample moved, and nothing
else moved" — which on a 24-sample-per-class grid is the single most likely outcome of
noise, not evidence of a causally class-selective circuit.

### 3.2 The threshold resolves on float residue

`base/happiness` and `base/surprise` both print `selectivity_ratio = 2.500000e+00` in
Table VII. Both are 4.166667 / 1.666667. They receive **opposite** booleans from the same
expression `bool(selectivity_ratio >= 2.5)`. The gate is being decided by rounding error
below the printed precision.

Corroborating evidence in the same table: `base/fear` has
`mean_non_target_drop = 5.684342e-15`, i.e. a quantity that should be exactly zero
carrying accumulated float error. The comparison is at a knife-edge that the arithmetic
cannot resolve.

That accounts for the fourth positive verdict. The fifth, `finetuned/disgust` at 5.00x, is
the only one produced by neither an epsilon nor a tie — and it is 4.1667 / 0.8333, i.e.
one target sample moved and one non-target sample moved. **Zero of five survive.**

### 3.3 R is a ratio of two single samples (Week 3, Day 15)

```python
r = ft_drop / base_drop if abs(base_drop) > 1e-9 else float('nan')
```

Both operands are quantized to {0, 4.1667, 8.3333}. R can therefore only take the values
0, 1, 2, or NaN — which is exactly what the table shows. `R = 1.0` labelled
"Substrate Preservation" means *one sample moved in the base model and one sample moved in
the fine-tuned model*. Three of six rows are NaN purely because `base_drop` happened to
land on zero.

The row that proves R carries no information:

| class | base_ft_top5_overlap | R | outcome |
|---|---|---|---|
| happiness | **0/5** | 1.0 | Substrate Preservation |

Zero shared neurons between the two models' top-5 sets, and the metric reports perfect
substrate preservation.

## 4. The caveat does not reach the exported artifacts

Cell 25 prints an explicit warning that all six rows are single-sample noise, and computes
`small_base_drop_flag = True` for every class. Week 4 then exports:

```python
table_ix = retention_df[['class', 'cosine_similarity', 'ft_drop_pp', 'R', 'outcome']]
```

`small_base_drop_flag` is dropped. `outcome` — the string "Substrate Preservation" — is
kept. Tables VII and VIII do the same thing with `causally_class_selective`: the boolean
ships, the denominator that produced it does not.

All three publication CSVs therefore carry verdicts with the disqualifying caveat left
behind in notebook stdout. This is the direct answer to "downstream stats look sus": the
notebook diagnoses its own problem correctly and then exports the conclusions anyway.

## 5. Secondary observation: the "top-5" framing is thin for some classes

The Day 3 assert is `n_nonzero_weights >= 5`. `finetuned/anger` passes with exactly 5/64
and `base/anger` with 6/64. For those classes the "top-5 neurons" is the entire set L1
retained, ordered by a ranking with almost nothing left to rank. Not a bug — the assert
does what it says — but it weakens the claim the ablation is meant to test, and anger is
one of the classes carrying a positive selectivity verdict in Table VIII.

## 6. What was changed

Five code cells in `multimodal-causal-ablation-v3.ipynb` were rewritten. Their stale
outputs were cleared, so **Weeks 2–4 must be re-run**; Weeks 0–1 are untouched and their
outputs still stand. The audit above describes the numbers as they were before this
change, and none of the new code has been executed yet.

### 6.1 The measurement (the real fix)

Argmax per-class accuracy on 24 samples cannot resolve the effect being studied. No guard,
threshold, or epsilon change rescues it, so the primary quantity was replaced with two
that are defined **per sample**, in Week 2 Days 6–7 (`per_sample_effects`,
`ablation_effects`):

- **True-class softmax probability drop**, in pp of probability mass. Every one of the 144
  samples contributes, and movement that does not cross a decision boundary is no longer
  discarded.
- **Decision margin drop**, `logit[true] − max(logit[other])`, in logit units. Positive
  exactly when the sample is classified correctly, so it measures *distance* to the
  boundary rather than which side of it a sample happens to sit on.
- **Paired, class-stratified bootstrap** (2000 draws, fixed seed) on every reported effect.
  The per-sample difference is formed before resampling, so the interval is a CI on the
  effect itself; resampling happens within class, which is the population the per-class
  mean is a mean over. All ablations share one seed, so comparisons between them are not
  inflated by independent resampling noise.
- Argmax accuracy survives under `legacy_acc_*` names as a clearly-labelled secondary
  column. No ratio is computed from it any more.

`MME2E.weighted_fusion` is `nn.Linear(3, 1, bias=False)` (`src/models/e2e.py:83`), so the
fast path's logits are exact in absolute value, not merely argmax-exact — which is what
licenses reading probabilities and margins off them.

### 6.2 The guards

- `max(x, 1e-6)` is gone. `selectivity_verdict` returns `selectivity_ratio = NaN` with a
  stated reason whenever the denominator cannot support a ratio. Two independent
  conditions void it: no non-target class has an effect whose CI excludes zero (the
  collateral damage being summarised is not itself resolvable), or the mean absolute
  non-target effect is below a `MIN_DENOM_PP = 0.10` floor (a near-zero denominator
  explodes the ratio however it arose). A voided denominator **with** a resolvable target
  effect is reported as an *isolated* effect by name — the strongest possible outcome, and
  precisely the one a ratio cannot express — rather than converted into a large finite
  number.
- The resolvability condition is phrased as "at least one non-target class is resolvable"
  rather than "the denominator's own CI excludes zero" because `mean|x|` is strictly
  positive for pure noise: a CI on it never contains zero, so that check would pass
  unconditionally. Both are computed and reported (`denominator_ci95_lo/hi`,
  `n_resolvable_non_target`); only the one with teeth gates the ratio.
- `MIN_DENOM_PP` remains a chosen constant, so the sweep cell now prints the min, median
  and max observed denominator against it, plus counts of ratios computed vs. voided by
  each cause. If the floor is in the wrong decade for this data it shows up immediately as
  an all-void column or a floor that never fires, and it can be calibrated from the first
  real run instead of left asserted.
- A selectivity verdict now requires three conditions, not one: a finite ratio, a target
  effect whose CI excludes zero, and `ratio >= 2.5`.
- R gets the same treatment. Its `abs(base_drop) > 1e-9` test (nine orders of magnitude
  below the old grid spacing, so it never fired as intended) is replaced by the same floor
  plus a CI check, and R now carries its **own** bootstrap CI, resampling the same test
  rows in both models so the ratio stays within-sample. A band whose CI spans a band
  boundary is flagged `band_stable_across_ci = False`.
- The zero-shared-neurons-but-"Preservation" contradiction from §3.3 is now detected and
  printed as an explicit `WARNING` rather than exported silently.

### 6.3 The exports

- Tables VII/VIII carry `target_prob_drop_ci95_lo/hi`, `target_ci_excludes_zero`,
  `mean_abs_non_target_prob_drop_pp`, `non_target_denom_ci95_lo/hi`,
  `n_resolvable_non_target`, and `selectivity_undefined_reason` beside the
  `causally_class_selective` boolean.
- Table IX carries `R_ci95_lo/hi`, `band_stable_across_ci`, and `ft_ci_excludes_zero`
  beside `outcome`.
- New `results/measurement_resolution.json` records split size, samples per class, the
  4.1667pp accuracy resolution, the bootstrap parameters, and the denominator floor.
- The dose-response figure gains a second row: the primary measure with its CI band on
  top, the old argmax accuracy drop below it drawn against a dotted 4.1667pp grid, so the
  staircase is visible rather than argued about.

### 6.4 One check added while in there

The Days 6–7 discriminator compared only per-class accuracy, which is **permutation
invariant** — it would pass even if the cached rows were misaligned with the loader, and
every continuous measure above depends on that alignment. It now also compares fused
probabilities elementwise and halts if they differ by more than `1e-3` (float32 GPU-vs-CPU
noise is ~`1e-6`; a row permutation would be O(1)).

## 7. Provenance of the evidence

`results/` on the local Drive mount is empty, but the artifacts **do** exist in web Drive —
`week2_base_ablation_sweep.json` (38412 B) and `week2_finetuned_ablation_sweep.json`
(38888 B), both written 2026-08-11T06:09Z. The local mount is simply behind; the Colab
runtime writes to `/content/drive/MyDrive/...`, which is authoritative. Local Drive files
should not be trusted as current for anything the notebook produced.

**Everything in §§1–5 is drawn from the k=5 slice printed in the committed notebook
outputs**, which is exactly what Tables VII/VIII/IX were built from. The full sweep JSONs
would only add whether the dose-response curve is monotone at k = 16/32/48/64 — a
nice-to-have that changes no conclusion here, and the aggregate audio effect is already
established independently (76.39% → 56.25% at k=64).

---

## 8. Post-run verification (2026-08-11, session 2)

The notebook was re-run end to end after the rewrite. Weeks 0–2 are sound and the
measurement defects of §3 are gone. Week 3 is not publishable as it stands, for reasons the
new numbers expose rather than create.

### 8.1 The measurement fix held

- The fast path reproduces hooked forward passes exactly: `max per-class delta = 0.00e+00pp`
  and `max|prob delta|` of 1.45e-07 / 4.72e-08 / 1.21e-07 across the three checks. The row
  alignment assert did not fire.
- **96/96** target-class effects (2 models × 6 classes × 8 values of k) have a 95% bootstrap
  CI that excludes zero. Nothing rests on an unresolvable effect any more.
- The selectivity denominator is healthy on real data: min 0.0759pp, median 2.4642pp, max
  30.0007pp, with **1/96** cells below the `MIN_DENOM_PP = 0.10pp` floor. The floor is
  therefore doing what it was meant to do — voiding a single genuinely degenerate cell — and
  is not silently voiding or silently passing the table. 95 ratios carry a verdict, 1 is
  undefined with a stated reason.
- Selectivity at k=5 is now a discriminating column rather than a constant: base model 4/6
  selective (anger 10.66×, sadness 5.14×, surprise 4.43×, disgust 3.14×; fear 1.10× and
  happiness 2.48× fall below the 2.5 threshold), fine-tuned model 3/6 (anger 10.69×, sadness
  5.70×, surprise 3.64×).

### 8.2 Blocking: the base model's neurons out-damage the fine-tuned model's own

Comparing the fine-tuned sweep (its own probe-selected top-5, cell 20) against the transfer
cell (the *base* model's top-5 applied to the fine-tuned model, cell 23), on the same model
and the same rows:

| class | FT's own top-5 | base's top-5 in FT | ratio |
|---|---|---|---|
| anger | 5.767pp | 7.315pp | 1.27 |
| disgust | 1.918pp | 4.792pp | 2.50 |
| fear | 1.687pp | 2.771pp | 1.64 |
| happiness | 2.861pp | 2.845pp | 0.99 |
| sadness | 6.807pp | 5.354pp | 0.79 |
| surprise | 2.865pp | 4.742pp | 1.65 |

In four of six classes an *externally chosen* set of five dimensions damages the fine-tuned
model more than the set its own L1 probe ranked highest — by 2.5× for disgust. The precise
claim this supports is that **|probe weight| does not predict ablation damage magnitude**,
not that the Week 1 ranking is invalid: the two models' probes are fit on different
activations, so a set of five dimensions that happens to lie along the fine-tuned model's
high-variance directions can out-damage that model's own probe pick without the probe being
wrong. It is still a problem, because the transfer analysis assumes the top-5 set is *the*
causally important set for its model, and that assumption is what fails here. It needs an
explanation, or a ranking criterion validated against measured ablation effect, before a
transfer claim is built on top of it.

### 8.3 The retention band scheme is undefined over the range the data actually occupies

Every one of the six classes returned R > 1 (1.32 to 4.34) and every one was labelled
*Substrate Preservation*. The band boundaries — Preservation ≥ 0.80, Reassignment
0.20–0.80, Dispersion < 0.20 — were written on the assumption that R ∈ [0, 1]. Fear
(R = 4.34, CI [2.68, 9.34]) is labelled Preservation while meaning that the base model's
neurons hurt the fine-tuned model 4.3× *more* than they hurt the base model, which is not
what "preservation" asserts. A column that takes one value for all six rows tests nothing.
Either an explicit R > 1 category is defined and justified, or the bands are dropped and R
is reported with its CI.

### 8.4 R is largely a global sensitivity difference, not class-specific transfer

The fine-tuned model is simply more ablation-sensitive overall. Taking the ratio of the
fine-tuned to base *mean absolute non-target* drop at k=5 as a rough global scale factor:

| class | global scale (FT/base, non-target) | R |
|---|---|---|
| anger | 1.329 | 1.690 |
| disgust | 1.382 | 2.083 |
| fear | 1.705 | 4.339 |
| happiness | 1.733 | 1.393 |
| sadness | 1.511 | 1.317 |
| surprise | 1.524 | 2.072 |

Happiness and sadness sit *below* the global scale, and anger, disgust and surprise sit
close to it. Only fear stands clearly above. R as currently defined does not separate
"the class substrate transferred" from "this model is more fragile everywhere".

**The check that would discriminate:** ablate five *random* `a_cls` dimensions in both
models over many draws, on the same fast path, and bootstrap the fine-tuned/base
probability-drop ratio into a null distribution `R_null` with a CI. Any class whose R CI
overlaps `R_null` has no transfer claim. This is matmuls on cached activations and costs
almost nothing.

### 8.5 Top-5 overlap is at chance for three of six classes

Overlap between the two models' top-5 sets, out of 64 dimensions, is hypergeometric under
the null: expected overlap 0.391, P(X ≥ 1) = 0.343, P(X ≥ 2) = 0.045, P(X ≥ 3) = 0.0023.

- anger 3/5 and surprise 3/5 are genuinely above chance (p = 0.0023).
- fear 2/5 is marginal (p = 0.045).
- disgust 1/5 and sadness 1/5 are at chance (p = 0.343); happiness 0/5 is the modal outcome
  under the null.

The notebook's WARNING for happiness (zero overlap yet "Substrate Preservation", and the
one row whose band is not stable across its CI) is therefore not an edge case to note in
passing — three rows have no evidence of shared substrate at all, while cosine similarity
between the full 64-dim selectivity vectors is 0.96–0.99 for every class. That gap is itself
the finding: a near-unit cosine coexisting with chance-level top-k overlap means the
selectivity vectors are dominated by a shared component and the top-k *ranking* is unstable.
Top-k index overlap should not be reported as a substrate measure without that caveat.

### 8.6 Verdict

Weeks 0–2 are ready to write up. Week 3 is not: §8.2 undermines the neuron ranking the whole
transfer analysis selects, §8.3 makes the band column uninformative, §8.4 shows R is
confounded with a global sensitivity change, and §8.5 shows half the classes share no
neurons above chance. The random-ablation null of §8.4 is the smallest piece of work that
would settle whether a transfer claim survives.

---

## 9. Week 3 remediation (2026-08-11, session 2) — implemented, awaiting a run

All four §8 problems are now instrumented in `multimodal-causal-ablation-v3.ipynb`. The
decisions and their rationale are `docs/adr/0006-week3-transfer-null-and-band-withdrawal.md`;
this section is the map from problem to cell.

| §8 problem | Fix | Where |
| :--- | :--- | :--- |
| 8.2 ranking never validated | 64 single-dimension ablations per model; probe top-5 vs. damage-ranked top-5, their damage, and a top-16-union Spearman | **Week 3, Day 13b** (new cell), Table X |
| 8.4 R has no reference point | Random-ablation null: 2000 random 5-subsets, same subsets in both models, ratio of means, bootstrap over subsets | **Week 3, Day 14b** (new cell), `results/week3_random_ablation_null.json` |
| 8.3 bands cannot classify | Bands withdrawn. R reported with its CI against `R_null` and one binary verdict | **Week 3, Day 15** (rewritten), Table IX |
| 8.5 overlap without a chance level | Exact Hypergeometric(64, 5, 5) p-value and expected overlap emitted beside every overlap count; standing caveat printed for p > 0.05 | **Week 3, Day 15** |
| 8.5 cosine without a null | Full 6×6 base × fine-tuned cosine matrix; matched diagonal reported against the mismatched off-diagonal | **Week 3, Days 11–13** (extended) |

Guards that make the null comparable to the estimate it calibrates: Day 13b asserts its
ablation helper reproduces the Days 8–9 sweep to 1e-9 for all twelve top-5 sets, and both new
cells halt unless the Days 6–7 fast-path discriminator passed. The null's seed (`20260812`) is
deliberately not `BOOTSTRAP_SEED`.

**What to read first when the run finishes.** Day 15's summary line: how many classes have an
R whose CI clears the null. If that count is zero the cell prints a `NEGATIVE RESULT` block,
and the Week 3 subsection is written as a null result with a null distribution behind it —
not as a shared-substrate finding. §8.6's verdict stands until that number exists.
