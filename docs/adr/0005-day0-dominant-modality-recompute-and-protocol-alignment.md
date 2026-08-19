# ADR 0005: Day 0 Dominant-Modality Recompute, Protocol Alignment, and the Exact Fast Path

## Status

Accepted. Extends `0004-v3-restart-data-provenance-and-fidelity-gate.md` (still authoritative on data provenance and the fidelity gate). Amends ADR 0004 Decision 5 and resolves a standing conflict between `0001-causal-validation-methodology.md` Decision 4 and `CONTEXT.md`.

## Context

ADR 0004 closed the v1/v2 data-provenance failure but carried two things forward on trust rather than measurement:

1. **The dominant modality.** ADR 0004 Decision 5 retained ADR 0001's Audio selection and characterised it as "a near-tie resolved by a dimension-invariant tie-break." That characterisation dates from the v1/v2 era and was never recomputed against the SHA-verified checkpoints.
2. **The Day 0 blocking precondition** ("RML split IDs must be confirmed as keys in `meta_one_hot_label_six_categories.pkl`"). This was resolved during the 2026-08-09 session but recorded only in the journal, leaving the ADR reading as though it were still open.

Separately, a pre-run review on 2026-08-09 (session 2) surfaced three problems in `multimodal-causal-ablation-v3.ipynb` that would have cost a run each, and one documentation conflict that would have surfaced only at write-up time.

## Decisions & Rationale

### 1. Dominant modality is recomputed in the notebook, not inherited: and it is not a tie

`multimodal-causal-ablation-v3.ipynb` now recomputes the dominant modality at Day 0 step 2 from the upstream DeepSHAP output for both SHA-verified RML checkpoints (`checkpoints/RML_{origin,finetune}_SHAP_value.pkl`, copied off the handover drive from `Final_result/Origin_training/SHAP/SHAP_value/`). The result, computed on 2026-08-09:

| Model | View | text | video | audio | rank-1 |
| :--- | :--- | ---: | ---: | ---: | :--- |
| base | aggregate mass `sum(\|φ\|)` | 0.258486 | 0.010561 | 0.084228 | text |
| base | per-neuron `mean(\|φ\|)` | 0.000252 | 0.000165 | 0.001316 | **audio (5.2×)** |
| fine-tuned | aggregate mass `sum(\|φ\|)` | 0.265253 | 0.006042 | 0.114995 | text |
| fine-tuned | per-neuron `mean(\|φ\|)` | 0.000259 | 0.000094 | 0.001797 | **audio (6.9×)** |

Audio is rank-1 on per-neuron attribution for **all six classes in both models**, unanimously. Video is a distant third on both views.

**This is not two metrics disagreeing.** `mean(|φ|)` *is* `sum(|φ|)` divided by the modality's dimension count. It is one measurement read two ways: text leads on aggregate attribution mass because it has 1024 dimensions against audio's 64, a 16:1 advantage, while audio leads decisively per neuron. Any write-up that frames these as competing metrics will not survive review.

Consequences:

- **ADR 0004 Decision 5 is amended.** The RML dominant-modality situation is not a near-tie, and ADR 0001 Decision 1's 5%-parity tie-break is never invoked. The correct framing is per-neuron vs. aggregate.
- **Section VI-D's wording must change accordingly.** It should state that audio carries the highest per-neuron attribution (the metric that matters when the intervention is per-neuron), that text carries the largest aggregate mass by virtue of dimensionality, and that both together reproduce the thesis's own "the dominant modality is text and audio" (§4.4). The final paper text is the co-authors' call; this ADR records what the data says.
- Audio also holds ~24% of total attribution mass, so it is genuinely load-bearing, not selected on a technicality.

### 2. Day 0 step 2's provenance is proved, not assumed

The SHAP pickles' `test_feature` is the 1152-d concatenation `[text_cls 1024 | v_cls 64 | a_cls 64]` over the 144 RML test samples, the **same CLS representations this experiment ablates**. The attribution space is therefore literally the intervention space, which independently confirms the ADR 0004 finding that hooking `model.a_transformer` is correct.

Week 1 now asserts that `test_feature[:, 1088:1152]` correlates with the freshly extracted test `a_cls` at r > 0.99 for both models. Before this check, the only evidence tying the pickles to the checkpoints was a filename. Correlation rather than `allclose` because the upstream run used a different GPU, torch build, and batch size.

### 3. Notebook structure follows the protocol's Day/Week schedule

`experiment_protocol[From Prof KC Lan].docx` organises the work as Day 0 → Week 1 (Days 1–5) → Week 2 (Days 6–10) → Week 3 (Days 11–15) → Week 4 (Days 16–21+). The notebook now uses that vocabulary throughout; the older "Phase A–E" labels are retired from the notebook and the README roadmap. Dominant-modality identification is protocol Day 0 step 2, which is where it now lives.

Prior journals and ADRs 0001–0004 keep their original "Phase" wording, they are historical records and are not rewritten.

### 4. Day 0 gains a data-integrity gate below the fidelity gate

`IEMOCAP.__getitem__` selects video frames by **file count**, not by directory contents:

```python
nums = len(glob.glob(f'{folder}/*')) - 2
sampled = [f'image_{i}.jpg' for i in range(0, nums, int(500/1000*30))]
```

The `- 2` is exact only because every utterance folder carries both `audio.wav` and `audio_16000.wav`, verified across all 720 RML folders. **Superseded 2026-08-11:** it stopped being exact once the corpus moved into the runtime, because `RML_RAW_PROCESSED_Face.tar.gz` carries no `audio_16000.wav` and each folder therefore has one non-image member, not two. `sample_imgs_by_interval` now counts `image_*.jpg` directly, which holds both with the 16 kHz overlay (`data_bundle/RML_audio16k_all720.tar`, 111 members per folder, as on the training machine) and without it. The hazard below is unchanged and is the reason the direct count is the safer form. A stray `.DS_Store` (this repository lives in `My Drive` on macOS, so Finder can write one at any time) or a partially-synced Drive folder shifts `nums` silently, changing which frames the model sees relative to training. A `nums <= 0` folder is dropped outright by `collate_fn` without an error. A constructed path that doesn't exist surfaces as an opaque DataLoader **worker** crash from PIL, mid-epoch.

Day 0 now replicates `sample_imgs_by_interval` byte-for-byte over all 720 folders and asserts every constructed frame path exists, no stray files are present, and no folder samples zero frames, before any model is loaded. This is the ADR 0004 failure class (wrong data, no error) one layer down. `docs/rml_file_manifest.json` records the 2026-08-09 per-folder file counts as a secondary drift diagnostic.

Consequently, Week 1 asserts sample counts against **retained utterance IDs** rather than the literals 518/58/144, and requires the base and fine-tuned runs to have retained identical ID lists in identical order.

### 5. Weeks 2–3 use an exact fast path, verified against real forward passes

With `mod='tav'`, `MME2E.forward` ends in `weighted_fusion(stack([t_out(t_cls), v_out(v_cls), a_out(a_cls)], -1)).squeeze(-1)` where `weighted_fusion` is `nn.Linear(3, 1, bias=False)`. Mean ablation touches only `a_cls`. Therefore the `(t_logits, v_logits, a_cls)` triple cached in Week 1 Days 1–2 determines the model output under *any* audio-CLS ablation, exactly.

Days 8–9 plus Day 14 plus baselines run over 100 evaluations. On the slow path each re-reads 144 wav files and ~1,300 JPEGs across the Colab Drive mount, roughly 150,000 file reads. GPU time was never the constraint (the original training log shows test eval at ~9 it/s); I/O is.

The fast path is **proved before use**: the notebook reproduces the unablated baseline, the full-64 knockout, and one class at k=5 through real hooked forward passes and compares per-class accuracies. If any check disagrees it prints loudly, records the failure in `results/week2_fast_path_verification.json`, and falls back to real forward passes, the slow path is the reference implementation, so falling back is always safe.

The Week 2 sweep additionally asserts that its own unablated baseline reproduces Day 0's fidelity-gate accuracy exactly, catching any drift between the cached activations and the live model.

### 6. Falsification pair gains a threshold-free proof that the hook writes through

The existing `assert full_knockout_drop > 10.0` conflates two different failures: a hook that never fired, and a modality that isn't load-bearing. A mechanical check now runs first, under a full 64-dimension clamp every sample must receive an *identical* `a_cls`, so `a_out(a_cls)` contributes the same constant to every sample. This depends on no tuned number. With it in place, the 10-point threshold's failure message can state plainly that the hook *is* firing and the finding is therefore scientific, not mechanical.

### 7. Substrate taxonomy follows `CONTEXT.md`, not ADR 0001 Decision 4

The two disagree:

| Source | Preservation | Reassignment | Dispersion |
| :--- | :--- | :--- | :--- |
| ADR 0001 Decision 4 | R ≥ 0.70 | R < 0.30 with FT *sparse* drop | R < 0.30 with FT *dense* drop |
| `CONTEXT.md` | R ≥ 0.80 | 0.20 ≤ R < 0.80 | R < 0.20 |

ADR 0001's version is not implementable as written: it leaves 0.30 ≤ R < 0.70 unclassified, and its Reassignment/Dispersion split depends on a "sparse vs. dense drop" measure the protocol never defines. `CONTEXT.md`'s bands are total, and the README names `CONTEXT.md` as the canonical terminology. The notebook implements `CONTEXT.md`'s bands.

Raw R is reported for every class alongside `base_drop_pp` and `ft_drop_pp`, so relabelling later costs nothing. Classes with a small or **negative** `base_drop` are flagged explicitly, a negative denominator makes R's sign uninterpretable as a retention ratio, and those rows must be read from the raw drops.

### 8. ADR 0001 Decision 2's layer fallback has no second rung

ADR 0001 Decision 2 (and protocol Day 4) prescribe falling back from "FFN output" to "Encoder CLS" if probe AUC is too low. ADR 0004 established that for `MME2E` these are the same tensor: `a_transformer(..., get_cls=True)` returns `inputs[0]`, the CLS output, which feeds directly into `a_out`. There is no second rung. If the Day 4 threshold fires, the real options are a different modality's CLS (which would contradict Decision 1 above) or a different layer depth, either is a methodology change requiring its own ADR, not a rerun. The notebook's Day 4 warning says so.

## Resolutions

- **ADR 0004's blocking precondition is RESOLVED.** All 518 train + 58 valid + 144 test RML split IDs are present as keys in `meta_one_hot_label_six_categories.pkl` (63,723 keys total), with zero missing, and the one-hot label encoding matches `getEmotionDict()`'s ordering exactly. Verified 2026-08-09 and re-verified in session 2. No ID map is needed. Do not re-litigate this.
- **ADR 0004's fidelity-gate targets now have primary sources in-repo.** `docs/provenance/RML_{origin,finetune}_training_log.txt` are the original `main.py` stdout captures: `Test (29) 0.763889` and `Test (15) 0.798611`. Their first lines are also the exact training command lines, confirming every architecture argument in `MODEL_ARGS`. (The origin run used `-lr=1e-5 -wd=1e-3 --early-stop=6`; the fine-tune run used `-lr=5e-5 --early-stop=40`. They differ only in optimizer and schedule settings, which are irrelevant at inference.)
- **ADR 0004 Decision 5 is amended** per Decision 1 above: not a near-tie.
