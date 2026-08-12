# Neuron-Level Causal Validation of Modality Competition in Multimodal Emotion Recognition

This repository contains my implementation of Section VI-D for a paper on transfer learning effects in multimodal emotion recognition, co-authored with Prof. KC Lan and Tsung-Yi Ko. The central question: when a multimodal model learns to recognize emotions from audio, text, and video simultaneously, does it develop specialized neurons for the modality it relies on most — and do those neurons survive fine-tuning?

## Motivation

Multimodal emotion recognition models fuse audio, text, and visual signals to classify emotional states. Prior work by my co-authors demonstrated that during training, modalities *compete* — the model suppresses weaker modalities in favor of a dominant one. But that observation was correlational. My contribution is the causal test: I identify the specific neurons responsible for the dominant modality's predictions, ablate them, and measure whether the model's per-class prediction confidence actually drops. Then I test whether fine-tuning keeps those same neurons load-bearing, by ablating the base model's top neurons inside the fine-tuned model and comparing the damage against a random-ablation null. As of the 2026-08-12 run, the answer is negative: the base model's top neurons do no more damage inside the fine-tuned model than arbitrary neurons do — see [ADR 0006](docs/adr/0006-week3-transfer-null-and-band-withdrawal.md).

## Methodology

The experiment follows the four-week protocol in [`experiment_protocol[From Prof KC Lan].docx`](experiment_protocol[From%20Prof%20KC%20Lan].docx), with the methodology decisions formally documented in [ADR 0001](docs/adr/0001-causal-validation-methodology.md). `multimodal-causal-ablation-v3.ipynb` is organised by that protocol's Day/Week schedule:

- **Day 0 — Setup & gates:** Verify checkpoint identity by SHA-256; confirm every data file the loader will read actually exists; reproduce the paper's published RML test accuracy from the canonical data source or halt; identify the dominant modality from DeepSHAP attribution; prove the ablation hook has a real, appropriately-scaled causal effect with a full-modality knockout and a random-neuron control.
- **Week 1 (Days 1–5) — Activation extraction & neuron ranking:** Cache the dominant modality's CLS activations for every sample under both models, fit L1-penalized logistic probes per emotion class on the train split only, and sanity-check them on the held-out test split.
- **Week 2 (Days 6–10) — Causal ablation:** Replace the top-k most class-selective neurons with their train-set mean activations during inference. A target-class accuracy drop of at least 2.5× the mean absolute non-target drop marks those neurons as causally class-selective, not merely correlated.
- **Week 3 (Days 11–15) — Preservation vs. reassignment:** Compare per-class selectivity vectors between models by cosine similarity, then run the decisive test — ablate the base model's top neurons inside the fine-tuned model — and compute the Transfer Retention Ratio against a measured random-ablation null (not a fixed threshold; see [ADR 0006](docs/adr/0006-week3-transfer-null-and-band-withdrawal.md)).
- **Week 4 (Days 16–21+) — Write-up:** Publication tables and figures, then the Section VI-D text for the IEEE submission.

## Roadmap (Live Status)

This table tracks experiment progress. Update it at the end of every session.

**2026-08-09 restart:** all prior numbers (both `multimodal-causal-ablation.ipynb` and `-v2.ipynb`) are retracted. Root cause: the notebook evaluated the correct, SHA-verified checkpoints against a fabricated `meta.pkl` instead of the canonical merged-corpus data source. See [ADR 0004](docs/adr/0004-v3-restart-data-provenance-and-fidelity-gate.md) and [`journals/2026-08-09-session1.md`](journals/2026-08-09-session1.md) for the full diagnosis. The notebook was then hardened and re-aligned to the protocol's Day/Week schedule — see [ADR 0005](docs/adr/0005-day0-dominant-modality-recompute-and-protocol-alignment.md) and [`journals/2026-08-09-session2.md`](journals/2026-08-09-session2.md).

**2026-08-11/12 — v3 run to completion:** the notebook ran Weeks 1–4 end to end on real data. Week 3's substrate-preservation bands were withdrawn — every class's raw transfer ratio sat above 1 under any band scheme, but that turned out to be the default state of a more-fragile fine-tuned model, not evidence of preservation. Testing against a measured random-ablation null instead: 0 of 6 classes clear it. **Week 3 is a negative result** (write it as one, never as shared substrate) — see [ADR 0006](docs/adr/0006-week3-transfer-null-and-band-withdrawal.md) and [`journals/2026-08-11-session2.md`](journals/2026-08-11-session2.md) / [`session3`](journals/2026-08-11-session3.md). On 2026-08-12 the whole notebook was re-run from committed code as a reproduction check: fidelity gate still passes exactly (76.39% / 79.86%), 8 of 9 tracked artifacts came back byte-identical and the 9th differs only by GPU float noise, and both Week 3 negative results reproduced exactly — see [`journals/2026-08-12-session1.md`](journals/2026-08-12-session1.md). Paper Tables VII–X are filled from real artifacts in [`docs/vi-d-paper-writeup-guidance.md`](docs/vi-d-paper-writeup-guidance.md); the experiment itself is done, what remains is editorial (unwritten prose blocks, two table headers asking for a quantity accuracy can't resolve — see below).

| Protocol stage | Status | Summary |
| :--- | :--- | :--- |
| Day 0 — Setup & gates | ✅ PASSED | Checkpoint SHA, data integrity, fidelity (76.39% / 79.86%, both exact) and falsification pair all pass; reproduced from a cold run on 2026-08-12 |
| Day 0 step 2 — Dominant modality | ✅ RECOMPUTED | Audio, unanimous across 6 classes × 2 models on per-neuron `mean(\|φ\|)` (5.2× / 6.9× over text). Not a tie — see [ADR 0005](docs/adr/0005-day0-dominant-modality-recompute-and-protocol-alignment.md) |
| Week 1 — Activations & probes | ✅ COMPLETE | CLS activations cached, L1-logistic probes fit and sanity-checked on held-out test split (`results/week1_probe_auc.json`) |
| Week 2 — Causal ablation | ✅ COMPLETE | Mean-ablation sweep run for both models (`results/week2_{base,finetuned}_ablation_sweep.json`); probability drop (not accuracy) is the primary measure — 144-sample test split quantizes accuracy to 4.17pp steps, smaller than every measured effect |
| Week 3 — Transfer retention | ✅ COMPLETE — negative result | Substrate bands withdrawn (they separated nothing — every class sat above 1 under any scheme). Tested against a measured random-ablation null instead: 0 of 6 classes clear it. Report as "no transfer signal above random ablation," never as shared substrate. See [ADR 0006](docs/adr/0006-week3-transfer-null-and-band-withdrawal.md) |
| Week 4 — Results & paper integration | 🟨 IN PROGRESS | Tables VII/VIII/IX/X filled from real artifacts in [`docs/vi-d-paper-writeup-guidance.md`](docs/vi-d-paper-writeup-guidance.md); experiment is done, remaining work is editorial (4 unwritten prose blocks, 2 table headers need rewording, Table IX caption and Section IV-C wording need alignment) |

## Key Files Reference

| File | Purpose |
| :--- | :--- |
| `multimodal-causal-ablation-v3.ipynb` | Main experiment notebook (in progress) — runs on Google Colab GPU |
| `CONTEXT.md` | Domain glossary with canonical terminology. |
| `docs/adr/0001-causal-validation-methodology.md` | Locked methodology decisions: thresholds, fallback tiers, taxonomy |
| `docs/adr/0002-infrastructure-and-journaling-protocol.md` | Infrastructure decisions: Git scope, journals, artifact naming, seed policy |
| `docs/adr/0004-v3-restart-data-provenance-and-fidelity-gate.md` | Root-cause diagnosis of the v1/v2 failures, canonical data source, halting fidelity gate |
| `docs/adr/0005-day0-dominant-modality-recompute-and-protocol-alignment.md` | Dominant-modality recompute, protocol Day/Week alignment, data-integrity gate, exact fast path, taxonomy conflict resolution |
| `docs/adr/0006-week3-transfer-null-and-band-withdrawal.md` | Withdraws the Week 3 substrate-preservation bands, replaces them with a measured random-ablation null, switches the primary measure to probability drop |
| `docs/vi-d-paper-writeup-guidance.md` | Section VI-D tables (VII–X) filled from real result artifacts, rigor verdict, and what's still unwritten in the paper draft |
| `docs/v3-downstream-statistics-audit.md` | Audit of the downstream statistics pipeline that motivated the probability-drop measure and the random-ablation null |
| `experiment_protocol[From Prof KC Lan].docx` | The four-week protocol the notebook implements, day by day |
| `docs/provenance/RML_{origin,finetune}_training_log.txt` | Original `main.py` stdout for both checkpoints — the primary source for the 76.39% / 79.86% fidelity-gate targets and for every `MODEL_ARGS` value |
| `docs/provenance/upstream_SHAP_analysis.ipynb` | The previous researchers' DeepSHAP notebook — documents the 1152-d `[text 1024 \| video 64 \| audio 64]` feature layout |
| `docs/rml_file_manifest.json` | Per-folder file counts for all 720 RML utterance folders, snapshot 2026-08-09; Day 0's secondary drift check |
| `results/` | All numbered artifacts (Week 1 probe AUC, Week 2 ablation sweeps, Week 3 transfer ratios and null, Tables VII–X) produced by the 2026-08-12 cold run |
| `checkpoints/README.md` | SHA-256 checksums for model weights and SHAP pickles |
| `journals/GUIDELINES.md` | Internal journal writing template and tone rules |
| `journals/README.md` | Chronological index of all journal entries with stage and status |
| `src/utils.py` | Contains `set_deterministic_seed(seed=0)` |

## Reproducibility

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (experiments run on Google Colab T4/V100)
- PyTorch 2.2.2 (pinned for C++ ABI compatibility with upstream model code)

### Data & Checkpoints

The RML dataset, model checkpoints, and DeepSHAP pickles are too large for Git and live on Google Drive / an external handover drive only. Canonical data source: `BIG_DATA_RAW_PROCESSED_FACE.tar.gz` filtered to RML split IDs (see [ADR 0004](docs/adr/0004-v3-restart-data-provenance-and-fidelity-gate.md) — do not use the retired `RML_RAW_PROCESSED_Face` folder). Checkpoint SHA-256 checksums are recorded in ADR 0004.

Gitignored artifacts the notebook expects to find via Drive sync:

| Path | Source on the handover drive |
| :--- | :--- |
| `Model/Dig-Data_Model-Main/data/BIG_DATA_RAW_PROCESSED_FACE/` | `data/Processed data/BIG_DATA_RAW_PROCESSED_FACE.tar.gz`, RML subset only (720 utterance folders + `meta_one_hot_label_six_categories.pkl`, ~1.2 GB) |
| `checkpoints/base_model.pt`, `checkpoints/finetuned_model.pt` | `Final_result/Origin_training/models/RML_{origin,finetune}/` |
| `checkpoints/RML_{origin,finetune}_SHAP_value.pkl` | `Final_result/Origin_training/SHAP/SHAP_value/1225_RML_{origin,finetune}_*_SHAP_value.pkl` (8.6 MB each) |

Every one of these is checked for existence, and where possible for identity, by a halting assert in Day 0. **Wait for Drive sync to finish before running** — a partially synced folder is indistinguishable from missing data to the loader, which is why Day 0's error messages say so explicitly.

### Running

The experiment is designed to run on Google Colab with a GPU runtime. Open `multimodal-causal-ablation-v3.ipynb`, connect to at least a T4 GPU instance, and execute cells sequentially from the top. It will run into some errors because of colab environment conflicts/issues. If that happens, I just restart colab session and run go through all cells again.

Day 0 is a sequence of halting gates, in this order. Each one stops the notebook rather than warning and continuing. All five passed on the 2026-08-12 cold run:

1. **Checkpoint identity** — SHA-256 of both `.pt` files must match ADR 0004's recorded hashes.
2. **Data integrity** — every `audio.wav` and every video frame path the loader will construct must exist, across all 720 folders, with no stray files that would shift the frame count.
3. **Fidelity** — both checkpoints must reproduce 76.39% / 79.86% test accuracy within 1 percentage point. There is no label-permutation solver; the label order is the fixed constant from `getEmotionDict()`.
4. **Dominant modality** — recomputed from the DeepSHAP pickles, and required to rank audio first per neuron for every class in both models.
5. **Falsification pair** — a mechanical proof that the ablation hook writes through, then a full-64 knockout (must move accuracy) and a random-5 control (must not).

If a cell halts, read its message before changing anything: each `HALT` states what to check and, where relevant, what *not* to assume (a Colab Drive-mount glitch looks exactly like missing data).

## Lab Notebook

I maintain a daily experiment journal in [`journals/`](journals/). Each entry follows a structured scientific format documenting objectives, runtime state, execution parameters, empirical findings, debugging logs, interpretive analysis, and a handoff checklist for resuming work on the next session.

## Key Decisions

Methodology and infrastructure choices are documented as architectural decision records:

- [ADR 0001 — Causal Validation Methodology](docs/adr/0001-causal-validation-methodology.md): Modality selection protocol, probe thresholds, ablation parameters, and substrate outcome taxonomy.
- [ADR 0002 — Infrastructure & Journaling Protocol](docs/adr/0002-infrastructure-and-journaling-protocol.md): Git scope, artifact storage, session lifecycle, and deterministic seed policy.
- [ADR 0004 — V3 Restart: Data Provenance & Fidelity Gate](docs/adr/0004-v3-restart-data-provenance-and-fidelity-gate.md): Root-cause diagnosis of the v1/v2 failures, canonical data source, and the halting Day 0 fidelity gate. Supersedes both prior `0003` ADRs.
- [ADR 0005 — Day 0 Dominant-Modality Recompute & Protocol Alignment](docs/adr/0005-day0-dominant-modality-recompute-and-protocol-alignment.md): Recomputed dominant modality (audio, per neuron — not a tie), Day/Week protocol alignment, the frame-path integrity gate, the exact fast path for Weeks 2–3, and the ADR 0001 / `CONTEXT.md` taxonomy conflict. Extends ADR 0004.
- [ADR 0006 — Week 3 Transfer Null & Band Withdrawal](docs/adr/0006-week3-transfer-null-and-band-withdrawal.md): Withdraws the substrate-preservation bands (they separated nothing — every class landed above 1 under any scheme), replaces them with a measured random-ablation null (0/6 classes clear it), and switches the primary measure from argmax accuracy to probability drop since the 144-sample test split quantizes accuracy coarser than every measured effect.

## Acknowledgments

This work is part of a collaboration with Prof. KC Lan and Tsung-Yi Ko on the paper *"The Effect of Transfer Learning on Modality Competition in Multimodal Emotion Recognition."* I am responsible for Section VI-D: neuron-level causal validation of the modality-alignment finding.
