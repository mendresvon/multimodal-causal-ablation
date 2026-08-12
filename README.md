# Neuron-Level Causal Validation of Modality Competition in Multimodal Emotion Recognition

This repository contains my implementation of Section VI-D for a paper on transfer learning effects in multimodal emotion recognition, co-authored with Prof. KC Lan and Tsung-Yi Ko. The central question: when a multimodal model learns to recognize emotions from audio, text, and video simultaneously, does it develop specialized neurons for the modality it relies on most, and do those neurons survive fine-tuning?

**Status as of 2026-08-12:** the experiment is complete and the headline answer is negative. The base model's top neurons do no more damage inside the fine-tuned model than arbitrary neurons do. Remaining work is editorial. Full detail in [ADR 0006](docs/adr/0006-week3-transfer-null-and-band-withdrawal.md).

> **Note on running this yourself.** The dataset, checkpoints and SHAP pickles are not in this repository — they are ~1.5 GB and live on Google Drive and an external handover drive. Without them the notebook halts at Day 0 by design. What is reproducible from a clone alone: the methodology (ADRs), the recorded results (`results/`), and the full experimental record (`journals/`). See [Reproducibility](#reproducibility).

## Motivation

Multimodal emotion recognition models fuse audio, text, and visual signals to classify emotional states. Prior work by my co-authors showed that during training, modalities *compete*: the model suppresses weaker modalities in favor of a dominant one. That observation was correlational.

My contribution is the causal test. I identify the specific neurons responsible for the dominant modality's predictions, ablate them, and measure whether per-class prediction confidence actually drops. I then test whether fine-tuning keeps those same neurons load-bearing, by ablating the base model's top neurons inside the fine-tuned model and comparing the damage against a random-ablation null.

## Methodology

The experiment follows the four-week protocol Prof. KC Lan supplied (`experiment_protocol[From Prof KC Lan].docx`, not tracked — unpublished working document), with methodology decisions recorded in [ADR 0001](docs/adr/0001-causal-validation-methodology.md). `multimodal-causal-ablation-v3.ipynb` is organised by that protocol's Day/Week schedule:

- **Day 0 — Setup & gates:** Verify checkpoint identity by SHA-256; confirm every data file the loader will read exists; reproduce the paper's published RML test accuracy from the canonical data source or halt; identify the dominant modality from DeepSHAP attribution; prove the ablation hook has a real, appropriately scaled causal effect using a full-modality knockout and a random-neuron control.
- **Week 1 (Days 1–5) — Activation extraction & neuron ranking:** Cache the dominant modality's CLS activations for every sample under both models, fit L1-penalized logistic probes per emotion class on the train split only, and sanity-check them on the held-out test split.
- **Week 2 (Days 6–10) — Causal ablation:** Replace the top-k most class-selective neurons with their train-set mean activations during inference. A target-class drop of at least 2.5× the mean absolute non-target drop marks those neurons as causally class-selective rather than merely correlated.
- **Week 3 (Days 11–15) — Preservation vs. reassignment:** Compare per-class selectivity vectors between models by cosine similarity, then run the decisive test: ablate the base model's top neurons inside the fine-tuned model and compute the Transfer Retention Ratio against a measured random-ablation null rather than a fixed threshold ([ADR 0006](docs/adr/0006-week3-transfer-null-and-band-withdrawal.md)).
- **Week 4 (Days 16–21+) — Write-up:** Publication tables and figures, then the Section VI-D text for the IEEE submission.

## Status by protocol stage

| Protocol stage | Status | Summary |
| :--- | :--- | :--- |
| Day 0 — Setup & gates | ✅ PASSED | Checkpoint SHA, data integrity, fidelity (76.39% / 79.86%, both exact) and falsification pair all pass; reproduced from a cold run on 2026-08-12 |
| Day 0 step 2 — Dominant modality | ✅ RECOMPUTED | Audio, unanimous across 6 classes × 2 models on per-neuron `mean(\|φ\|)` (5.2× / 6.9× over text). Not a tie — see [ADR 0005](docs/adr/0005-day0-dominant-modality-recompute-and-protocol-alignment.md) |
| Week 1 — Activations & probes | ✅ COMPLETE | CLS activations cached, L1-logistic probes fit and sanity-checked on held-out test split (`results/week1_probe_auc.json`) |
| Week 2 — Causal ablation | ✅ COMPLETE | Mean-ablation sweep run for both models (`results/week2_{base,finetuned}_ablation_sweep.json`). Probability drop, not accuracy, is the primary measure: the 144-sample test split quantizes accuracy to 4.17pp steps, coarser than every measured effect |
| Week 3 — Transfer retention | ✅ COMPLETE — negative result | Substrate bands withdrawn; tested against a measured random-ablation null instead, and 0 of 6 classes clear it. Report as "no transfer signal above random ablation," never as shared substrate ([ADR 0006](docs/adr/0006-week3-transfer-null-and-band-withdrawal.md)) |
| Week 4 — Results & paper integration | 🟨 IN PROGRESS | Tables VII–X filled from real artifacts in [`docs/vi-d-paper-writeup-guidance.md`](docs/vi-d-paper-writeup-guidance.md). Remaining work is editorial: 4 unwritten prose blocks, 2 table headers need rewording, Table IX caption and Section IV-C wording need alignment |

### How the current numbers were arrived at

**2026-08-09 — restart.** All prior numbers, from both `multimodal-causal-ablation.ipynb` and `-v2.ipynb`, are retracted. The notebook had been evaluating correct, SHA-verified checkpoints against a fabricated `meta.pkl` rather than the canonical merged-corpus data source. Diagnosis in [ADR 0004](docs/adr/0004-v3-restart-data-provenance-and-fidelity-gate.md) and [`journals/2026-08-09-session1.md`](journals/2026-08-09-session1.md); the subsequent hardening and protocol re-alignment in [ADR 0005](docs/adr/0005-day0-dominant-modality-recompute-and-protocol-alignment.md) and [`journals/2026-08-09-session2.md`](journals/2026-08-09-session2.md).

**2026-08-11 — v3 run to completion.** Weeks 1–4 ran end to end on real data. Every class's raw transfer ratio sat above 1 under any band scheme, which initially looked like preservation but turned out to be the default state of a more-fragile fine-tuned model. Against a measured null, 0 of 6 classes clear it. See [`journals/2026-08-11-session2.md`](journals/2026-08-11-session2.md) and [`session3`](journals/2026-08-11-session3.md).

**2026-08-12 — reproduction check.** The whole notebook was re-run from committed code. The fidelity gate passed exactly, 8 of 9 tracked artifacts came back byte-identical and the 9th differed only by GPU float noise, and both Week 3 negative results reproduced exactly. See [`journals/2026-08-12-session1.md`](journals/2026-08-12-session1.md).

## Repository layout

```
multimodal-causal-ablation-v3.ipynb   Main experiment notebook (Colab GPU)
multimodal-causal-ablation{,-v2}.ipynb  Retracted v1/v2 runs, kept as evidence for ADR 0004
CONTEXT.md                            Domain glossary and canonical terminology
docs/adr/                             Architectural decision records
docs/provenance/                      Upstream training logs and the previous team's SHAP notebook
results/                              All numbered artifacts from the 2026-08-12 cold run
figures/                              Generated plots
journals/                             Daily lab notebook, index, and errata
src/utils.py                          set_deterministic_seed(seed=0)
Model/Dig-Data_Model-Main/            Vendored upstream MME2E model code (third party)
```

### Key files

| File | Purpose |
| :--- | :--- |
| `docs/vi-d-paper-writeup-guidance.md` | Section VI-D tables (VII–X) from real artifacts, rigor verdict, and what is still unwritten |
| `docs/v3-downstream-statistics-audit.md` | Audit of the downstream statistics pipeline that motivated the probability-drop measure and the random-ablation null |
| `docs/provenance/RML_{origin,finetune}_training_log.txt` | Original `main.py` stdout for both checkpoints; the source for the 76.39% / 79.86% fidelity targets and every `MODEL_ARGS` value |
| `docs/provenance/upstream_SHAP_analysis.ipynb` | The previous researchers' DeepSHAP notebook; documents the 1152-d `[text 1024 \| video 64 \| audio 64]` layout |
| `docs/rml_file_manifest.json` | Per-folder file counts for all 720 RML utterance folders, snapshot 2026-08-09; Day 0's secondary drift check |
| `checkpoints/README.md` | SHA-256 checksums for model weights and SHAP pickles |
| `data/README.md` | Dataset provenance, canonical source, and extraction layout |
| `journals/README.md` | Chronological index of all journal entries |
| `journals/ERRATA.md` | Corrections raised against journal entries after the fact |

## Reproducibility

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (experiments run on Google Colab T4/V100)
- PyTorch 2.2.2, pinned for C++ ABI compatibility with the upstream model code

### Data & checkpoints

The RML dataset, model checkpoints, and DeepSHAP pickles are too large for Git and live on Google Drive and an external handover drive only. The canonical data source is `BIG_DATA_RAW_PROCESSED_FACE.tar.gz` filtered to RML split IDs; do not use the retired `RML_RAW_PROCESSED_Face` folder ([ADR 0004](docs/adr/0004-v3-restart-data-provenance-and-fidelity-gate.md), which also records the checkpoint SHA-256 checksums).

Untracked artifacts the notebook expects to find via Drive sync:

| Path | Source on the handover drive |
| :--- | :--- |
| `Model/Dig-Data_Model-Main/data/BIG_DATA_RAW_PROCESSED_FACE/` | `data/Processed data/BIG_DATA_RAW_PROCESSED_FACE.tar.gz`, RML subset only (720 utterance folders + `meta_one_hot_label_six_categories.pkl`, ~1.2 GB) |
| `checkpoints/base_model.pt`, `checkpoints/finetuned_model.pt` | `Final_result/Origin_training/models/RML_{origin,finetune}/` |
| `checkpoints/RML_{origin,finetune}_SHAP_value.pkl` | `Final_result/Origin_training/SHAP/SHAP_value/1225_RML_{origin,finetune}_*_SHAP_value.pkl` (8.6 MB each) |

Each of these is checked for existence, and where possible for identity, by a halting assert in Day 0. **Wait for Drive sync to finish before running.** A partially synced folder is indistinguishable from missing data to the loader, which is why Day 0's error messages say so explicitly.

### Running

Open `multimodal-causal-ablation-v3.ipynb` in Google Colab, connect a T4 or better GPU runtime, and execute cells sequentially from the top.

Colab's preinstalled packages conflict with the pinned PyTorch 2.2.2 often enough that a first pass may fail on import. Restarting the runtime and re-running from the top resolves it; the notebook is written to be safely re-runnable from cell 1.

Day 0 is a sequence of halting gates, in this order. Each stops the notebook rather than warning and continuing. All five passed on the 2026-08-12 cold run:

1. **Checkpoint identity** — SHA-256 of both `.pt` files must match ADR 0004's recorded hashes.
2. **Data integrity** — every `audio.wav` and every video frame path the loader will construct must exist across all 720 folders, with no stray files that would shift the frame count.
3. **Fidelity** — both checkpoints must reproduce 76.39% / 79.86% test accuracy within 1 percentage point. There is no label-permutation solver; the label order is the fixed constant from `getEmotionDict()`.
4. **Dominant modality** — recomputed from the DeepSHAP pickles, required to rank audio first per neuron for every class in both models.
5. **Falsification pair** — a mechanical proof that the ablation hook writes through, then a full-64 knockout (must move accuracy) and a random-5 control (must not).

If a cell halts, read its message before changing anything. Each `HALT` states what to check and, where relevant, what *not* to assume — a Colab Drive-mount glitch looks exactly like missing data.

## Lab notebook

I keep a daily experiment journal in [`journals/`](journals/). Each entry records objectives, runtime state, execution parameters, empirical findings, debugging logs, interpretation, and a handoff checklist for the next session. Entries are not edited after the fact; corrections are recorded in [`journals/ERRATA.md`](journals/ERRATA.md).

Entries from July 27 to August 7 use a Phase A–E vocabulary that predates the v3 restart and their numbers are retracted. They are kept because they are the evidence base for the ADR 0004 diagnosis.

## Key decisions

Methodology and infrastructure choices are recorded as architectural decision records in [`docs/adr/`](docs/adr/):

- **[ADR 0001](docs/adr/0001-causal-validation-methodology.md)** — Causal validation methodology: modality selection, probe thresholds, ablation parameters, substrate outcome taxonomy.
- **[ADR 0002](docs/adr/0002-infrastructure-and-journaling-protocol.md)** — Infrastructure and journaling: Git scope, artifact storage, session lifecycle, deterministic seed policy.
- **ADR 0003** — Two records were written under this number on different days (`0003-targeting-cls-tokens-for-ablation.md` and `0003-causal-ablation-v2-methodology-remediation.md`). Both are superseded by ADR 0004 and retained only as history. The numbering collision is left as-is rather than rewritten.
- **[ADR 0004](docs/adr/0004-v3-restart-data-provenance-and-fidelity-gate.md)** — V3 restart: root-cause diagnosis of the v1/v2 failures, canonical data source, halting Day 0 fidelity gate. Supersedes both 0003 records.
- **[ADR 0005](docs/adr/0005-day0-dominant-modality-recompute-and-protocol-alignment.md)** — Dominant-modality recompute, Day/Week protocol alignment, frame-path integrity gate, exact fast path for Weeks 2–3, and resolution of the ADR 0001 / `CONTEXT.md` taxonomy conflict. Extends ADR 0004.
- **[ADR 0006](docs/adr/0006-week3-transfer-null-and-band-withdrawal.md)** — Withdraws the Week 3 substrate-preservation bands, replaces them with a measured random-ablation null, and switches the primary measure from argmax accuracy to probability drop.

## Acknowledgments

This work is part of a collaboration with Prof. KC Lan and Tsung-Yi Ko on the paper *"The Effect of Transfer Learning on Modality Competition in Multimodal Emotion Recognition."* I am responsible for Section VI-D: neuron-level causal validation of the modality-alignment finding.

`Model/Dig-Data_Model-Main/` contains the upstream MME2E implementation from the previous research team, vendored unmodified so the checkpoints load against the exact code that trained them. I did not write it.
