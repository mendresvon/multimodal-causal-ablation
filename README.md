# Neuron-Level Causal Validation of Modality Competition in Multimodal Emotion Recognition

This repository contains my implementation of Section VI-D for a paper on transfer learning effects in multimodal emotion recognition, co-authored with Prof. KC Lan and Tsung-Yi Ko. The central question: when a multimodal model learns to recognize emotions from audio, text, and video simultaneously, does it develop specialized neurons for the modality it relies on most — and do those neurons survive fine-tuning?

## Motivation

Multimodal emotion recognition models fuse audio, text, and visual signals to classify emotional states. Prior work by my co-authors demonstrated that during training, modalities *compete* — the model suppresses weaker modalities in favor of a dominant one. But that observation was correlational. My contribution is the causal test: I identify the specific neurons responsible for the dominant modality's predictions, ablate them, and measure whether the model's accuracy for individual emotion classes actually drops. Then I check whether fine-tuning preserves, reassigns, or disperses those same neurons.

## Methodology

The experiment follows a five-phase protocol, formally documented in [ADR 0001](docs/adr/0001-causal-validation-methodology.md):

- **Phase A — Dominant Modality Verification:** Confirm which input modality (audio, text, or visual) the trained model relies on most, using aggregated DeepSHAP attribution scores.
- **Phase B — Probe Signal Validation:** Fit L1-penalized logistic regression probes on cached neuron activations to verify that the dominant modality's FFN layer encodes linearly separable emotion signals. If probe quality is too low (mean AUC < 0.65), fall back through a defined layer hierarchy.
- **Phase C — Causal Ablation:** Replace the top-k most class-selective neurons with their dataset-mean activations during inference. If the target emotion class accuracy drops by at least 2.5× the average non-target class drop, those neurons are causally class-selective — not just correlated.
- **Phase D — Transfer Retention Analysis:** Ablate the base model's top neurons inside the fine-tuned model to compute a Transfer Retention Ratio. This classifies the fine-tuning outcome as Substrate Preservation (neurons sharpened), Substrate Reassignment (representation shifted to different neurons), or Substrate Dispersion (representation became distributed).
- **Phase E — Results & Paper Integration:** Generate publication figures and tables for the IEEE submission.

## Phase Roadmap (Live Status)

This table tracks experiment progress. Update it at the end of every session.

**2026-08-09 restart:** all prior Phase A-D numbers (both `multimodal-causal-ablation.ipynb` and `-v2.ipynb`) are retracted. Root cause: the notebook evaluated the correct, SHA-verified checkpoints against a fabricated `meta.pkl` instead of the canonical merged-corpus data source. See [ADR 0004](docs/adr/0004-v3-restart-data-provenance-and-fidelity-gate.md) and [`journals/2026-08-09.md`](journals/2026-08-09.md) for the full diagnosis.

| Phase | Status | Summary |
| :--- | :--- | :--- |
| A — Dominant Modality Verification | ⬜ NOT STARTED (v3) | Prior verdict retracted; will recompute once Day 0 fidelity gate passes |
| B — Probe Signal Validation | ⬜ NOT STARTED (v3) | Prior numbers retracted (fit on mismatched data) |
| C — Causal Ablation | ⬜ NOT STARTED (v3) | Prior sweep retracted; blocked on Day 0 fidelity gate + falsification pair |
| D — Transfer Retention Analysis | ⬜ NOT STARTED (v3) | Prior taxonomy retracted; ε=0.05 screen withdrawn pending real drops |
| E — Results & Paper Integration | ⬜ PENDING | Tables VII/VIII/IX, Section VI-D text |

## Key Files Reference

| File | Purpose |
| :--- | :--- |
| `multimodal-causal-ablation-v3.ipynb` | Main experiment notebook (in progress) — runs on Google Colab GPU |
| `CONTEXT.md` | Domain glossary with canonical terminology. |
| `docs/adr/0001-causal-validation-methodology.md` | Locked methodology decisions: thresholds, fallback tiers, taxonomy |
| `docs/adr/0002-infrastructure-and-journaling-protocol.md` | Infrastructure decisions: Git scope, journals, artifact naming, seed policy |
| `docs/adr/0004-v3-restart-data-provenance-and-fidelity-gate.md` | Root-cause diagnosis of the v1/v2 failures, canonical data source, halting fidelity gate |
| `checkpoints/README.md` | SHA-256 checksums for model weights and SHAP pickles |
| `journals/GUIDELINES.md` | Internal journal writing template and tone rules |
| `journals/README.md` | Chronological index of all journal entries with phase and status |
| `src/utils.py` | Contains `set_deterministic_seed(seed=0)` |
## Reproducibility

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (experiments run on Google Colab T4/V100)
- PyTorch 2.2.2 (pinned for C++ ABI compatibility with upstream model code)

### Data & Checkpoints

The RML dataset and model checkpoints are too large for Git and live on Google Drive / an external handover drive only. Canonical source: `BIG_DATA_RAW_PROCESSED_FACE.tar.gz` filtered to RML split IDs (see [ADR 0004](docs/adr/0004-v3-restart-data-provenance-and-fidelity-gate.md) — do not use the retired `RML_RAW_PROCESSED_Face` folder). Checkpoint SHA-256 checksums are recorded in ADR 0004.

### Running

The experiment is designed to run on Google Colab with a GPU runtime. Open `multimodal-causal-ablation-v3.ipynb`, connect to a T4 or V100 instance, and execute cells sequentially. Day 0 includes a halting fidelity gate — if base/fine-tuned test accuracy don't reproduce the paper's published RML numbers (76.39% / 79.86%) within tolerance, the notebook stops there rather than proceeding on unverified data.

## Lab Notebook

I maintain a daily experiment journal in [`journals/`](journals/). Each entry follows a structured scientific format documenting objectives, runtime state, execution parameters, empirical findings, debugging logs, interpretive analysis, and a handoff checklist for resuming work on the next session.

## Key Decisions

Methodology and infrastructure choices are documented as architectural decision records:

- [ADR 0001 — Causal Validation Methodology](docs/adr/0001-causal-validation-methodology.md): Modality selection protocol, probe thresholds, ablation parameters, and substrate outcome taxonomy.
- [ADR 0002 — Infrastructure & Journaling Protocol](docs/adr/0002-infrastructure-and-journaling-protocol.md): Git scope, artifact storage, session lifecycle, and deterministic seed policy.
- [ADR 0004 — V3 Restart: Data Provenance & Fidelity Gate](docs/adr/0004-v3-restart-data-provenance-and-fidelity-gate.md): Root-cause diagnosis of the v1/v2 failures, canonical data source, and the halting Day 0 fidelity gate. Supersedes both prior `0003` ADRs.

## Acknowledgments

This work is part of a collaboration with Prof. KC Lan and Tsung-Yi Ko on the paper *"The Effect of Transfer Learning on Modality Competition in Multimodal Emotion Recognition."* I am responsible for Section VI-D: neuron-level causal validation of the modality-alignment finding.
