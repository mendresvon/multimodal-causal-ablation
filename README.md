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

## Repository Structure

```
multimodal-emotion-causal-validation/
├── multimodal-emotion-causal-validation.ipynb  # Main experiment notebook (runs on Colab GPU)
├── src/
│   └── utils.py                                # Deterministic seed utility
├── Model/Dig-Data_Model-Main/
│   ├── src/                                    # Upstream model architecture (PyTorch)
│   │   ├── models/e2e.py                       # End-to-end multimodal model definition
│   │   ├── datasets.py                         # Data loading and preprocessing
│   │   └── ...
│   └── main.py                                 # Training entry point
├── checkpoints/                                # Model weights & SHAP pickles (not in Git; see README)
├── data/                                       # RML dataset (not in Git; see README)
├── results/                                    # Numerical experiment outputs (CSV, JSON)
├── figures/                                    # Publication-quality plots (PNG, SVG)
├── journals/                                   # Daily lab notebook entries
├── docs/adr/                                   # Architectural decision records
├── CONTEXT.md                                  # Domain glossary and canonical terminology
└── requirements.txt                            # Pinned Python dependencies
```

## Reproducibility

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (experiments run on Google Colab T4/V100)
- PyTorch 2.2.2 (pinned for C++ ABI compatibility with upstream model code)

### Setup

```bash
pip install -r requirements.txt
```

### Data & Checkpoints

The RML dataset (~3.2 GB) and model checkpoints (~173 MB each) are too large for Git. See [`data/README.md`](data/README.md) for acquisition instructions and [`checkpoints/README.md`](checkpoints/README.md) for SHA-256 integrity checksums.

### Running

The experiment is designed to run on Google Colab with a GPU runtime. Open `multimodal-emotion-causal-validation.ipynb`, connect to a T4 or V100 instance, and execute cells sequentially. Each phase includes resume gates that detect cached intermediate results, so you don't need to re-run expensive forward passes after a runtime reset.

## Lab Notebook

I maintain a daily experiment journal in [`journals/`](journals/). Each entry follows a structured scientific format documenting objectives, runtime state, execution parameters, empirical findings, debugging logs, interpretive analysis, and a handoff checklist for resuming work on the next session.

## Key Decisions

Methodology and infrastructure choices are documented as architectural decision records:

- [ADR 0001 — Causal Validation Methodology](docs/adr/0001-causal-validation-methodology.md): Modality selection protocol, probe thresholds, ablation parameters, and substrate outcome taxonomy.
- [ADR 0002 — Infrastructure & Journaling Protocol](docs/adr/0002-infrastructure-and-journaling-protocol.md): Git scope, artifact storage, session lifecycle, and deterministic seed policy.

## Acknowledgments

This work is part of a collaboration with Prof. KC Lan and Tsung-Yi Ko on the paper *"The Effect of Transfer Learning on Modality Competition in Multimodal Emotion Recognition."* I am responsible for Section VI-D: neuron-level causal validation of the modality-alignment finding.
