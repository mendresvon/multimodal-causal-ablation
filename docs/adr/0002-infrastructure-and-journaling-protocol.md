# Infrastructure, Journaling Protocol, and Reproducibility Architecture

Over two days of experiment setup, I found that Google Colab's ephemeral runtimes, the multi-week experiment timeline, and the IEEE publication target required a formal infrastructure backbone before any Phase A–E execution could begin. This ADR records the decisions made during a structured grilling session on 2026-07-29/30 to establish that backbone.

## Status
Accepted

## Context

Section VI-D requires running neuron-level causal ablation experiments across multiple Colab GPU sessions over days or weeks. Each session starts from scratch, no installed packages, no cached tensors, no execution state. Without explicit infrastructure, work would be lost between sessions, results would lack provenance, and the experiment would not meet IEEE reproducibility standards.

Additionally, this repository will serve as a grad school application portfolio piece, so its public GitHub presence needs to reflect research maturity.

## Decisions & Rationale

### 1. Git Scope & Binary Exclusion
Track all code (`src/`, `Model/Dig-Data_Model-Main/src/`), documentation (`docs/`, `CONTEXT.md`), journals, lightweight results (`results/`), and figures (`figures/`) in Git. Exclude heavy binaries (model weights, datasets, SHAP pickles, cached activation tensors) via `.gitignore`. Provide SHA-256 checksum manifests in `checkpoints/README.md` and `data/README.md` so anyone can verify data integrity after acquisition.

*Why not Git LFS?* The checkpoints (~173 MB each) and dataset (~3.2 GB) exceed GitHub's free LFS quota. Google Drive is already the execution environment, so it serves as the large-file store without added cost or tooling.

### 2. Seven-Part Scientific Journal Schema
Daily entries in `journals/YYYY-MM-DD.md` follow a fixed 7-part structure: (1) Objective & Hypothesis, (2) Session & Runtime State, (3) Experimental Execution & Parameters, (4) Empirical Findings & Data Links, (5) Roadblocks & Debugging, (6) Interpretability Retrospective, (7) Handoff Checklist. A master index at `journals/README.md` tracks phase and completion status.

*Why 7 parts instead of free-form?* Colab sessions end abruptly. The Handoff Checklist (part 7) guarantees the next session can resume without guesswork. The Runtime State (part 2) captures the exact environment, so results can be attributed to specific dependency versions. Free-form entries from the first two days lacked this structure and were harder to pick up from.

### 3. Artifact Naming and Storage Convention
- `checkpoints/activations/`: cached tensors (Google Drive only, not Git-tracked)
- `results/`: lightweight CSV/JSON metrics (Git-tracked)
- `figures/`: publication plots in PNG and SVG (Git-tracked)
- Naming pattern: `{phase}_{model}_{description}.{ext}`

*Why separate results from checkpoints?* Results are small, reviewable, and diffable in Git. Cached tensors are large and regenerable from checkpoints + code. Mixing them would either bloat the repo or lose provenance.

### 4. Deterministic Seed Protocol
A `set_deterministic_seed(seed=0)` utility in `src/utils.py` pins `torch.manual_seed`, `torch.cuda.manual_seed_all`, `numpy.random.seed`, `random.seed`, and `torch.backends.cudnn.deterministic = True`. Called at the top of every Phase cell.

*Why seed=0?* Matches the upstream checkpoint naming convention (`seed0`). Using a different seed would raise questions about whether results are comparable to the original training runs.

### 5. Colab Session Lifecycle
The notebook's first cell auto-detects runtime (Colab vs. local), mounts Google Drive, verifies checkpoint existence, and installs pinned dependencies from `requirements.txt`. Each Phase cell starts with a "resume gate" that checks for cached activation tensors before re-running forward passes.

*Why resume gates?* A full forward pass over RML for activation extraction takes ~15 minutes on a T4 GPU. If the Colab runtime resets after Phase B but before Phase C, re-running Phase B from scratch wastes GPU time and risks hitting Colab's daily usage limits.

### 6. Internal-Only Files
`journals/GUIDELINES.md`, `.agents/`, `skills-lock.json`, and all `.docx` files are excluded from Git. These contain project configuration, unpublished paper drafts, and internal writing rules that would undermine the professional impression of the public repository.
