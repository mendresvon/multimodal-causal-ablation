# Experiment Journal — Table of Contents

Chronological index of daily lab notebook entries for the Section VI-D neuron-level causal validation experiment.

| Date | Stage | Status | Summary |
| :--- | :--- | :--- | :--- |
| [2026-07-27](2026-07-27.md) | Setup | `[COMPLETE]` | Data acquisition via Kaggle, Colab GPU setup, PyTorch ABI debugging |
| [2026-07-29 (session 1)](2026-07-29-session1.md) | Setup | `[COMPLETE]` | Project restructuring & persistence architecture |
| [2026-07-29 (session 2)](2026-07-29-session2.md) | Setup | `[COMPLETE]` | Checkpoint discovery, preliminary modality resolution (pending re-verification) |
| [2026-07-30 (session 1)](2026-07-30-session1.md) | Setup | `[COMPLETE]` | Infrastructure backbone: Git repo, journaling protocol, artifact architecture, deterministic seed |
| [2026-07-30 (session 2)](2026-07-30-session2.md) | Setup | `[COMPLETE]` | Agent continuity & context preservation, AGENTS.md rules |
| [2026-07-31](2026-07-31.md) | Phase A | `[COMPLETE]` | Notebook restructuring, Phase A DeepSHAP aggregated attribution execution & Dominant Modality verdict |
| [2026-08-02](2026-08-02.md) | Phase B | `[COMPLETE]` | Fixed dataset mismatch, identified Text branch has no FFN (Tier 1 = Tier 2), successful L1-logistic validation on 1024-d ALBERT CLS token |
| [2026-08-03](2026-08-03.md) | Phase C | `[PREPARED]` | Drafted Colab notebook code for class-selectivity ratio computation and ablation hooks |
| [2026-08-04 (session 1)](2026-08-04-session1.md) | Phase C | `[COMPLETE]` | Mean-ablation sweep k={1,3,5,10} evaluated; extreme substrate dispersion observed |
| [2026-08-04 (session 2)](2026-08-04-session2.md) | Phase D | `[COMPLETE]` | Initial cross-ablation done; methodology audit found 16 issues requiring remediation |
| [2026-08-04 (session 3)](2026-08-04-session3.md) | Phase A | `[COMPLETE]` | Implemented and executed incremental batch saving for P1 fix; saved 73/90 batches |
| [2026-08-04 (session 4)](2026-08-04-session4.md) | Setup | `[COMPLETE]` | Journal cleanup, erratum additions, roadmap reset to Phase A |
| [2026-08-05 (session 1)](2026-08-05-session1.md) | Phase A | `[PREPARED]` | Final code scrutiny for Phase A/B/C/D execution, fixed caching issue and ablation targets |
| [2026-08-05 (session 2)](2026-08-05-session2.md) | Phase C | `[COMPLETE]` | Identified critical bug: causal ablation proxy inference was evaluating the training set; requires slicing to test set |
| [2026-08-06 (session 1)](2026-08-06.md) | Phase D | `[COMPLETE]` | Experiment concluded. Discovered that multimodal redundancy suppresses causal ablation drops despite high SHAP attribution |
| [2026-08-06 (session 2)](2026-08-06-session2.md) | Audit | `[COMPLETE]` | Comprehensive quadruple-check audit against Prof KC Lan's protocol; 7 critical/high issues identified including SHAP formula bias |
| [2026-08-06 (session 3)](2026-08-06-session3.md) | Phase A–D | `[COMPLETE]` | Executed audit remediation: mean(|phi|) SHAP formula (Audio 75.9% dominant), dose-response k=1..64 sweeps, full 720-sample eval, and non-zero retention ratios |
| [2026-08-07](2026-08-07.md) | Remediation | `[COMPLETE]` | Remediated pipeline v2 architecture (zero-leakage 80/20 train/test probing, ADR 0003, epsilon-screened retention taxonomy, notebook blueprint v2) |
| [2026-08-09 (session 1)](2026-08-09.md) | Restart | `[COMPLETE]` | Diagnosed root cause of v1/v2 garbage results (fabricated meta.pkl, correct checkpoints); wrote ADR 0004, built `multimodal-causal-ablation-v3.ipynb` with a halting fidelity gate and falsification pair; confirmed canonical data source and RML ID intersection |
| [2026-08-09 (session 2)](2026-08-09-session2.md) | Pre-flight | `[COMPLETE]` | Pre-run audit of v3: recomputed dominant modality from DeepSHAP (audio, 5.2x/6.9x per neuron — not a tie), added Day 0 frame-path integrity gate, proved an exact fast path for Weeks 2–3, fixed 4 run-killing notebook bugs, wrote ADR 0005, realigned notebook and README to the protocol's Day/Week schedule |
| [2026-08-09 (session 3)](2026-08-09-session3.md) | Housekeeping | `[COMPLETE]` | Directory audit: confirmed `data/`, `journals/`, and empty `results/`/`figures/`/`checkpoints/activations/` are all empty/flat by design; kept both `0003` ADRs and retracted v1/v2 notebooks as documented historical record; removed stray `.DS_Store` files and an already-merged local branch |
