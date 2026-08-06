# Experiment Journal — Table of Contents

Chronological index of daily lab notebook entries for the Section VI-D neuron-level causal validation experiment.

| Date | Phase | Status | Summary |
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
