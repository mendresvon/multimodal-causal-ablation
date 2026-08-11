# Experiment Journal — Table of Contents

Chronological index of daily lab notebook entries for the Section VI-D neuron-level causal validation experiment.

| Date | Stage | Status | Summary |
| :--- | :--- | :--- | :--- |
| [2026-07-27](2026-07-27-session1.md) | Setup | `[COMPLETE]` | Data acquisition via Kaggle, Colab GPU setup, PyTorch ABI debugging |
| [2026-07-29 (session 1)](2026-07-29-session1.md) | Setup | `[COMPLETE]` | Project restructuring & persistence architecture |
| [2026-07-29 (session 2)](2026-07-29-session2.md) | Setup | `[COMPLETE]` | Checkpoint discovery, preliminary modality resolution (pending re-verification) |
| [2026-07-30 (session 1)](2026-07-30-session1.md) | Setup | `[COMPLETE]` | Infrastructure backbone: Git repo, journaling protocol, artifact architecture, deterministic seed |
| [2026-07-30 (session 2)](2026-07-30-session2.md) | Setup | `[COMPLETE]` | Agent continuity & context preservation, AGENTS.md rules |
| [2026-07-31](2026-07-31-session1.md) | Phase A | `[COMPLETE]` | Notebook restructuring, Phase A DeepSHAP aggregated attribution execution & Dominant Modality verdict |
| [2026-08-02](2026-08-02-session1.md) | Phase B | `[COMPLETE]` | Fixed dataset mismatch, identified Text branch has no FFN (Tier 1 = Tier 2), successful L1-logistic validation on 1024-d ALBERT CLS token |
| [2026-08-03](2026-08-03-session1.md) | Phase C | `[PREPARED]` | Drafted Colab notebook code for class-selectivity ratio computation and ablation hooks |
| [2026-08-04 (session 1)](2026-08-04-session1.md) | Phase C | `[COMPLETE]` | Mean-ablation sweep k={1,3,5,10} evaluated; extreme substrate dispersion observed |
| [2026-08-04 (session 2)](2026-08-04-session2.md) | Phase D | `[COMPLETE]` | Initial cross-ablation done; methodology audit found 16 issues requiring remediation |
| [2026-08-04 (session 3)](2026-08-04-session3.md) | Phase A | `[COMPLETE]` | Implemented and executed incremental batch saving for P1 fix; saved 73/90 batches |
| [2026-08-04 (session 4)](2026-08-04-session4.md) | Setup | `[COMPLETE]` | Journal cleanup, erratum additions, roadmap reset to Phase A |
| [2026-08-05 (session 1)](2026-08-05-session1.md) | Phase A | `[PREPARED]` | Final code scrutiny for Phase A/B/C/D execution, fixed caching issue and ablation targets |
| [2026-08-05 (session 2)](2026-08-05-session2.md) | Phase C | `[COMPLETE]` | Identified critical bug: causal ablation proxy inference was evaluating the training set; requires slicing to test set |
| [2026-08-06 (session 1)](2026-08-06-session1.md) | Phase D | `[COMPLETE]` | Experiment concluded. Discovered that multimodal redundancy suppresses causal ablation drops despite high SHAP attribution |
| [2026-08-06 (session 2)](2026-08-06-session2.md) | Audit | `[COMPLETE]` | Comprehensive quadruple-check audit against Prof KC Lan's protocol; 7 critical/high issues identified including SHAP formula bias |
| [2026-08-06 (session 3)](2026-08-06-session3.md) | Phase A–D | `[COMPLETE]` | Executed audit remediation: mean(|phi|) SHAP formula (Audio 75.9% dominant), dose-response k=1..64 sweeps, full 720-sample eval, and non-zero retention ratios |
| [2026-08-07](2026-08-07-session1.md) | Remediation | `[COMPLETE]` | Remediated pipeline v2 architecture (zero-leakage 80/20 train/test probing, ADR 0003, epsilon-screened retention taxonomy, notebook blueprint v2) |
| [2026-08-09 (session 1)](2026-08-09-session1.md) | Restart | `[COMPLETE]` | Diagnosed root cause of v1/v2 garbage results (fabricated meta.pkl, correct checkpoints); wrote ADR 0004, built `multimodal-causal-ablation-v3.ipynb` with a halting fidelity gate and falsification pair; confirmed canonical data source and RML ID intersection |
| [2026-08-09 (session 2)](2026-08-09-session2.md) | Pre-flight | `[COMPLETE]` | Pre-run audit of v3: recomputed dominant modality from DeepSHAP (audio, 5.2x/6.9x per neuron — not a tie), added Day 0 frame-path integrity gate, proved an exact fast path for Weeks 2–3, fixed 4 run-killing notebook bugs, wrote ADR 0005, realigned notebook and README to the protocol's Day/Week schedule |
| [2026-08-10](2026-08-10-session1.md) | Data | `[COMPLETE]` | Traced the 97-problem data-integrity failure to Google Drive stream-mount sync conflicts, not data loss; re-extracted all 720 RML folders from the external handover archive and rebuilt the Drive copy via the web UI (720/720 IDs verified, no duplicate objects) |
| 2026-08-11 (session 1) — *entry file missing* | Day 0 | `[COMPLETE]` | Day 0 fidelity gate PASSED (base 76.39%, fine-tuned 79.86%, both exact): found the audio branch was reading `audio.wav` at 44100 Hz instead of the training run's `audio_16000.wav`, shipped the real 16 kHz clips as `data_bundle/RML_audio16k_all720.tar`, and worked around a Google Drive upload stall by rebinding the loader in the runtime |
| [2026-08-11 (session 2)](2026-08-11-session2.md) | Weeks 1–4 | `[COMPLETE]` | Ran the v3 notebook end to end, audited the downstream statistics, replaced argmax accuracy with a continuous probability-drop measure, added a random-ablation null for the Week 3 transfer ratio, and withdrew the substrate bands (ADR 0006) |
