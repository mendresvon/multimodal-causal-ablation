# Experiment Journal: Table of Contents

Chronological index of daily lab notebook entries for the Section VI-D neuron-level causal validation experiment.

**This journal is closed.** The repository was frozen on 2026-08-19; the final entry is [2026-08-19 (session 1)](2026-08-19-session1.md). Active work continues in the successor repository `neuron-feature-mapping`, which keeps its own journal under a revised protocol.

Entries are never edited after the fact. Corrections raised against an entry are recorded in [`ERRATA.md`](ERRATA.md).

**Note on the July 27 – August 7 entries.** They use a Phase A–E vocabulary that predates the v3 restart, and their numbers are retracted, the pipeline they describe was evaluating correct checkpoints against a fabricated `meta.pkl`. See [ADR 0004](../docs/adr/0004-v3-restart-data-provenance-and-fidelity-gate.md). They are kept because they are the evidence base for that diagnosis. Everything from August 9 onward is on real data and uses the protocol's Day/Week vocabulary.

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
| [2026-08-06 (session 2)](2026-08-06-session2.md) | Audit | `[COMPLETE]` | Line-by-line audit against Prof KC Lan's protocol; 7 critical/high issues identified including SHAP formula bias |
| [2026-08-06 (session 3)](2026-08-06-session3.md) | Phase A–D | `[COMPLETE]` | Executed audit remediation: mean(|phi|) SHAP formula (Audio 75.9% dominant), dose-response k=1..64 sweeps, full 720-sample eval, and non-zero retention ratios |
| [2026-08-07](2026-08-07-session1.md) | Remediation | `[COMPLETE]` | Remediated pipeline v2 architecture (zero-leakage 80/20 train/test probing, ADR 0003, epsilon-screened retention taxonomy, notebook blueprint v2) |
| [2026-08-09 (session 1)](2026-08-09-session1.md) | Restart | `[COMPLETE]` | Diagnosed root cause of v1/v2 garbage results (fabricated meta.pkl, correct checkpoints); wrote ADR 0004, built `multimodal-causal-ablation-v3.ipynb` with a halting fidelity gate and falsification pair; confirmed canonical data source and RML ID intersection |
| [2026-08-09 (session 2)](2026-08-09-session2.md) | Pre-flight | `[COMPLETE]` | Pre-run audit of v3: recomputed dominant modality from DeepSHAP (audio, 5.2x/6.9x per neuron: not a tie), added Day 0 frame-path integrity gate, proved an exact fast path for Weeks 2–3, fixed 4 run-killing notebook bugs, wrote ADR 0005, realigned notebook and README to the protocol's Day/Week schedule |
| [2026-08-10](2026-08-10-session1.md) | Data | `[COMPLETE]` | Traced the 97-problem data-integrity failure to Google Drive stream-mount sync conflicts, not data loss; re-extracted all 720 RML folders from the external handover archive and rebuilt the Drive copy via the web UI (720/720 IDs verified, no duplicate objects) |
| [2026-08-11 (session 1)](2026-08-11-session1.md) | Day 0 | `[COMPLETE]` | Day 0 fidelity gate PASSED (base 76.39%, fine-tuned 79.86%, both exact) after switching the audio branch from `audio.wav` at 44100 Hz to the training run's `audio_16000.wav`, shipped as `data_bundle/RML_audio16k_all720.tar`. Entry reconstructed 2026-08-12; the sample-rate route was disputed and resolved: see [E1](ERRATA.md#e1--2026-08-11-session-1-the-audio-sample-rate-route) |
| [2026-08-11 (session 2)](2026-08-11-session2.md) | Weeks 1–4 | `[COMPLETE]` | Ran the v3 notebook end to end, audited the downstream statistics, replaced argmax accuracy with a continuous probability-drop measure, added a random-ablation null for the Week 3 transfer ratio, and withdrew the substrate bands (ADR 0006) |
| [2026-08-11 (session 3)](2026-08-11-session3.md) | Week 3 | `[COMPLETE]` | Ran the random-ablation null: 0 of 6 classes have a transfer ratio whose 95% CI clears it, so Week 3 is a negative result; wrote ADR 0006, added the Day 13b probe-rank check (probe top-5 under-damages a measured top-5 in all 12 pairs), and brought the repository to a clean tree |
| [2026-08-12 (session 1)](2026-08-12-session1.md) | Reproduction | `[COMPLETE]` | Ran the restored v3 notebook end to end from committed code: fidelity gate PASSED (76.39% / 79.86%), 8 of 9 tracked artifacts byte-identical to HEAD and the 9th differing only by GPU float noise ~130x inside tolerance. Fixed the AppleDouble sidecar halt in the 16 kHz overlay, and closed the audio-route provenance question via the Day 0 step 2 SHAP cross-check (r = 1.000000) rather than artifact identity. Both Week 3 negative results reproduce exactly |
| [2026-08-19 (session 1)](2026-08-19-session1.md) | Closeout | `[COMPLETE]` | Closed the repository. Recorded the review outcome: Prof. KC Lan rejected Table IX as adding no meaning to the paper and assigned the FACS/eGeMAPS replacement direction; Tables VII and VIII stand. Committed the two untracked documents, including the professor briefing that was the fullest account of the experiment and had never been staged. Established that the 24/25 branch divergence was duplicate commit objects over an identical tree, and that the notebook's 3479-line diff was Colab reflowing the JSON onto one line, not a stripped file |
