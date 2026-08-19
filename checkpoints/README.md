# Checkpoint Provenance & Integrity

These files are too large for Git and are stored on Google Drive only. To verify integrity after syncing or transferring, compare against the SHA-256 checksums below.

## Files

| File | Size | SHA-256 | Source |
| :--- | :--- | :--- | :--- |
| `base_model.pt` | 173.3 MB | `565bc220d187f2286500481fdab4d3b3dc4f92a2006ffb8f02ca4c882bbd82db` | `Final_result/Origin_training/models/RML_origin/mme2e_tav_0.6724_0.6704_0.6792_0.6711_imginvl500_seed0.pt`, from `/Volumes/ADATA HD710 PRO/handover/` |
| `finetuned_model.pt` | 173.3 MB | `a4c1707f7bcc189d3103b42ea82c9d01f6ec2f992850e8cc1072f38f4dadd0de` | `Final_result/Origin_training/models/RML_finetune/mme2e_tav_0.7586_0.7611_0.7708_0.7627_imginvl500_seed0.pt`, from `/Volumes/ADATA HD710 PRO/handover/` |
| `RML_origin_SHAP_value.pkl` | 8.6 MB | `60fbd3c74eb592892bedf1ac989e8d8609afe98369cfc462d693a5c60d372644` | `Final_result/Origin_training/SHAP/SHAP_value/1225_RML_origin_mme2e_tav_0.6724_..._seed0_SHAP_value.pkl`, copied 2026-08-09 |
| `RML_finetune_SHAP_value.pkl` | 8.6 MB | `4211cf2e0eeb18e2af38a941f664d80dbf913d4e6861f7fc410383f820d7ef72` | `Final_result/Origin_training/SHAP/SHAP_value/1225_RML_finetune_mme2e_tav_0.7586_..._seed0_SHAP_value.pkl`, copied 2026-08-09 |

### DeepSHAP pickles

Each is `{'SHAP_value': [6 arrays of (144, 1152)], 'test_feature': tensor(144, 1152)}` over the RML test split, where the 1152 features are `[text_cls 1024 | v_cls 64 | a_cls 64]` (layout confirmed against `docs/provenance/upstream_SHAP_analysis.ipynb`). Columns 1088 onward are the same audio CLS tensor this experiment ablates, so Week 1 cross-checks them against the freshly extracted activations, see ADR 0005 Decision 2. They hold CUDA tensors and need a CPU-mapping unpickler to load on a CPU machine; the notebook includes one.

## Important: filename numbers are validation accuracy, not test accuracy

Both filenames encode the **validation**-split Accuracy/Recall/Precision/F1 at the epoch the checkpoint was saved (`Valid` row in the original training log), not test accuracy. This caused a false "wrong checkpoint" alarm during the 2026-08-09 diagnosis (see `docs/adr/0004-v3-restart-data-provenance-and-fidelity-gate.md`).

The actual **test** accuracy of these exact checkpoints, taken from the original training run logs (now copied in-repo as `docs/provenance/RML_{origin,finetune}_training_log.txt`), is:

| Checkpoint | Epoch saved | Test Accuracy | Matches paper Figure 4.3? |
| :--- | :--- | :--- | :--- |
| `base_model.pt` (RML_origin) | 29 | **76.39%** | Yes: base model |
| `finetuned_model.pt` (RML_finetune) | 15 | **79.86%** | Yes: fine-tuned model |

This is the fidelity target the v3 notebook's Day 0 gate asserts against, on the canonical 144-sample RML test split.

## Verification

```bash
shasum -a 256 checkpoints/base_model.pt checkpoints/finetuned_model.pt \
              checkpoints/RML_origin_SHAP_value.pkl checkpoints/RML_finetune_SHAP_value.pkl
```

## Subdirectory: `activations/`

Cached tensors extracted via forward hooks during Week 1 Days 1–2. Generated on-the-fly on Colab GPU and not distributed. Naming: `{model}_{split}_{acts|tlogits|vlogits|labels}.pt`, plus `{model}_{split}_ids.json` and `{model}_{split}_provenance.json`.

Each split carries a provenance stamp recording the two checkpoint SHAs, the meta path, and a hash of the split ID list. A resumed run whose stamp doesn't match, or a cache directory from an older notebook with no stamp at all, halts rather than reusing tensors from a different era. If that happens, delete `checkpoints/activations/` and re-extract; do not edit the stamp.
