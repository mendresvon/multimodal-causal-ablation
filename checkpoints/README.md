# Checkpoint Provenance & Integrity

These files are too large for Git and are stored on Google Drive only. To verify integrity after syncing or transferring, compare against the SHA-256 checksums below.

## Files

| File | Size | SHA-256 | Source |
| :--- | :--- | :--- | :--- |
| `base_model.pt` | 173.3 MB | `565bc220d187f2286500481fdab4d3b3dc4f92a2006ffb8f02ca4c882bbd82db` | `Final_result/Origin_training/models/RML_origin/mme2e_tav_0.6724_0.6704_0.6792_0.6711_imginvl500_seed0.pt`, from `/Volumes/ADATA HD710 PRO/handover/` |
| `finetuned_model.pt` | 173.3 MB | `a4c1707f7bcc189d3103b42ea82c9d01f6ec2f992850e8cc1072f38f4dadd0de` | `Final_result/Origin_training/models/RML_finetune/mme2e_tav_0.7586_0.7611_0.7708_0.7627_imginvl500_seed0.pt`, from `/Volumes/ADATA HD710 PRO/handover/` |

## Important: filename numbers are validation accuracy, not test accuracy

Both filenames encode the **validation**-split Accuracy/Recall/Precision/F1 at the epoch the checkpoint was saved (`Valid` row in the original training log), not test accuracy. This caused a false "wrong checkpoint" alarm during the 2026-08-09 diagnosis (see `docs/adr/0004-v3-restart-data-provenance-and-fidelity-gate.md`).

The actual **test** accuracy of these exact checkpoints, taken from the original training run logs (`Final_result/Origin_training/wrong_stat/RML_{origin,finetune}/*.txt` on the handover drive), is:

| Checkpoint | Epoch saved | Test Accuracy | Matches paper Figure 4.3? |
| :--- | :--- | :--- | :--- |
| `base_model.pt` (RML_origin) | 29 | **76.39%** | Yes — base model |
| `finetuned_model.pt` (RML_finetune) | 15 | **79.86%** | Yes — fine-tuned model |

This is the fidelity target the v3 notebook's Day 0 gate asserts against, on the canonical 144-sample RML test split.

## Verification

```bash
shasum -a 256 checkpoints/base_model.pt checkpoints/finetuned_model.pt
```

## Subdirectory: `activations/`

Cached activation tensors extracted via forward hooks during Week 1 execution. These are generated on-the-fly on Colab GPU and are not distributed. They follow the naming convention `{phase}_{model}_{description}.pt`.
