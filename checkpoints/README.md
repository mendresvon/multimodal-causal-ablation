# Checkpoint Provenance & Integrity

These files are too large for Git and are stored on Google Drive only. To verify integrity after syncing or transferring, compare against the SHA-256 checksums below.

## Files

| File | Size | SHA-256 | Source |
| :--- | :--- | :--- | :--- |
| `base_model.pt` | 173.3 MB | `565bc220d187f2286500481fdab4d3b3dc4f92a2006ffb8f02ca4c882bbd82db` | Pre-trained RML base model (`RML_origin`), from `/Volumes/ADATA HD710 PRO/handover/` |
| `finetuned_model.pt` | 173.3 MB | `a4c1707f7bcc189d3103b42ea82c9d01f6ec2f992850e8cc1072f38f4dadd0de` | Fine-tuned RML model (`RML_finetune`), from `/Volumes/ADATA HD710 PRO/handover/` |
| `base_shap.pkl` | 8.6 MB | `60fbd3c74eb592892bedf1ac989e8d8609afe98369cfc462d693a5c60d372644` | Pre-computed DeepSHAP attributions for base model |
| `finetuned_shap.pkl` | 8.6 MB | `4211cf2e0eeb18e2af38a941f664d80dbf913d4e6861f7fc410383f820d7ef72` | Pre-computed DeepSHAP attributions for fine-tuned model |

## Verification

```bash
shasum -a 256 checkpoints/base_model.pt checkpoints/finetuned_model.pt checkpoints/base_shap.pkl checkpoints/finetuned_shap.pkl
```

## Subdirectory: `activations/`

Cached activation tensors extracted via forward hooks during Phase B execution. These are generated on-the-fly on Colab GPU and are not distributed. They follow the naming convention `{phase}_{model}_{description}.pt`.
