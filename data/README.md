# Dataset Provenance & Acquisition

The RML data used by this project is too large for Git and is stored on Google Drive / an external handover drive only. **Canonical source, per [ADR 0004](../docs/adr/0004-v3-restart-data-provenance-and-fidelity-gate.md):**

- **Source:** `BIG_DATA_RAW_PROCESSED_FACE.tar.gz` (~42 GB, all six datasets merged), at `/Volumes/ADATA HD710 PRO/handover/data/Processed data/`.
- **Label/text file:** `BIG_DATA_RAW_PROCESSED_FACE/meta_one_hot_label_six_categories.pkl`, one dict, keyed by utterance ID, covering all six datasets.
- **RML subset:** selected via `Model/Dig-Data_Model-Main/data/data_split/all_single_label_six_category/with_valid/Final_{train,valid,test}_split_six_categories_RML.txt` (518 / 58 / 144 samples).
- **Location expected by the v3 notebook:** `Model/Dig-Data_Model-Main/data/BIG_DATA_RAW_PROCESSED_FACE/` (containing the per-utterance folders + `meta_one_hot_label_six_categories.pkl`).

## Cleared at freeze, 2026-08-19

Both the local dataset archive and its extraction were deleted when the repository was closed,
to stop ~1.6 GB of regenerable bytes sitting on the Drive mount forever. Nothing unique was lost:

| Deleted | Was | Recover from |
|---|---|---|
| `data_bundle/RML_RAW_PROCESSED_Face.tar.gz` | 592,913,715 bytes | Two byte-identical copies on Drive under `handover/data/Processed data/`, one at the top of that folder and one inside `Database_processed_targz/` |
| `Model/Dig-Data_Model-Main/data/BIG_DATA_RAW_PROCESSED_FACE/` | 106,772 files, ~0.98 GB | Re-extract the archive above into that path, then follow Verification below |
| `checkpoints/RML_origin_SHAP_value.pkl` | 0 bytes: a failed Drive upload, never real data | `docs/provenance/upstream_SHAP_analysis.ipynb` regenerates it; its checksum in `checkpoints/README.md` was never met by this file |

Kept in place because they have no second copy: `data_bundle/RML_audio16k_all720.tar` (231 MB,
referenced in place by `neuron-feature-mapping`, see `FROZEN.md`), `checkpoints/base_model.pt`
and `checkpoints/finetuned_model.pt`.

## Verification

After extraction, confirm:
```
Model/Dig-Data_Model-Main/data/
└── BIG_DATA_RAW_PROCESSED_FACE/
    ├── meta_one_hot_label_six_categories.pkl
    └── <utterance_id>/            (one per sample, e.g. s4_f5hnh_di3/)
        ├── image_0.jpg ...
        └── audio.wav
```
The v3 notebook's Day 0 cell asserts that all 518+58+144 RML split IDs are keys in the meta file before proceeding, if that assertion fails, the extraction is incomplete or wrong.

## Do not use the archive's own meta.pkl

`RML_RAW_PROCESSED_Face.tar.gz` supplies the per-utterance `image_*.jpg` and
`audio.wav` files and nothing else. Its bundled `meta.pkl` is genuine but is not
usable here: its utterance IDs are a separate namespace (`s1_an1`) that does not
intersect the split files' IDs, and its labels are strings rather than one-hot
vectors. The notebook deletes it right after extraction. Labels and text always
come from `meta_one_hot_label_six_categories.pkl`. See ADR 0004 Decision 1.
