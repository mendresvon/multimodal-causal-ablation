# Dataset Provenance & Acquisition

The RML data used by this project is too large for Git and is stored on Google Drive / an external handover drive only. **Canonical source, per [ADR 0004](../docs/adr/0004-v3-restart-data-provenance-and-fidelity-gate.md):**

- **Source:** `BIG_DATA_RAW_PROCESSED_FACE.tar.gz` (~42 GB, all six datasets merged), at `/Volumes/ADATA HD710 PRO/handover/data/Processed data/`.
- **Label/text file:** `BIG_DATA_RAW_PROCESSED_FACE/meta_one_hot_label_six_categories.pkl` — one dict, keyed by utterance ID, covering all six datasets.
- **RML subset:** selected via `Model/Dig-Data_Model-Main/data/data_split/all_single_label_six_category/with_valid/Final_{train,valid,test}_split_six_categories_RML.txt` (518 / 58 / 144 samples).
- **Location expected by the v3 notebook:** `Model/Dig-Data_Model-Main/data/BIG_DATA_RAW_PROCESSED_FACE/` (containing the per-utterance folders + `meta_one_hot_label_six_categories.pkl`).

## Do not use

`RML_RAW_PROCESSED_Face.tar.gz` / the `RML_RAW_PROCESSED_Face/` folder is **retired**. Its `meta.pkl` was fabricated by an earlier agent session and does not exist in that archive's actual contents — evaluating the correct model checkpoints against it produced the garbage results diagnosed in `journals/2026-08-09.md`.

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
The v3 notebook's Day 0 cell asserts that all 518+58+144 RML split IDs are keys in the meta file before proceeding — if that assertion fails, the extraction is incomplete or wrong.
