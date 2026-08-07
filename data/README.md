# Dataset Provenance & Acquisition

The RML dataset files are too large for Git (~3.2 GB raw, ~593 MB preprocessed) and are stored on Google Drive or external media only.

## Raw Data (Optional / Pruned Locally)

- **Source:** [Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) via Kaggle API / External Handover Drive
- **Status:** Pruned from local workspace to save ~3.2 GB disk space. Raw `.avi` files are not required for notebook execution.
- **Acquisition (If needed):**
  ```bash
  kaggle datasets download -d uwrfkaggler/ravdess-emotional-speech-audio
  ```

## Preprocessed Data (Required for Pipeline)

- **Source:** `RML_RAW_PROCESSED_Face.tar.gz` (593 MB), extracted from `/Volumes/ADATA HD710 PRO/handover/`
- **Location:** `Model/Dig-Data_Model-Main/data/RML_RAW_PROCESSED_Face/`
- **Contents:** Face-cropped, aligned frames and extracted audio ready for PyTorch DataLoader ingestion.

## Verification

After acquisition or transfer, confirm the active directory structure matches:
```
data/
└── Processed data/
    └── Database_processed_targz/

Model/Dig-Data_Model-Main/data/
└── RML_RAW_PROCESSED_Face/
```

