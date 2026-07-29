# Dataset Provenance & Acquisition

The RML dataset files are too large for Git (~3.2 GB raw, ~593 MB preprocessed) and are stored on Google Drive only.

## Raw Data

- **Source:** [Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) via Kaggle API
- **Location:** `data/Raw data/RML/`
- **Structure:** Subject folders `s1` through `s8`, each containing `.avi` audio/video emotion recordings
- **Acquisition:**
  ```bash
  kaggle datasets download -d uwrfkaggler/ravdess-emotional-speech-audio
  ```

## Preprocessed Data

- **Source:** `RML_RAW_PROCESSED_Face.tar.gz` (593 MB), extracted from `/Volumes/ADATA HD710 PRO/handover/`
- **Location:** `Model/Dig-Data_Model-Main/data/RML_RAW_PROCESSED_Face/`
- **Contents:** Face-cropped, aligned frames ready for PyTorch DataLoader ingestion

## Verification

After acquisition or transfer, confirm the directory structure matches:
```
data/
├── Raw data/
│   └── RML/
│       ├── s1/
│       ├── s2/
│       ...
│       └── s8/
└── Processed data/

Model/Dig-Data_Model-Main/data/
└── RML_RAW_PROCESSED_Face/
```
