# Spec: Neuron-Physiology Alignment Experiment (FACS / eGeMAPS pivot)

Status: planning only. No code written yet. This spec is meant to seed a new,
separate repo outside the Google-Drive-mounted working directory.

## 1. Background

The professor reviewed the VI-D causal-ablation work in
`docs/section-vi-d-professor-briefing.md` and rejected Table IX (cross-model
coordinate transfer/alignment — already a documented negative result, see
project memory `v3-week3-retention-open-confound.md`). He said the transfer
claim doesn't add meaning to the paper and assigned a replacement direction
instead of asking for a fix.

The replacement direction: test whether the already-found, causally-validated
class-selective neuron sets from Tables VII/VIII (12/12 positive target
effects, 7/12 pass concentration) actually correspond to known,
physiologically-grounded emotion markers, rather than being an opaque
coordinate set that merely moves a softmax. Two candidate reference schemes:

- **Video:** Ekman's FACS (Facial Action Coding System) Action Units — a
  muscle-movement-level coding scheme, later mapped to typical AU
  combinations per basic emotion (e.g. anger = AU4+AU5+AU7+AU23, happiness =
  AU6+AU12 "Duchenne smile").
- **Audio:** eGeMAPS (extended Geneva Minimalistic Acoustic Parameter Set,
  Eyben/Scherer) — the closest acoustic analog to FACS, standardized via
  openSMILE. Covers F0 statistics, loudness/energy, jitter, shimmer, HNR,
  formants F1–F3, speech rate.

This also functions as the answer to "does the existing VI-D work still have
value" — it converts Tables VII/VIII from abstract coordinates into
coordinates that track known emotion physiology, without needing the killed
transfer claim.

## 2. Scope: start with one feature, one neuron set, one dataset pair

Full scope (all AUs x all eGeMAPS params x all 6 classes x both modalities x
7 datasets) is over 1,000 correlation tests and not where this starts.
First slice, chosen for lowest risk and reuse of already-validated work:

- **Feature:** F0 (fundamental frequency / pitch) — anger is marked by pitch
  going up and becoming more variable, a well-known, simple acoustic marker.
- **Modality:** audio only (eGeMAPS), not video/FACS yet — see Section 5 for
  why video is deferred.
- **Neurons:** the already-found, already causally-validated anger
  coordinates for the base model — indices 20, 3, 39, 27, 53 (from Table VII,
  `docs/section-vi-d-professor-briefing.md`). No new probe-fitting needed for
  this slice — reuse the existing set and just correlate.
- **Datasets:** RML first (proves the pipeline on data the checkpoint was
  actually trained/tested on), then IEMOCAP as the generalization check.

This is deliberately a correlation check against a known set, not a full
neuron-to-bucket mapping sweep. The full 64 x n_features matrix (which
resolves both "which neuron owns this bucket" and "does the known set
correlate" as the same computation — the mapping is the argmax per feature
column, the validation is the class-relevant submatrix) is the natural
next step after this slice works, not the starting point.

## 3. Why RML then IEMOCAP, not RML alone

The professor was explicit: don't limit this to RML. The project already has
processed data for 7 datasets backed up on Google Drive (folder
`Database_processed_targz`, all in the same `*_RAW_PROCESSED_Face.tar.gz`
pipeline format as RML):

| Dataset | Size |
|---|---|
| RML | 593 MB |
| eNTERFACE'05 | 689 MB |
| RAVDESS | 1.3 GB |
| IEMOCAP | 6.9 GB |
| MELD | 7.0 GB |
| MOSEI | 7.2 GB |
| LIRIS | 13.5 GB |

IEMOCAP was picked as the second dataset because the repo already has
everything needed except a checkpoint:

- Split files already exist:
  `Model/Dig-Data_Model-Main/data/data_split/all_single_label_six_category/with_valid/`
  has train/valid/test/10p/90p variants for IEMOCAP (and MELD, RAVDESS,
  LIRIS, eNTERFACE'05).
- Dataset loader code already exists: `get_dataset_iemocap()`
  (`Model/Dig-Data_Model-Main/src/datasets.py:26`) is dataset-agnostic
  despite the name — per ADR 0004, it's the same function that trained and
  evaluated every dataset in the original thesis, including RML. Switching
  dataset is a matter of pointing at a different split-file, not writing new
  dataloader code.
- A preprocessing notebook exists: `preprocess_iemocap.ipynb`.

So extending to IEMOCAP is wiring, not greenfield.

## 4. Open blocker: checkpoint provenance for IEMOCAP

Only `checkpoints/base_model.pt` and `checkpoints/finetuned_model.pt` exist
locally, both RML-related (`finetuned_model.pt` = base pretrained on the
other five datasets, then fine-tuned on RML, per
`docs/section-vi-d-professor-briefing.md:31`).

**Must confirm before running the IEMOCAP step:** did `base_model.pt`'s
pretraining corpus include IEMOCAP? `main.py` has a commented-out load
referencing `pretrain_no_mosei_no_IEMOCAP_LIRIS`, implying an
IEMOCAP-excluded pretrain variant exists somewhere (not in local
`checkpoints/`). If `base_model.pt` was pretrained including IEMOCAP,
testing on IEMOCAP is train-data leakage, not a clean generalization check.
This needs to be resolved (find the IEMOCAP-excluded checkpoint, or train
one, or explicitly caveat the result as train-set correlation rather than
generalization) before the IEMOCAP correlation number is presented as
evidence.

## 5. Why video/FACS is deferred, not dropped

The professor's own example (lips, presumably AU6/AU12) was about video. But
video is the weakest branch in the model by every attribution measure
already on record: aggregate SHAP 0.010561 (base) / 0.006042 (fine-tuned)
vs. audio's 0.084/0.115; per-coordinate SHAP 0.000165/0.000094 vs audio's
0.001316/0.001797. ADR 0005 calls video "a distant third."

Before installing any AU-detection tool (OpenFace or py-feat), the video CLS
signal itself needs checking: hook `v_transformer` for its 64-d CLS (no hook
exists yet — only `a_transformer`/`a_out` and `v_out` are hooked in the
current notebook, and `v_out` is only 6 logits, not enough to recover the
64-d CLS), then rerun the existing L1-probe + AUC pipeline on it. If
per-class AUC sits near chance (audio's was 0.884/0.910 mean), the FACS arm
is dead on arrival before any AU tool is built — a reportable finding in its
own right, but it changes what's worth building. This check is why video is
sequenced after the audio slice proves the method, not before.

## 6. Reused assets vs. new work

**Reused as-is (from this repo):**
- Anger's causally-validated top-5 coordinates (base model): 20, 3, 39, 27,
  53.
- `checkpoints/base_model.pt` for RML inference.
- IEMOCAP split files and dataset loader code.
- The probe/ablation/bootstrap methodology pattern from VI-D, as a template
  for how to report the correlation result (effect size + CI, not a bare
  p-value).

**New work (belongs in the new repo):**
- openSMILE eGeMAPS extraction pipeline, run on `audio_16000.wav` — the
  16kHz route the checkpoints were actually trained on, not the 44.1kHz
  files (see project memory `v3-audio-route-is-44100-not-16k`).
- Forward hook on `v_transformer` for the 64-d video CLS (needed later for
  Section 5, not for the first slice).
- Correlation/statistical test between the 5 neuron activations and F0
  stats across the RML and IEMOCAP anger test clips.
- Resolution of the checkpoint-provenance blocker in Section 4.

## 7. Alignment hazards to carry over (from prior work on this model)

- openSMILE must run on `audio_16000.wav`, not the 44.1kHz audio files —
  the checkpoints were trained on the 16kHz route.
- Any frame- or sample-level extraction must exactly match the ordering and
  file-list the model's own dataloader used (including `.DS_Store`
  exclusion) — extracting over "everything in the folder" silently
  misaligns activations against features.
- Build a scratch Python environment outside any Google-Drive-mounted
  directory. Drive-mounted venvs are known to hang the shell.
- If/when the full 64 x n_features matrix is eventually built (not this
  slice), multiplicity correction (Benjamini–Hochberg FDR) must be
  preregistered up front — the VI-D briefing already lists uncorrected
  multiplicity as limitation #1 for Tables VII/VIII, and repeating that in
  a second experiment is not survivable.

## 8. Repo setup (not done yet)

This experiment will live in a new repo, separate from
`multimodal-causal-ablation`, to avoid known Google-Drive-mount hazards
(hanging venvs, hanging `git commit`). Location/creation deferred — user
wants the spec finished first.

## 9. Next steps, in order

1. Resolve the checkpoint-provenance question in Section 4 (repo-side
   check, no new repo needed yet).
2. Create the new repo (local git init or GitHub — decision deferred).
3. Build the openSMILE eGeMAPS extraction pipeline, run on RML
   `audio_16000.wav` anger test clips.
4. Correlate against the 5 known base-model anger neurons; report effect
   size + CI, no multiplicity correction needed for a single feature.
5. Repeat step 3–4 on IEMOCAP anger clips using `base_model.pt` (pending
   Section 4 resolution) or the correct checkpoint.
6. Only after 3–5 land: revisit Section 5 (video CLS AUC check) to decide
   whether the FACS/video arm is worth building.
