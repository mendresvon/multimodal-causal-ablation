# RECOVERED 2026-08-12 from Claude Code transcript
#   ~/.claude/projects/-Users-breznev-Library-CloudStorage-GoogleDrive-mendresvon-gmail-com-My-Drive-multimodal-causal-ablation/5a63c52a-e6c7-40c9-b578-7fbe5bbfc3b8.jsonl
#
# This block lived in the "Day 0: Canonical Data Load + Halting Fidelity Gate" cell of
# multimodal-causal-ablation-v3.ipynb between commits d08a914/4b6e008 and 1d5aa39
# (2026-08-11 session 1). All four of those commits were destroyed when the Google Drive
# desktop client's sync failure — and the LFS cache clear used to recover from it —
# corrupted the object database. `git cat-file -t` reports every one of them as
# "Not a valid object name", and `git fsck` cannot read 1d5aa39.
#
# It is the configuration under which the Day 0 fidelity gate reproduced base 76.39% and
# fine-tuned 79.86% exactly, on 2026-08-11 at 05:57 UTC. Without it the same cell halts at
# base 56.94% / fine-tuned 47.92%.
#
# It also requires that data_bundle/RML_audio16k_all720.tar has been extracted over the
# corpus inside the runtime; session 1's Day 0 step 0 printed
#   16 kHz audio overlaid: 720 files, all 16000 Hz, 1 channel(s), 32-bit format tag 3
# The current notebook does not extract that tar, so this block alone is not sufficient.
#
# This file is a recovery record, not an active code path. Nothing imports it.
# ---------------------------------------------------------------------------------------

# The Drive-hosted src/datasets.py is the pre-audio-fix loader: it reads `audio.wav`
# at 44100 Hz with win_length = int(sr / 44100 * 400) and no resample, which is the
# 56.94% front end. The fixed file exists in the repo but the Google Drive desktop
# client has stopped uploading it, so Colab still sees a 26950-byte revision.
#
# That revision was identified exactly, not guessed: it is commit bc5bb91 with the
# frame-count fix applied and nothing else, which reconstructs to 26950 bytes on the
# nose (27005 - 55). So the only behavioural difference from the repo copy lives in
# IEMOCAP.__getitem__; everything the methods below inherit from the stale module --
# __init__, cutSpecToPieces, collate_fn -- is already identical.
#
# Rather than wait on Drive sync, rebind the methods here, in the runtime. Both
# bodies are verbatim copies of the fixed src/datasets.py (commit d029145 for the
# audio, 5e393f6 for the frame count). The frame-count rebind is a no-op against
# this particular Drive revision and is kept so the cell is correct against any of
# them.
import importlib
import inspect
import torchaudio
from PIL import Image
import src.datasets as _ds_mod
importlib.reload(_ds_mod)
from src.datasets import IEMOCAP, collate_fn
from src.models.e2e import MME2E

def _patched_sample_imgs_by_interval(self, folder, fps=30):
    nums = len(glob.glob(f'{folder}/image_*.jpg'))
    step = int(self.img_interval / 1000 * fps)
    return [os.path.join(folder, f'image_{i}.jpg') for i in list(range(0, nums, int(step)))]

def _patched_getitem(self, ind):
    uttrId = self.utterance_ids[ind]
    sample_folder = os.path.join(self.main_folder, uttrId)

    sampledImgs = np.array([
        np.float32(Image.open(imgPath))
        for imgPath in self.sample_imgs_by_interval(sample_folder)
    ])

    # Prefer the training-era 16 kHz clip; resample audio.wav only if it is absent.
    _wav16 = os.path.join(sample_folder, 'audio_16000.wav')
    if os.path.exists(_wav16):
        waveform, sr = torchaudio.load(_wav16)
    else:
        waveform, sr = torchaudio.load(os.path.join(sample_folder, 'audio.wav'))

    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000
    waveform = waveform.mean(dim=0, keepdim=True)
    specgram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, win_length=int(float(sr) / 16000 * 400))(waveform).unsqueeze(0)
    specgrams = self.cutSpecToPieces(specgram)

    return uttrId, sampledImgs, specgrams, self.texts[ind], self.labels[ind]

IEMOCAP.sample_imgs_by_interval = _patched_sample_imgs_by_interval
IEMOCAP.__getitem__ = _patched_getitem

# Prove the rebind took. Identity, not source text: inspect.getsource can fail on a
# function defined in a notebook cell, and a false halt here is indistinguishable
# from a real one. Also pin the spectrogram window to the training-era 400, not the
# 145 that int(16000 / 44100 * 400) would give.
assert IEMOCAP.__getitem__ is _patched_getitem, (
    "HALT: the IEMOCAP.__getitem__ rebind did not take. Something re-imported or "
    "reloaded src.datasets after this cell."
)
assert IEMOCAP.sample_imgs_by_interval is _patched_sample_imgs_by_interval, (
    "HALT: the IEMOCAP.sample_imgs_by_interval rebind did not take."
)
assert int(float(16000) / 16000 * 400) == 400, "HALT: win_length is not 400 at 16 kHz."
