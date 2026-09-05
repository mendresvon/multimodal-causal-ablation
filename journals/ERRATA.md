# Journal Errata

Corrections raised against journal entries after the fact, and their resolutions. Entries themselves are never silently edited; anything contested is recorded here instead.

---

## E1: 2026-08-11 session 1: the audio sample-rate route

**Status:** resolved. The original entry was accurate; the erratum against it is withdrawn.

### The claim under dispute

The session-1 entry reported that the Day 0 fidelity gate passed only after the audio branch was switched from `audio.wav` (44100 Hz) to `audio_16000.wav`, the 16 kHz clips the original training run read. The 16 kHz files were shipped as `data_bundle/RML_audio16k_all720.tar`, and a Google Drive upload stall was worked around by rebinding the loader inside the live runtime.

### Erratum, raised 2026-08-11 session 4: withdrawn

I could not find support for the 16 kHz claim anywhere else in the repository, and concluded the row was wrong:

- No committed revision of the notebook rebinds the audio loader. Checked at `57fa3b4`, `4f28f5b` and `1691356`; `git log -S` over the full history returned nothing.
- `src/datasets.py:390`, in the `IEMOCAP` class the notebook imports, loads `audio.wav`, and line 402 scales `win_length` by `sr/44100`. The `audio_16000.wav` variants at lines 391–392 and 403 are commented out.
- The two routes are genuinely different inputs, not a no-op. Measured headers: `audio.wav` in `RML_RAW_PROCESSED_Face.tar.gz` is 44100 Hz, 16-bit PCM, stereo; `audio_16000.wav` in `RML_audio16k_all720.tar` is 16000 Hz, 32-bit float, mono.
- Since the Day 0 gate halts on failure and sessions 2 and 3 ran Weeks 1–4 to completion, the 44100 Hz route must therefore clear 76.39% / 79.86% within the 1.0 pp tolerance.

On that basis I recommended treating the 16 kHz overlay tar as unused.

### Counter-erratum, 2026-08-12: the erratum was unsound

Both of the erratum's arguments fail.

The decisive test was `git log -S` over surviving history. But the four commits that carried the rebind, `d08a914`, `4b6e008`, `0ed79f3`, `1d5aa39`, had already been destroyed by the Google Drive sync failure and the LFS-cache clear I used to recover from it. `git cat-file -t` reports each as "Not a valid object name", and `git fsck` cannot read `1d5aa39`. **Pickaxe cannot see pruned objects, so it is not a sound negative proof on this repository.**

The second argument, that sessions 2 and 3 completing Weeks 1–4 proves the 44100 Hz route clears the gate, assumes what it needs to show. The gate passed under the rebind, in a live runtime. The surviving record does not establish which route was bound when those weeks executed.

The entry file has since been restored at [`2026-08-11-session1.md`](2026-08-11-session1.md), reconstructed from the session transcript. It records the 56.94% / 47.92% halt, the runtime rebind of `IEMOCAP.__getitem__` onto `audio_16000.wav` with `win_length` pinned to 400, and the passing gate at 05:57 UTC: base 76.39%, fine-tuned 79.86%, both exact. The rebind is preserved verbatim at [`../docs/recovered/2026-08-11-session1-audio-rebind.py`](../docs/recovered/2026-08-11-session1-audio-rebind.py).

### Where this landed

The committed notebook halts at 56.94% on the 44100 Hz route, and the 16 kHz route is the one measured to reproduce the training log. The provenance question was closed on 2026-08-12 by the Day 0 step 2 SHAP cross-check (r = 1.000000) rather than by artifact identity, see [`2026-08-12-session1.md`](2026-08-12-session1.md).

---

## E2: the freeze declaration and commit `c761017`

**Status:** resolved by moving the two documents into the successor repository.

`FROZEN.md` declares this repository read-only as of 2026-08-19, and the closing entry
[`2026-08-19-session1.md`](2026-08-19-session1.md) states there is no next runtime for it. Commit
`c761017`, dated 2026-09-05 and pushed to `origin/main`, nevertheless added two new documents:
`docs/future-work-causal-screening-spec.md` and
`docs/research-notes/neuron-causal-screening-methodology.md`.

Both are forward-looking. They propose Approach B, direct single-neuron causal ablation screening,
as a replacement for the L1 probe selection used in Section VI-D, and they argue it from the
Table X audit already in `results/`. Neither corrects the record here, and neither belongs to a
frozen archive.

They were copied on 2026-09-05 into the successor repository `../neuron-feature-mapping` as
`docs/plans/2026-09-05-causal-screening-approach-b-spec.md` and
`docs/research/2026-09-05-neuron-causal-screening-methodology.md`, each carrying a provenance
header naming this repository and commit `c761017`. Those copies are the live versions.

The originals are left in place rather than deleted. `c761017` is already in the pushed history,
so removing the files would make the history describe a state that never existed, which is the
failure mode the freeze exists to prevent. `FROZEN.md` now records the move in its asset table.
The freeze itself stands unchanged: this repository remains read-only, and the exception is
recorded here rather than treated as precedent.
