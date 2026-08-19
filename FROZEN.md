# This repo is frozen

Read-only as of 2026-08-19. Active work moved to the sibling repo:

**`../neuron-feature-mapping`**, https://github.com/mendresvon/neuron-feature-mapping

## What lives here

A finished experiment and its full record. Section VI-D ran to a definite answer on two
questions: a class's top audio neurons are causally load-bearing for that class (12 of 12
target intervals exclude zero, Tables VII/VIII), and those neurons do not stay load-bearing
through fine-tuning (0 of 6 classes clear a measured random-ablation null, `docs/adr/0006-*`).
The second answer is negative and is the result, not a gap, it is why the preservation
taxonomy was withdrawn rather than reported. Table IX was later scratched on review as an
editorial call about what belongs in the paper, not a challenge to the measurement.

Also here: the upstream thesis work and 26 journal entries in `journals/`. This is the history
of how the successor project got its starting assets. It is worth reading and not worth editing.

The closing journal entry is `journals/2026-08-19-session1.md`. It records the review outcome
that ended this line of work and what remained editorial.

## What moved

| Asset | Where it went |
|---|---|
| `Model/Dig-Data_Model-Main/` | `neuron-feature-mapping/model/` |
| `checkpoints/base_model.pt` | `neuron-feature-mapping/checkpoints/` (gitignored there too) |
| `checkpoints/activations/` | `neuron-feature-mapping/data/activations/` |
| `docs/neuron-feature-bucket-plan.md` | `neuron-feature-mapping/docs/`: this copy is now a pointer |
| `docs/facs-egemaps-experiment-spec.md` | Stays here as history. It was the first spec for the replacement direction and is superseded by `neuron-feature-mapping/docs/neuron-feature-bucket-plan.md` |
| `docs/section-vi-d-professor-briefing.md` | Stays here. Nothing in the successor repository restates it, and it is the fullest single account of Tables VII–X |
| Frozen results tables (Table VII, Table X, week1 probe AUC) | `neuron-feature-mapping/results/` |

Still referenced in place, not copied, to avoid a 231 MB Drive re-upload:
`data_bundle/RML_audio16k_all720.tar`.

## Why frozen rather than archived

The successor repo depends on nothing here at runtime, but the journals and ADRs are the
only record of several decisions the paper relies on. Deleting it would lose that. Editing
it would make the record unreliable. So: frozen.
