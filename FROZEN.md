# This repo is frozen

Read-only as of 2026-08-19. Active work moved to the sibling repo:

**`../neuron-feature-mapping`** — https://github.com/mendresvon/neuron-feature-mapping

## What lives here

The upstream thesis work, the causal-ablation results behind Tables VII/VIII, the scratched
Table IX cross-model transfer result, and 26 journal entries in `journals/`. This is the
history of how the successor project got its starting assets. It is worth reading and not
worth editing.

The closing journal entry is `journals/2026-08-19-session1.md`. It records the review outcome
that ended this line of work and what was left unfinished.

## What moved

| Asset | Where it went |
|---|---|
| `Model/Dig-Data_Model-Main/` | `neuron-feature-mapping/model/` |
| `checkpoints/base_model.pt` | `neuron-feature-mapping/checkpoints/` (gitignored there too) |
| `checkpoints/activations/` | `neuron-feature-mapping/data/activations/` |
| `docs/neuron-feature-bucket-plan.md` | `neuron-feature-mapping/docs/` — this copy is now a pointer |
| `docs/facs-egemaps-experiment-spec.md` | Stays here as history. It was the first spec for the replacement direction and is superseded by `neuron-feature-mapping/docs/neuron-feature-bucket-plan.md` |
| `docs/section-vi-d-professor-briefing.md` | Stays here. Nothing in the successor repository restates it, and it is the fullest single account of Tables VII–X |
| Frozen results tables (Table VII, Table X, week1 probe AUC) | `neuron-feature-mapping/results/` |

Still referenced in place, not copied, to avoid a 231 MB Drive re-upload:
`data_bundle/RML_audio16k_all720.tar`.

## Why frozen rather than archived

The successor repo depends on nothing here at runtime, but the journals and ADRs are the
only record of several decisions the paper relies on. Deleting it would lose that. Editing
it would make the record unreliable. So: frozen.
