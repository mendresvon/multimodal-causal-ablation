# ADR 0003: Targeting CLS-Tokens for Ablation Instead of FFN Output

## Context
Prof KC Lan's causal ablation protocol explicitly instructs us to extract and ablate the **FFN output** of the dominant modality:
> *"Pick the layer to hook: the FFN output for modality M... Confirm the feature dimensionality of that layer (should be 64... FFN output dim may differ from the raw CLS dim)."*

However, upon inspecting the architecture of the `MME2E` model, we found that the Feed-Forward Network (FFN) represents the final classification head. Because the RML dataset has exactly 6 emotion classes, the FFN output projects the representation down to a tiny **6-dimensional vector** (the class logits). 

The 64-dimensional representation referenced in the paper (Section IV-C) is actually the post-Transformer **CLS-token representation** that feeds *into* this FFN classification head.

## Decision
We will systematically diverge from the literal wording of the protocol and target the **CLS-token representations** rather than the FFN output for all Day 1-2 extractions and Week 2 ablations. 

We will accomplish this by taking the `.npy` shortcut: leveraging the pre-cached 1152-d full-dataset feature representations (which are concatenations of the CLS tokens) and slicing out the Audio dimensions (`1088:1152`).

## Consequences
- **Mathematical Sanity:** We correctly target the 64-dimensional feature space, avoiding the mathematical impossibility of ranking "top neurons" on a 6-dimensional logits vector.
- **Speed:** By using the cached CLS representations (`.npy` files), we bypass the need to run heavy PyTorch forward passes on raw RML datasets, saving massive amounts of compute time.
- **Protocol Documentation:** We must explicitly note this architectural deviation in the final paper/write-up so the Professor understands why the CLS-tokens were targeted instead of the FFN outputs.
