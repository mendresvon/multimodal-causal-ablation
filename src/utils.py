"""
Deterministic seed utility for reproducible experiments across
ephemeral Google Colab GPU runtimes.

Call set_deterministic_seed() at the top of every Phase cell in
the experiment notebook to ensure identical probe rankings,
ablation sweeps, and retention ratio computations across sessions.
"""

import random

import numpy as np
import torch


def set_deterministic_seed(seed: int = 0) -> None:
    """Pin all sources of randomness for full reproducibility.

    Args:
        seed: Integer seed value. Default is 0, matching the
              upstream checkpoint naming convention (seed0).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
