from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass

