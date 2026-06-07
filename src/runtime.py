"""Runtime helpers: optional imports, device selection, seeding, memory.

Isolating these lets the rest of the package import torch/transformers
lazily and degrade gracefully (e.g. on a CPU-only CI runner that has no
GPU and may not have the heavy libraries installed).
"""

import os
import gc
import random

import numpy as np

# --- Torch (optional) ---
try:
    import torch
    TORCH_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment dependent
    print("Torch not available:", exc)
    TORCH_AVAILABLE = False

# --- HuggingFace Transformers (optional) ---
try:
    from transformers import set_seed as hf_set_seed
    TRANSFORMERS_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment dependent
    print("Transformers not available:", exc)
    TRANSFORMERS_AVAILABLE = False


def get_device():
    """Return the active compute device.

    Returns
    -------
    device : torch.device or str
        CUDA device if a GPU is available, CPU device if torch is present
        without a GPU, or the string ``'cpu'`` if torch is unavailable.
    """
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.device("cuda")
    if TORCH_AVAILABLE:
        return torch.device("cpu")
    return "cpu"


def set_all_seeds(seed):
    """Seed all relevant RNGs for cross-library reproducibility.

    Seeds Python's :mod:`random`, NumPy, the ``PYTHONHASHSEED`` environment
    variable, PyTorch (CPU and CUDA), and HuggingFace, and enables
    deterministic cuDNN behaviour so multi-seed runs are comparable.

    Parameters
    ----------
    seed : int
        The random seed to apply.

    Returns
    -------
    None
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if TRANSFORMERS_AVAILABLE:
        hf_set_seed(seed)


def clear_memory():
    """Free Python and CUDA memory between experiment runs.

    Runs garbage collection and, if CUDA is available, empties the cache
    and collects inter-process CUDA memory. Important on Colab to avoid
    out-of-memory errors across many sequential runs.

    Returns
    -------
    None
    """
    gc.collect()
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


DEVICE = get_device()
