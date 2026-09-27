"""Exclusive writes for new derived archives; preserve existing research artifacts."""
from pathlib import Path
import numpy as np


def save_new_archive(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
