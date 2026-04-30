import os
import tempfile

import torch


def atomic_torch_save(state, file_path):
    """Save a checkpoint via a temporary file and atomically replace the target."""
    directory = os.path.dirname(file_path) or '.'
    os.makedirs(directory, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        dir=directory,
        prefix=os.path.basename(file_path) + '.',
        suffix='.tmp',
    )
    os.close(fd)

    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, file_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
