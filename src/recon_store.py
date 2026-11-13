"""Save and load reconstructions next to their result records."""

import numpy as np
import torch


def to_array(recon):
    """A reconstruction tensor as a compact float16 array in [0, 1]."""
    recon = torch.nan_to_num(recon.detach().float(), nan=0.0, posinf=1.0, neginf=0.0)
    return recon.clamp(0, 1).cpu().numpy().astype(np.float16)


def pop_recons(records):
    """Move each record's "recon" array into a list and leave a "recon_id" behind."""
    recons = []
    for rec in records:
        recons.append(rec.pop("recon"))
        rec["recon_id"] = len(recons) - 1
    return recons


def save(path, recons):
    np.savez_compressed(path, **{f"r{i}": r for i, r in enumerate(recons)})


def load(path):
    data = np.load(path)
    return [data[f"r{i}"] for i in range(len(data.files))]
