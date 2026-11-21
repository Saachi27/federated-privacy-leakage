"""Checkpoint hashes for the journal runs."""

import hashlib

import numpy as np

TRAINED_SHA256 = {
    0: "fde91521e6e3d9a9db28aae6cb3b9eb0b9207e82118cccc4092faf4a96f694c7",
    1: "38d90b158a594f974d13a89e34cf9e6079538f12ea1ff2d03ac9118c9c495814",
    2: "4cd2a7c445c9f9125b91126fca0eddd0adde87cc9a0cd147fb86f5f2f6e62cff",
}

FRESH_STATE_SHA256 = {
    0: "42e4cc7a027b3e158ce01a217131f7293d0f0ae8947474821a457e17f87a0fac",
    1: "1c0fd59967787b7b0e83aa6d30dc3f6cab95cd0c028cbbb58bee70cb7eab90ee",
    2: "f663b4402199ae66fc5f1c80422b14f10178493b05f9093f61b5a695b0754579",
}


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def state_dict_sha256(model):
    h = hashlib.sha256()
    for name, t in model.state_dict().items():
        a = np.ascontiguousarray(t.detach().cpu().float().numpy())
        h.update(name.encode())
        h.update(str(tuple(a.shape)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def checkpoint_hash(model, k, model_path=None):
    """The hash of checkpoint k: the file's SHA-256 if it has a file, else the state dict's."""
    return file_sha256(model_path.format(seed=k)) if model_path else state_dict_sha256(model)


def verify(model, k, model_path=None):
    """Return the checkpoint hash; raise if it does not match the recorded one."""
    got = checkpoint_hash(model, k, model_path)
    want = (TRAINED_SHA256 if model_path else FRESH_STATE_SHA256).get(k)
    if want is None:
        raise RuntimeError(f"no recorded hash for checkpoint {k} ({'trained' if model_path else 'fresh'})")
    if got != want:
        raise RuntimeError(f"checkpoint {k} hash {got} does not match the recorded {want}; stopping")
    return got
