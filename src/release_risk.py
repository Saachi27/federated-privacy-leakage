"""Risk per release."""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics import _gaussian_window

PASS = 0.9


class SSIMSimilarity:
    """Valid-window SSIM of each query against each pool image, as src/metrics.ssim."""

    def __init__(self, window_size=11, sigma=1.5, data_range=1.0):
        self.window_size, self.sigma = window_size, sigma
        self.c1, self.c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
        self.data_range = data_range

    def _conv(self, x, window):
        return F.conv2d(x, window, groups=x.shape[1])

    def prepare(self, pool):
        pool = pool.float().clamp(0, self.data_range)
        window = _gaussian_window(self.window_size, self.sigma, pool.shape[1], pool.device, pool.dtype)
        mu = self._conv(pool, window)
        var = self._conv(pool * pool, window) - mu ** 2
        return {"pool": pool, "window": window, "mu": mu, "var": var}

    def score(self, queries, state):
        pool, window, mu_b, var_b = state["pool"], state["window"], state["mu"], state["var"]
        out = []
        for q in queries.float().clamp(0, self.data_range):
            a = q[None]
            mu_a = self._conv(a, window)
            var_a = self._conv(a * a, window) - mu_a ** 2
            cov = self._conv(a * pool, window) - mu_a * mu_b
            m = ((2 * mu_a * mu_b + self.c1) * (2 * cov + self.c2)) / \
                ((mu_a ** 2 + mu_b ** 2 + self.c1) * (var_a + var_b + self.c2))
            out.append(m.mean(dim=(1, 2, 3)))
        return torch.stack(out)

    def __call__(self, queries, pool):
        return self.score(queries, self.prepare(pool))


def neg_mse(queries, pool):
    """An alternative similarity: minus the mean squared error."""
    q = queries.float().flatten(1)
    p = pool.float().flatten(1)
    return -((q[:, None, :] - p[None, :, :]) ** 2).mean(-1)


def _prepare(similarity, pool):
    return similarity.prepare(pool) if hasattr(similarity, "prepare") else pool


def _score(similarity, queries, state):
    return similarity.score(queries, state) if hasattr(similarity, "score") else similarity(queries, state)


def ssim_many(query, pool):
    """Standard SSIM of one image (C,H,W) against each pool image (M,C,H,W)."""
    return SSIMSimilarity()(query[None], pool)[0]


class Gallery:
    """All candidate images, split by class, with per-class work cached."""

    def __init__(self, images, labels, ids=None, similarity=None):
        images = torch.as_tensor(images).float()
        labels = np.asarray(labels)
        ids = np.arange(len(labels)) if ids is None else np.asarray(ids)
        self.similarity = similarity if similarity is not None else SSIMSimilarity()
        self.pools, self.ids, self._state = {}, {}, {}
        for c in sorted(set(labels.tolist())):
            mask = labels == c
            self.pools[c] = images[torch.as_tensor(mask)]
            self.ids[c] = ids[mask]

    def sizes(self):
        return {c: len(v) for c, v in self.ids.items()}

    def state(self, c):
        if c not in self._state:
            self._state[c] = _prepare(self.similarity, self.pools[c])
        return self._state[c]

    def top1(self, queries, classes, exclude_ids=None):
        """Gallery id of the best match for each query within its class."""
        queries = torch.as_tensor(np.asarray(queries, dtype=np.float32)) if not torch.is_tensor(queries) \
            else queries.float()
        classes = [int(c) for c in classes]
        exclude_ids = [None] * len(classes) if exclude_ids is None else list(exclude_ids)
        assert len(exclude_ids) == len(classes)
        picks = [None] * len(classes)
        for c in sorted(set(classes)):
            rows = [i for i, ci in enumerate(classes) if ci == c]
            scores = _score(self.similarity, queries[rows], self.state(c)).double()
            ids = np.asarray(self.ids[c])
            for row, i in enumerate(rows):
                sc = scores[row].clone()
                if exclude_ids[i] is not None:
                    sc[torch.as_tensor(ids == int(exclude_ids[i]))] = -float("inf")
                best = sc.max()
                tied = np.flatnonzero((sc == best).numpy())
                picks[i] = int(ids[tied].min())
        return picks


def _tensor(x):
    return x.float() if torch.is_tensor(x) else torch.as_tensor(np.asarray(x, dtype=np.float32))


def match(recons, targets):
    """The study's SSIM-maximizing match (as scripts/attack_sum.py:match_and_order)."""
    from scipy.optimize import linear_sum_assignment
    recons = torch.nan_to_num(_tensor(recons), nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
    targets = _tensor(targets).clamp(0, 1)
    sim = SSIMSimilarity()(recons, targets).double().numpy()  # (recon i, target j)
    rows, cols = linear_sum_assignment(sim, maximize=True)
    by_target = {int(j): int(i) for i, j in zip(rows, cols)}
    order = [by_target[j] for j in range(len(targets))]
    return [float(sim[order[j], j]) for j in range(len(targets))], recons[order], order


def matched_ssims(recons, targets, matched=False):
    """SSIM of each target with its reconstruction (matched first unless matched=True)."""
    from src.metrics import ssim
    recons, targets = _tensor(recons), _tensor(targets)
    if matched:
        return [ssim(r.clamp(0, 1), t.clamp(0, 1)) for r, t in zip(recons, targets)], recons, \
            list(range(len(targets)))
    return match(recons, targets)


def chance_hit(release_labels, gallery_sizes):
    """Exact hit chance of one uniform pick per image from its class gallery."""
    labels = [int(c) for c in release_labels]
    miss = 1.0
    for c in labels:
        miss *= 1.0 - labels.count(c) / gallery_sizes[c]
    return 1.0 - miss


def any_passes(ssims, threshold=PASS):
    return int(any(s >= threshold for s in ssims))


def retrieval_hit(recons, recon_classes, release_ids, gallery, exclude_ids=None):
    """1 if any reconstruction's top-1 gallery pick is one of the release's images."""
    picks = gallery.top1(_tensor(recons), recon_classes, exclude_ids)
    members = {int(i) for i in release_ids}
    return int(any(p in members for p in picks)), picks


def score_release(targets, recons, labels, release_ids=None, gallery=None, recon_classes=None,
                  matched=False, threshold=PASS, guess_ids=None):
    """All three scores for one release."""
    ssims, ordered, order = matched_ssims(recons, targets, matched)
    out = {"mean_ssim": float(np.mean(ssims)), "ssims": [float(s) for s in ssims],
           "any_pass": any_passes(ssims, threshold)}
    if gallery is not None:
        if recon_classes is None:
            classes = [int(labels[j]) for j in range(len(order))]
        else:
            classes = [int(recon_classes[order[j]]) for j in range(len(order))]
        exclude = None if guess_ids is None else [guess_ids[order[j]] for j in range(len(order))]
        out["hit"], out["picks"] = retrieval_hit(ordered, classes, release_ids, gallery, exclude)
    return out
