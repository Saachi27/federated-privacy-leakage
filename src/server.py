"""The federated server: orchestrates rounds and performs FedAvg aggregation."""

import copy

from .defenses import apply_mask, pairwise_masks


def average_states(states, num_samples):
    """Sample-weighted average of a list of model `state_dict`s (FedAvg)."""
    total = float(sum(num_samples))
    weights = [n / total for n in num_samples]

    avg = copy.deepcopy(states[0])
    for key in avg:
        if avg[key].dtype.is_floating_point:
            acc = states[0][key].double() * weights[0]
            for state, w in zip(states[1:], weights[1:]):
                acc += state[key].double() * w
            avg[key] = acc.to(states[0][key].dtype)
        else:
            avg[key] = states[0][key].clone()
    return avg


def secure_average_states(states, num_samples, base_seed, scale=1.0):
    """FedAvg where the server only ever sums pairwise-masked uploads."""
    total = float(sum(num_samples))
    keys = [k for k in states[0] if states[0][k].dtype.is_floating_point]
    masks = pairwise_masks([states[0][k].double() for k in keys], len(states),
                           base_seed=base_seed, scale=scale)
    uploads = [
        apply_mask([state[k].double() * (n / total) for k in keys], mask)
        for state, n, mask in zip(states, num_samples, masks)
    ]
    avg = copy.deepcopy(states[0])
    for layer, key in enumerate(keys):
        avg[key] = sum(u[layer] for u in uploads).to(states[0][key].dtype)
    return avg, uploads


def select_clients(num_clients, fraction, rng):
    """Pick a random subset of client indices for a round (FedAvg's C)."""
    k = max(1, int(round(fraction * num_clients)))
    return sorted(rng.sample(range(num_clients), k))
