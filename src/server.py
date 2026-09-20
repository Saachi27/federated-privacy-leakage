"""The federated server: orchestrates rounds and performs FedAvg aggregation."""

import copy


def average_states(states, num_samples):
    """Sample-weighted average of a list of model `state_dict`s (FedAvg)."""
    total = float(sum(num_samples))
    weights = [n / total for n in num_samples]

    avg = copy.deepcopy(states[0])
    for key in avg:
        if avg[key].dtype.is_floating_point:
            acc = states[0][key].clone().float() * weights[0]
            for state, w in zip(states[1:], weights[1:]):
                acc += state[key].float() * w
            avg[key] = acc.to(states[0][key].dtype)
        else:
            avg[key] = states[0][key].clone()
    return avg


def select_clients(num_clients, fraction, rng):
    """Pick a random subset of client indices for a round (FedAvg's C)."""
    k = max(1, int(round(fraction * num_clients)))
    return sorted(rng.sample(range(num_clients), k))
