"""Formal (epsilon, delta) differential-privacy accounting."""

from opacus.accountants import RDPAccountant


def compute_epsilon(noise_multiplier, sample_rate, steps, delta=1e-5):
    """Total epsilon for `steps` DP-SGD steps at a fixed noise + sampling rate."""
    if steps == 0 or noise_multiplier <= 0 or sample_rate <= 0:
        return float("inf")
    accountant = RDPAccountant()
    # History is a list of (noise_multiplier, sample_rate, num_steps) blocks.
    accountant.history = [(float(noise_multiplier), float(sample_rate), int(steps))]
    return float(accountant.get_epsilon(delta=delta))


def summarize(noise_multiplier, sample_rate, steps, delta=1e-5):
    """Convenience: return a dict describing the spent budget."""
    eps = compute_epsilon(noise_multiplier, sample_rate, steps, delta)
    return {
        "epsilon": eps, "delta": delta,
        "noise_multiplier": noise_multiplier,
        "sample_rate": sample_rate, "steps": steps,
    }
