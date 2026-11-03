"""Turn the raw runs in results/runs/ into the report's tables and figures."""

import glob
import json
import os
import re
import sys

import numpy as np
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.plotting import MARKERS, SERIES, plt, style

RUNS = "results/runs"

# (run-file prefix, label in tables and figures)
TRAINING = [
    ("mnist_baseline", "FedAvg, no protection"),
    ("mnist_mask", "FedAvg + pairwise masking"),
    ("mnist_dp0.5", "FedAvg + DP-SGD, σ=0.5"),
    ("mnist_dp1", "FedAvg + DP-SGD, σ=1"),
    ("mnist_dp2", "FedAvg + DP-SGD, σ=2"),
    ("mnist_alpha0.5", "Non-IID, Dirichlet α=0.5"),
    ("mnist_alpha0.1", "Non-IID, Dirichlet α=0.1"),
    ("cifar10_baseline", "CIFAR-10, no protection"),
]


def load_training(prefix):
    runs = []
    for path in sorted(glob.glob(f"{RUNS}/train_{prefix}_s*.json")):
        with open(path) as f:
            runs.append(json.load(f))
    return runs


def curves(runs):
    return np.array([[h["test_acc"] for h in r["history"]] for r in runs])


def training_table(all_runs):
    lines = ["| Setting | Rounds | Seeds | Final test accuracy | ε (δ=1e-5) |", "|---|---|---|---|---|"]
    for prefix, label in TRAINING:
        runs = all_runs[prefix]
        if not runs:
            continue
        final = curves(runs)[:, -1]
        eps = "—"
        if runs[0].get("privacy"):
            eps = f"{np.mean([r['privacy']['epsilon'] for r in runs]):.2f}"
        lines.append(f"| {label} | {len(runs[0]['history'])} | {len(runs)} "
                     f"| {final.mean():.4f} ± {final.std():.4f} | {eps} |")
    return lines


def plot_curves(all_runs, series, title, path):
    series = [(p, l) for p, l in series if all_runs[p]]
    if not series:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    for i, (prefix, label) in enumerate(series):
        acc = curves(all_runs[prefix])
        rounds = np.arange(1, acc.shape[1] + 1)
        ax.fill_between(rounds, acc.mean(0) - acc.std(0), acc.mean(0) + acc.std(0),
                        color=SERIES[i], alpha=0.12, linewidth=0)
        ax.plot(rounds, acc.mean(0), marker=MARKERS[i], markevery=3, color=SERIES[i], label=label)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("communication round")
    ax.set_ylabel("test accuracy")
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right")
    fig.savefig(path)
    plt.close(fig)


def plot_privacy_accuracy(all_runs, path):
    points = []
    for prefix in ("mnist_dp2", "mnist_dp1", "mnist_dp0.5"):
        runs = all_runs[prefix]
        if runs:
            points.append((np.mean([r["privacy"]["epsilon"] for r in runs]),
                           curves(runs)[:, -1].mean(), curves(runs)[:, -1].std(),
                           runs[0]["privacy"]["noise_multiplier"]))
    if len(points) < 2:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    eps, acc, std, sigmas = zip(*points)
    ax.errorbar(eps, acc, yerr=std, marker="o", capsize=3, color=SERIES[0])
    for e, a, s in zip(eps, acc, sigmas):
        ax.annotate(f"σ={s:g}", (e, a), textcoords="offset points", xytext=(0, -15),
                    ha="center", fontsize=9, color="#52514e")
    if all_runs["mnist_baseline"]:
        base = curves(all_runs["mnist_baseline"])[:, -1].mean()
        ax.axhline(base, color="#52514e", linewidth=1, linestyle="--")
        ax.annotate(f"no DP: {base:.3f}", (max(eps), base), textcoords="offset points",
                    xytext=(0, -13), ha="right", fontsize=9, color="#52514e")
    ax.set_xlabel("privacy budget ε per client (δ=1e-5; lower = more private)")
    ax.set_ylabel("final test accuracy")
    ax.set_title("The cost of differential privacy", loc="left")
    fig.savefig(path)
    plt.close(fig)


def load_attacks():
    """Return {model: {(round, attacker): [per-seed mean SSIM, success, ...]}}."""
    out = {}
    for path in sorted(glob.glob(f"{RUNS}/attack_*_s*.json")):
        with open(path) as f:
            d = json.load(f)
        out.setdefault(d["args"]["model"], []).append(d)
    return out


def attack_table(attacks):
    lines = ["| Model | Round | Test accuracy | Attacker | SSIM (mean±std) | Attack success |",
             "|---|---|---|---|---|---|"]
    for model, runs in attacks.items():
        rounds = sorted({r["round"] for d in runs for r in d["attacks"]})
        attackers = list(dict.fromkeys(r["attacker"] for d in runs for r in d["attacks"]))
        for rnd in rounds:
            accs = [d["history"][rnd - 1]["test_acc"] for d in runs] if rnd else []
            acc = f"{np.mean(accs):.3f}" if accs else "—"
            for attacker in attackers:
                recs = [rec for d in runs for r in d["attacks"]
                        if r["round"] == rnd and r["attacker"] == attacker for rec in r["records"]]
                ssim = np.array([r["ssim"] for r in recs])
                lines.append(f"| {model} | {rnd} | {acc} | {attacker} "
                             f"| {ssim.mean():.3f} ± {ssim.std():.3f} "
                             f"| {(ssim >= 0.9).mean() * 100:.0f}% |")
    return lines


def plot_attacks(attacks, path):
    if not attacks:
        return
    fig, axes = plt.subplots(1, len(attacks), figsize=(4 * len(attacks), 3.4),
                             sharey=True, squeeze=False)
    names = {"lenet": "Sigmoid LeNet", "smallcnn": "SmallCNN (ReLU, max-pool)"}
    for ax, (model, runs) in zip(axes[0], attacks.items()):
        rounds = sorted({r["round"] for d in runs for r in d["attacks"]})
        attackers = list(dict.fromkeys(r["attacker"] for d in runs for r in d["attacks"]))
        x = np.arange(len(rounds))
        for i, attacker in enumerate(attackers):
            per_round = [[rec["ssim"] for d in runs for r in d["attacks"]
                          if r["round"] == rnd and r["attacker"] == attacker
                          for rec in r["records"]] for rnd in rounds]
            mean = np.array([np.mean(v) for v in per_round])
            std = np.array([np.std(v) for v in per_round])
            ax.fill_between(x, mean - std, np.minimum(mean + std, 1), color=SERIES[i], alpha=0.12, linewidth=0)
            ax.plot(x, mean, marker=MARKERS[i], color=SERIES[i], label=attacker)
        ax.set_xticks(x, rounds)
        ax.set_xlabel("federated round the model was attacked at")
        ax.set_title(names.get(model, model), loc="left")
    axes[0][0].set_ylabel("reconstruction SSIM")
    axes[0][-1].legend(title="attacker", loc="center right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def mask_check(all_runs):
    """Measured masking cost: per-seed accuracy difference from the baseline."""
    base = {r["config"]["seed"]: r for r in all_runs["mnist_baseline"]}
    diffs = []
    for r in all_runs["mnist_mask"]:
        if r["config"]["seed"] in base:
            diffs.append(r["history"][-1]["test_acc"] - base[r["config"]["seed"]]["history"][-1]["test_acc"])
    return diffs


def main():
    style()
    all_runs = {prefix: load_training(prefix) for prefix, _ in TRAINING}
    attacks = load_attacks()
    labels = dict(TRAINING)

    plot_curves(all_runs, [(p, labels[p]) for p in ("mnist_baseline", "mnist_mask", "mnist_dp1")],
                "Training with each protection (MNIST, IID)", "results/fig_accuracy_curves.png")
    plot_curves(all_runs, [(p, labels[p]) for p in ("mnist_baseline", "mnist_alpha0.5", "mnist_alpha0.1")],
                "Effect of non-IID client data (MNIST)", "results/fig_noniid.png")
    plot_curves(all_runs, [("cifar10_baseline", labels["cifar10_baseline"])],
                "FedAvg on CIFAR-10", "results/fig_cifar10.png")
    plot_privacy_accuracy(all_runs, "results/fig_privacy_accuracy.png")
    plot_attacks(attacks, "results/fig_attack_vs_training.png")

    lines = ["# Results summary", "", "Generated by `scripts/summarize_results.py` from `results/runs/`.",
             "", "## Training", ""] + training_table(all_runs)
    diffs = mask_check(all_runs)
    if diffs:
        lines += ["", f"Masking vs baseline, same seed, final accuracy difference: "
                      f"{', '.join(f'{d:+.4f}' for d in diffs)}."]
    if attacks:
        lines += ["", "## Attacking the real global model during training", ""] + attack_table(attacks)
    for name, heading in (("evaluation", "Attack vs defense (MNIST)"),
                          ("evaluation_sweep", "DP noise sweep (MNIST)"),
                          ("evaluation_cifar10", "Attack vs defense (CIFAR-10)")):
        path = f"results/{name}.md"
        if os.path.exists(path):
            with open(path) as f:
                body = re.sub(r"^# ", "", f.read()).strip().split("\n")
            lines += ["", f"## {heading}", "", f"*{body[0]}*"] + body[1:]
    with open("results/summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
