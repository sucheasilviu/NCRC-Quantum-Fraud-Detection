"""Plot the dense sweeps (src/sweep.py): results/sweep_protocol.png and results/sweep_stress.png.

Every marker is a measured noise level (no interpolation); the band is ± 1 std over seeds 0-4
(stress: each seed first averaged over its shot draws).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import RESULTS_DIR

LINES = {"baseline": ("Baseline VQC (126 noise loc.)", "#C8553D", "-"),
         "ncrc_4q_L4_cnot": ("NCRC + CNOTs (ablation)", "#8A8F98", "--"),
         "ncrc_1q_L2_pruned": ("NCRC 1 qubit, 2 gates (ours)", "#7FA8F0", "-"),
         "ncrc_4q_L4": ("NCRC 4 qubits, 16 gates (ours)", "#2A6FDB", "-")}


def load(mode):
    rows = [{"model": r["model"], "seed": r["seed"], **x}
            for f in sorted((RESULTS_DIR / "sweep" / mode).glob("*.json")) for r in [json.loads(f.read_text())] for x in r["rows"]]
    df = pd.DataFrame(rows).groupby(["model", "seed", "p"])[["AUPRC", "Spearman rho", "F1"]].mean().reset_index()
    df["dAUPRC"] = df["AUPRC"] - df.groupby(["model", "seed"])["AUPRC"].transform(lambda s: s.iloc[0])
    return df.groupby(["model", "p"])[["AUPRC", "dAUPRC", "Spearman rho", "F1"]].agg(["mean", "std"])


def draw(mode, panels, out, xlabel_max):
    agg = load(mode)
    plt.rcParams.update({"axes.facecolor": "#FBFBF8", "font.size": 15, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 7.2))
    if "logreg" in agg.index.get_level_values(0):
        lr = agg.loc["logreg"]
        m, s = lr[("AUPRC", "mean")].iloc[0], lr[("AUPRC", "std")].iloc[0]
        axes[0].axhspan(m - s, m + s, color="#14213D", alpha=0.12)
        axes[0].axhline(m, color="#14213D", linestyle=":", linewidth=2.5, label="Logistic regression (classical, no gates)")
    for tag, (label, color, ls) in LINES.items():
        if tag not in agg.index.get_level_values(0):
            continue
        d = agg.loc[tag]
        for ax, (k, _, _) in zip(axes, panels):
            mean, std = d[(k, "mean")], d[(k, "std")]
            ax.fill_between(d.index, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
            ax.plot(d.index, mean, color=color, linestyle=ls, linewidth=2.5, marker="o", markersize=4, label=label)
    for ax, (k, title, ylim) in zip(axes, panels):
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("depolarizing probability p per gate")
        ax.set_xlim(0, xlabel_max)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
        if k == "dAUPRC":
            ax.axhline(0, color="#14213D", linewidth=1)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=14)
    plt.tight_layout(rect=(0, 0.13, 1, 1))
    plt.savefig(RESULTS_DIR / out, dpi=110, facecolor="#FBFBF8")
    print(out, "points per curve:", agg.loc[next(iter(agg.index.get_level_values(0)))].shape[0])


draw("protocol", [("AUPRC", "Test AUPRC (exact, full test set)", (0, 1.02)),
                  ("dAUPRC", "Change in AUPRC from noiseless", (-0.4, 0.05)),
                  ("Spearman rho", "Spearman ρ, noisy vs noiseless", (0, 1.02))], "sweep_protocol.png", 0.05)
draw("stress", [("AUPRC", "Test AUPRC, 1,024 shots, 500-row set", (0, 1.02)),
                ("Spearman rho", "Spearman ρ, shot-noisy vs noiseless", (0, 1.02))], "sweep_stress.png", 0.5)
