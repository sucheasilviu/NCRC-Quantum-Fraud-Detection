"""Slide figure: test AUPRC, change in AUPRC from noiseless, and Spearman rho vs noise (official protocol)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import RESULTS_DIR

LINES = [("baseline", "Baseline VQC (126 noise loc.)", "#C8553D", "-"),
         ("ncrc_4q_L4_cnot", "NCRC + CNOTs (ablation, 34)", "#8A8F98", "--"),
         ("ncrc_4q_L4", "NCRC, ours (16 noise loc.)", "#2A6FDB", "-")]
P = [0.0, 0.001, 0.01, 0.05]


def per_seed(tag):
    rows = [{"seed": r["seed"], **x} for f in sorted((RESULTS_DIR / "raw").glob(f"{tag}_seed*.json"))
            for r in [json.loads(f.read_text())] for x in r["protocol"]]
    df = pd.DataFrame(rows)
    # change from the same seed's noiseless AUPRC, so the spread is of the change itself
    df["Spearman rho"] = df.get("Spearman rho", 1.0)  # logreg rows have none: no gates, nothing to compare
    df["dAUPRC"] = df["AUPRC"] - df.groupby("seed")["AUPRC"].transform(lambda s: s.iloc[0])
    return df.groupby("p")[["AUPRC", "dAUPRC", "Spearman rho"]].agg(["mean", "std"]).reindex(P)


plt.rcParams.update({"axes.facecolor": "#FBFBF8", "font.size": 15, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(21, 7.2))
lr = per_seed("logreg")
m, s = lr[("AUPRC", "mean")].iloc[0], lr[("AUPRC", "std")].iloc[0]
# classical model has no gates: a constant reference band, not a series measured under noise
axes[0].axhspan(m - s, m + s, color="#14213D", alpha=0.12)
axes[0].axhline(m, color="#14213D", linestyle=":", linewidth=2.5, label="Logistic regression (classical, no gates)")
axes[1].axhline(0, color="#14213D", linewidth=1)
for tag, label, color, ls in LINES:
    d = per_seed(tag)
    for ax, k in zip(axes, ["AUPRC", "dAUPRC", "Spearman rho"]):
        ax.errorbar(range(4), d[(k, "mean")], yerr=d[(k, "std")], color=color, linestyle=ls, marker="o",
                    linewidth=3, markersize=8, capsize=5, label=label)
titles = ["Test AUPRC", "Change in AUPRC from noiseless", "Spearman ρ, noisy vs noiseless"]
for ax, title in zip(axes, titles):
    ax.set_xticks(range(4), ["0", "0.001", "0.01", "0.05"])
    ax.set_xlabel("depolarizing probability p per gate")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(alpha=0.25)
axes[0].set_ylim(0, 1.02)
axes[2].set_ylim(0, 1.02)
axes[1].set_ylim(-0.4, 0.05)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=14)
plt.tight_layout(rect=(0, 0.1, 1, 1))
plt.savefig(RESULTS_DIR / "deck_noise.png", dpi=110, facecolor="#FBFBF8")
for tag, *_ in LINES:
    d = per_seed(tag)
    print(tag, " ".join(f"p={p}: {d.loc[p, ('dAUPRC', 'mean')]:+.4f}±{d.loc[p, ('dAUPRC', 'std')]:.4f}" for p in P[1:]))
