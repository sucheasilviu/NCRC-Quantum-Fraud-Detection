"""Plot src/noise_types.py: results/noise_types.png and results/noise_types_summary.md.

One panel per noise kind. Thin lines = every individual run (each seed, and for hetero kinds each random device);
thick line = their mean. Every marker is a measured noise level. The summary reports the mean AND the worst run.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import RESULTS_DIR

LINES = {"baseline": ("Baseline VQC", "#C8553D", "-"),
         "ncrc_4q_L4_cnot": ("NCRC + CNOTs (ablation)", "#8A8F98", "--"),
         "ncrc_1q_L2_pruned": ("NCRC 1 qubit (ours)", "#7FA8F0", "-"),
         "ncrc_4q_L4": ("NCRC 4 qubits (ours)", "#2A6FDB", "-")}
KINDS = {"depolarizing": "Depolarizing (the challenge's noise)", "amplitude_damping": "Amplitude damping (energy loss, T1)",
         "phase_damping": "Phase damping (dephasing, T2)", "bit_flip": "Bit flip",
         "hetero_depolarizing": "Uneven depolarizing: own p per gate", "hetero_mixed": "Uneven mix: own type and p per gate"}

df = pd.DataFrame([{"model": r["model"], "seed": r["seed"], **x}
                   for f in sorted((RESULTS_DIR / "noise_types").glob("*.json")) for r in [json.loads(f.read_text())]
                   for x in r["rows"]])
lr = df[df.model == "logreg"].groupby("seed")["AUPRC"].first()

plt.rcParams.update({"axes.facecolor": "#FBFBF8", "font.size": 14, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 3, figsize=(21, 12.5), sharey=True)
for ax, (kind, title) in zip(axes.flat, KINDS.items()):
    ax.axhspan(lr.mean() - lr.std(), lr.mean() + lr.std(), color="#14213D", alpha=0.12)
    ax.axhline(lr.mean(), color="#14213D", linestyle=":", linewidth=2.5, label="Logistic regression (classical)")
    for tag, (label, color, ls) in LINES.items():
        d = df[(df.model == tag) & (df.kind == kind)]
        if d.empty:
            continue
        for _, run in d.groupby(["seed", "device"]):
            ax.plot(run.p, run.AUPRC, color=color, linestyle=ls, linewidth=0.8, alpha=0.35)
        m = d.groupby("p").AUPRC.mean()
        ax.plot(m.index, m.values, color=color, linestyle=ls, linewidth=3, marker="o", markersize=5, label=label)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("mean noise strength per gate")
    ax.set_xlim(0, 0.2)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
for ax in axes[:, 0]:
    ax.set_ylabel("Test AUPRC (500-row fraud-rich set)")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=15)
plt.tight_layout(rect=(0, 0.05, 1, 1))
plt.savefig(RESULTS_DIR / "noise_types.png", dpi=100, facecolor="#FBFBF8")

q = df[df.model != "logreg"]
lines = ["# Noise types and uneven noise (src/noise_types.py)", "",
         "Exact scores, 500-row fraud-rich test set, 5 seeds; uneven kinds: 5 random devices per seed (25 runs).",
         f"Logistic regression (no gates): AUPRC {lr.mean():.3f} ± {lr.std():.3f} under every noise.", "",
         "| model | noise | runs | AUPRC p=0 | mean p=0.1 | worst p=0.1 | mean p=0.2 | worst p=0.2 | worst ρ p=0.2 |",
         "|---|---|---|---|---|---|---|---|---|"]
for (tag, kind), d in q.groupby(["model", "kind"], sort=False):
    at = lambda p: d[d.p.round(3) == p]
    lines.append(f"| {tag} | {kind} | {len(at(0.0))} | {at(0.0).AUPRC.mean():.3f} | {at(0.1).AUPRC.mean():.3f} | "
                 f"{at(0.1).AUPRC.min():.3f} | {at(0.2).AUPRC.mean():.3f} | {at(0.2).AUPRC.min():.3f} | "
                 f"{at(0.2)['Spearman rho'].min():.3f} |")
(RESULTS_DIR / "noise_types_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
