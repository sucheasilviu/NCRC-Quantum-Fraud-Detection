"""Minimal slide figures built from saved results: results/slides/*.png

    python src/plot_slides.py

scatter.png   every test transaction's score without noise vs at p = 0.05 (seed 0, 500-row fraud-rich set)
practice.png  frauds caught, missed and false alarms on the full test set at p = 0.05 (mean of 5 seeds)
worst.png     worst AUPRC over all runs at mean noise 0.2, per model and noise type (src/noise_types.py)
cost.png      total circuit executions per training run, with test AUPRC at p = 0.05
The scatter needs model scores, so it trains seed 0 once and caches them in results/slides/scores_seed0.npz.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import RESULTS_DIR, load_data, make_model, noisy_predict, predict, prepare_seed, train_vqc
from models import MODELS
from stress import fraud_rich

OUT = RESULTS_DIR / "slides"
INK, MUTED, BG = "#14213D", "#6A7488", "#FBFBF8"
RED, BLUE, GRAY = "#C8553D", "#2A6FDB", "#8A8F98"
NAMES = {"baseline": "Baseline VQC", "ncrc_4q_L4": "NCRC (ours)", "ncrc_4q_L4_cnot": "NCRC + CNOTs",
         "ncrc_1q_L2_pruned": "NCRC 1 qubit (ours)", "logreg": "Logistic regression"}
plt.rcParams.update({"font.size": 18, "axes.facecolor": BG, "figure.facecolor": BG, "text.color": INK,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#C9CED6",
                     "axes.spines.top": False, "axes.spines.right": False})


def raw(tag):
    return [json.loads(f.read_text()) for f in sorted((RESULTS_DIR / "raw").glob(f"{tag}_seed[0-9].json"))]


def at(run, p):
    return next(r for r in run["protocol"] if abs(r["p"] - p) < 1e-9)


def scatter():
    cache = OUT / "scores_seed0.npz"
    if not cache.exists():
        d = prepare_seed(load_data(), 0)
        it = fraud_rich(d["y_test"], 0)
        Zt, s = d["Z_test"][it], {"y": d["y_test"][it]}
        for tag in ("baseline", "ncrc_4q_L4"):
            M = MODELS[tag]
            model = make_model(M["circuit"], M["n_wires"])
            w, _, _ = train_vqc(model, M["init"], d["Z_bal"], d["y_bal"], d["Z_val"], d["y_val"], 0)
            s[f"{tag}_clean"], s[f"{tag}_noisy"] = predict(model, Zt, w), noisy_predict(Zt, w, 0.05, M["circuit"], M["n_wires"])
        np.savez(cache, **s)
    s = np.load(cache)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.4))
    for ax, tag in zip(axes, ("baseline", "ncrc_4q_L4")):
        x, y, fraud = s[f"{tag}_clean"], s[f"{tag}_noisy"], s["y"] == 1
        ax.scatter(x[~fraud], y[~fraud], s=22, color=GRAY, alpha=0.45, linewidth=0, label="legitimate")
        ax.scatter(x[fraud], y[fraud], s=34, color=RED, alpha=0.85, linewidth=0, label="fraud")
        rho = spearmanr(x, y).statistic
        ax.set_title(f"{NAMES[tag]}   ·   rank agreement ρ = {rho:.3f}", loc="left", fontsize=20, fontweight="bold", color=INK)
        ax.set_xlabel("score without noise")
        ax.set_ylabel("score at p = 0.05")
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, loc="upper left")
    plt.tight_layout()
    plt.savefig(OUT / "scatter.png", dpi=110)


def practice():
    rows = []
    for tag in ("baseline", "ncrc_4q_L4_cnot", "ncrc_4q_L4", "logreg"):
        r = pd.DataFrame([at(run, 0.05 if tag != "logreg" else 0.0) for run in raw(tag)])
        rows.append((NAMES[tag], r.TP.mean(), r.FN.mean(), r.FP.mean()))
    fig, ax = plt.subplots(figsize=(16, 6.2))
    for i, (name, tp, fn, fp) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, tp, color=BLUE, height=0.55)
        ax.barh(y, fn, left=tp, color="#C9D8F2", height=0.55)
        ax.barh(y, fp, left=tp + fn + 4, color=RED, height=0.55)
        ax.text(tp / 2, y, f"{tp:.0f} caught", ha="center", va="center", color="white", fontweight="bold")
        ax.text(tp + fn / 2, y, f"{fn:.0f}", ha="center", va="center", color=INK)
        ax.text(tp + fn + 4 + fp + 2, y, f"{fp:.0f} false alarms", va="center", color=INK)
    top = len(rows) - 0.45
    ax.text(0, top, "the 98 test frauds: caught  |  missed", color=MUTED, va="bottom")
    ax.text(102, top, "legitimate transactions flagged", color=MUTED, va="bottom")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], color=INK)
    ax.set_xticks([])
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlim(0, 210)
    ax.set_ylim(-0.5, len(rows) - 0.1)
    plt.tight_layout()
    plt.savefig(OUT / "practice.png", dpi=110)


def worst():
    df = pd.DataFrame([{"model": r["model"], **x} for f in sorted((RESULTS_DIR / "noise_types").glob("*.json"))
                       for r in [json.loads(f.read_text())] for x in r["rows"]])
    kinds = {"depolarizing": "depolarizing", "amplitude_damping": "amplitude\ndamping", "phase_damping": "phase\ndamping",
             "bit_flip": "bit flip", "hetero_depolarizing": "uneven\ndepolarizing", "hetero_mixed": "uneven\nmix"}
    models = ["baseline", "ncrc_4q_L4_cnot", "ncrc_4q_L4", "ncrc_1q_L2_pruned"]
    w = df[np.isclose(df.p, 0.2)].groupby(["model", "kind"]).AUPRC.min().unstack()[list(kinds)].loc[models]
    lr = df[df.model == "logreg"].AUPRC.mean()
    fig, ax = plt.subplots(figsize=(16, 6.4))
    ax.imshow(w.values, cmap="Blues", vmin=0, vmax=1.15, aspect="auto")
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            v = w.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=24, fontweight="bold",
                    color="white" if v > 0.6 else INK)
    ax.set_xticks(range(len(kinds)), list(kinds.values()), color=INK)
    ax.set_yticks(range(len(models)), [NAMES[m] for m in models], color=INK)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    ax.set_xlabel(f"worst test AUPRC of all runs at mean noise 0.2  ·  random guessing = 0.20  ·  "
                  f"logistic regression = {lr:.2f} under every noise", labelpad=14)
    plt.tight_layout()
    plt.savefig(OUT / "worst.png", dpi=110)


def cost():
    rows = []
    for tag, label in (("baseline", "Baseline VQC"), ("ncrc_4q_L4", "NCRC 4q, standard gradients"),
                       ("ncrc_4q_L4_packed", "NCRC 4q, packed (default)"),
                       ("ncrc_4q_L4_cheap_packed", "NCRC 4q packed, 15 ep., 20% val."),
                       ("ncrc_1q_L2_pruned_cheap", "NCRC 1q pruned, 15 ep., 20% val.")):
        runs = raw(tag)
        rows.append((label, runs[0]["ledger"]["total_executions"] / 1e6, np.mean([at(r, 0.05)["AUPRC"] for r in runs]), tag))
    fig, ax = plt.subplots(figsize=(16, 6.4))
    for i, (label, ex, auprc, tag) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, ex, color=RED if tag == "baseline" else BLUE, height=0.55)
        ax.text(ex + 0.05, y, f"{ex:.2f} M   ·   AUPRC {auprc:.3f}", va="center", color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], color=INK)
    ax.set_xlabel("circuit executions to train (millions)")
    ax.spines[["left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 5)
    plt.tight_layout()
    plt.savefig(OUT / "cost.png", dpi=110)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    practice(), worst(), cost(), scatter()
    print("written:", sorted(p.name for p in OUT.glob("*.png")))
