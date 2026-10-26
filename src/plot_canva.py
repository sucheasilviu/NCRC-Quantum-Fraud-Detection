"""Figures for the Canva-style deck (same chart types and look as the Task 2 alternative notebook), from our results.

    python src/demo_seed0.py && python src/plot_canva.py   -> results/canva/*.png

Main models: logistic regression ("Linear"), baseline VQC ("Original VQC"), NCRC + CNOTs ("Entangled NCRC"),
NCRC 4 qubits ("NCRC, ours"). Five-seed numbers come from results/raw, results/shots, results/sweep and
results/noise_types; seed-0 per-transaction plots come from results/demo/seed0.npz.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve

from common import RESULTS_DIR, choose_threshold

OUT = RESULTS_DIR / "canva"
BG, INK, MUTED = "#F7F6F2", "#24212F", "#706A7C"
CAUGHT, MISSED = "#6B7F2A", "#7555DE"
MODELS = {"logreg": ("Linear", "#24212F"), "baseline": ("Original VQC", "#A7A2B3"),
          "ncrc_4q_L4_cnot": ("Entangled NCRC", "#AE99E9"), "ncrc_4q_L4": ("NCRC (ours)", "#7555DE")}
QUANTUM = ["baseline", "ncrc_4q_L4_cnot", "ncrc_4q_L4"]
TEST_LABEL = "Full common test: 56,962 transactions, 98 frauds (0.172%)"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "font.size": 13,
                     "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.edgecolor": "#D9D6CF", "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False})


def raw(tag):
    return [json.loads(f.read_text()) for f in sorted((RESULTS_DIR / "raw").glob(f"{tag}_seed[0-9].json"))]


def protocol():
    rows = [{"model": t, "seed": r["seed"], **x} for t in MODELS for r in raw(t) for x in r["protocol"]]
    df = pd.DataFrame(rows)
    lr = df[df.model == "logreg"]
    df = pd.concat([df[df.model != "logreg"]] + [lr.assign(p=p) for p in (0.0, 0.01, 0.05)])  # no gates: same at every p
    return df


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def bars_auprc_f1(df):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2))
    for row, p in enumerate((0.01, 0.05)):
        for col, metric in enumerate(("AUPRC", "F1")):
            ax = axes[row, col]
            g = df[np.isclose(df.p, p)].groupby("model")[metric].agg(["mean", "std"]).loc[list(MODELS)]
            ax.bar(range(4), g["mean"], yerr=g["std"], capsize=5, color=[c for _, c in MODELS.values()], ecolor=INK)
            for i, v in enumerate(g["mean"]):
                ax.text(i, 0.025, f"{v:.3f}", ha="center", color="white", fontweight="bold")
            ax.set_xticks(range(4), [n for n, _ in MODELS.values()])
            ax.set_ylim(0, 1.05)
            ax.set_title(f"p={p}: {metric} — higher is better (ideal = 1)")
            if metric == "AUPRC":
                ax.axhline(98 / 56962, color="#555", ls="--", label="Random ranking (0.172%)")
                ax.legend(fontsize=9)
    fig.suptitle(TEST_LABEL + " · mean ± std over five seeds")
    fig.tight_layout()
    save(fig, "bars_auprc_f1.png")


def headline_bars(df):
    """Slide-3 style: AUPRC at both noise levels, one compact chart."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    for ax, p in zip(axes, (0.01, 0.05)):
        g = df[np.isclose(df.p, p)].groupby("model")["AUPRC"].agg(["mean", "std"]).loc[list(MODELS)]
        ax.bar(range(4), g["mean"], yerr=g["std"], capsize=5, color=[c for _, c in MODELS.values()], ecolor=INK)
        for i, (m, s) in enumerate(zip(g["mean"], g["std"])):
            ax.text(i, m + s + 0.02, f"{m:.3f}\n± {s:.3f}", ha="center", fontsize=12)
        ax.set_xticks(range(4), [n for n, _ in MODELS.values()])
        ax.set_ylim(0, 1)
        ax.set_title(f"Gate noise p = {p}", fontweight="bold")
    axes[0].set_ylabel("Test AUPRC")
    fig.tight_layout()
    save(fig, "headline_auprc.png")


def delta_and_rho(df):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, p in zip(axes[:2], (0.01, 0.05)):
        piv = df[np.isclose(df.p, p)].pivot(index="seed", columns="model", values="AUPRC")
        for i, t in enumerate(QUANTUM):
            delta = piv[t] - piv["logreg"]
            ax.bar(i, delta.mean(), yerr=delta.std(ddof=1), capsize=5, color=MODELS[t][1], ecolor=INK)
            ax.text(i, delta.mean() + (0.012 if delta.mean() >= 0 else -0.03), f"{delta.mean():+.3f}", ha="center")
        ax.axhline(0, color=INK, lw=1)
        ax.set_xticks(range(3), [MODELS[t][0] for t in QUANTUM])
        ax.set_title(f"p={p}: VQC AUPRC minus linear AUPRC\nAbove 0: VQC better; below 0: linear better")
        ax.set_ylabel("Paired AUPRC difference (mean ± std)")
    for t in QUANTUM:
        g = df[df.model == t].groupby("p")["Spearman rho"].agg(["mean", "std"]).loc[[0.0, 0.01, 0.05]]
        axes[2].errorbar(range(3), g["mean"], yerr=g["std"], marker="o", capsize=4, color=MODELS[t][1], label=MODELS[t][0])
    axes[2].axhline(1, color="#888", ls="--")
    axes[2].set_xticks(range(3), ["Clean", "p=0.01", "p=0.05"])
    axes[2].set(title="Ranking stability under noise\nHigher rho = better order preservation", ylabel="Spearman rho", ylim=(0, 1.1))
    axes[2].legend(fontsize=9)
    fig.suptitle(TEST_LABEL)
    fig.tight_layout()
    save(fig, "delta_rho.png")


def demo():
    s = np.load(RESULTS_DIR / "demo" / "seed0.npz")
    y = s["y_test"]

    def scores(t, p):
        if t == "logreg":
            return s["logreg_val"], s["logreg_test"]
        return s[f"{t}_val_{p}"], s[f"{t}_test_{p}"]

    # precision-recall curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax, p in zip(axes, (0.01, 0.05)):
        for t, (name, color) in MODELS.items():
            st = scores(t, p)[1]
            prec, rec, _ = precision_recall_curve(y, st)
            ax.step(rec, prec, where="post", color=color, lw=2, label=f"{name}: AP {average_precision_score(y, st):.3f}")
        ax.axhline(y.mean(), ls="--", color="#555", label="Random (0.172%)")
        ax.set(xlabel="Recall — higher catches more fraud", ylabel="Precision — higher means fewer false alarms",
               title=f"p={p}: closer to top-right is better")
        ax.legend(fontsize=10, loc="lower left")
    fig.suptitle(TEST_LABEL + " · seed 0")
    fig.tight_layout()
    save(fig, "pr_curves.png")

    # confusion matrices
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.5))
    for row, p in enumerate((0.01, 0.05)):
        for col, (t, (name, _)) in enumerate(MODELS.items()):
            sv, st = scores(t, p)
            cm = confusion_matrix(y, st >= choose_threshold(s["y_val"], sv), labels=[0, 1])
            ax = axes[row, col]
            # good cells (TN, TP) tinted purple/green, error cells (FP, FN) a light lavender, so every cell is visible
            from matplotlib.colors import to_rgb
            ax.imshow(np.array([[to_rgb("#7555DE"), to_rgb("#E4DDF7")], [to_rgb("#E4DDF7"), to_rgb("#D3EE8D")]]))
            for (i, j), v in np.ndenumerate(cm):
                lab = [["TN", "FP"], ["FN", "TP"]][i][j]
                ax.text(j, i, f"{lab}: {v:,}\n{'higher' if lab in ('TN', 'TP') else 'lower'} better", ha="center",
                        va="center", color="white" if (i, j) == (0, 0) else INK, fontsize=11)
            ax.set_facecolor("#FFFFFF")
            ax.set_xticks([0, 1], ["Legit", "Fraud"])
            ax.set_yticks([0, 1], ["Legit", "Fraud"])
            ax.set(title=name, xlabel="Predicted", ylabel=f"p={p}: actual")
            ax.grid(False)
    fig.suptitle(TEST_LABEL + " · confusion matrices · seed 0")
    fig.tight_layout()
    save(fig, "confusion.png")

    # every test fraud against the alert threshold
    rng = np.random.default_rng(42)
    idx = np.sort(np.r_[np.flatnonzero(y == 1), rng.choice(np.flatnonzero(y == 0), 500, replace=False)])

    def margin_panel(ax, t, p, legend=False, title=True):
        sv, st = scores(t, p)
        m, a = st[idx] - choose_threshold(s["y_val"], sv), y[idx]
        pos = np.arange(len(idx))
        ax.scatter(pos[a == 0], m[a == 0], s=10, color="#B9B3C7", alpha=0.7, label="Legitimate examples")
        ax.scatter(pos[(a == 1) & (m >= 0)], m[(a == 1) & (m >= 0)], marker="*", s=140, color=CAUGHT, label="Fraud caught")
        ax.scatter(pos[(a == 1) & (m < 0)], m[(a == 1) & (m < 0)], marker="*", s=140, color=MISSED, label="Fraud missed")
        ax.axhline(0, color=INK, ls="--", label="Alert threshold")
        caught = int(((a == 1) & (m >= 0)).sum())
        if title:
            ax.set_title(f"{MODELS[t][0]} · p={p}: {caught} of 98 caught")
        return caught

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    margin_panel(ax, "ncrc_4q_L4", 0.05, title=False)
    ax.set(xlabel="Display position (all 98 frauds + 500 legitimate examples)", ylabel="Fraud score minus alert threshold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=False)
    fig.tight_layout()
    save(fig, "threshold_ours.png")

    fig, axes = plt.subplots(2, 4, figsize=(19, 8))
    for row, p in enumerate((0.01, 0.05)):
        for col, t in enumerate(MODELS):
            margin_panel(axes[row, col], t, p)
            axes[row, col].set_ylabel("Score − threshold" if col == 0 else "")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("All 98 test frauds + 500 legitimate rows for display · seed 0 · metrics use the full test")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "threshold_grid.png")

    # input-response slices
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("pg", [MISSED, "#F1EEF7", "#9CBF45"])
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
    ax_vals = s["axis"]
    for ax, (t, (name, _)) in zip(axes, MODELS.items()):
        thr = choose_threshold(s["y_val"], scores(t, 0.05)[0])
        z = (s["logreg_grid"] if t == "logreg" else s[f"{t}_grid"]).reshape(len(ax_vals), len(ax_vals)).T - thr
        lim = max(float(np.abs(z).max()), 1e-9)
        im = ax.contourf(ax_vals, ax_vals, z, levels=np.linspace(-lim, lim, 17), cmap=cmap)
        if z.min() < 0 < z.max():
            ax.contour(ax_vals, ax_vals, z, levels=[0], colors=INK, linewidths=1.5)
        ax.set(title=name, xlabel="Encoded feature 1", ylabel="Encoded feature 2")
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="Score − validation threshold")
    fig.suptitle("Input-response slice at p=0.05 · other angles fixed to training medians · positive = flagged")
    fig.tight_layout()
    save(fig, "slices.png")


def ablations():
    tags = {"logreg": "Linear", "baseline": "Original VQC", "ncrc_4q_L4_cnot": "Entangled NCRC", "ncrc_4q_L4": "NCRC 4 qubits (ours)",
            "ncrc_4q_L4_pairwise05": "NCRC + ranking loss", "ncrc_4q_L2": "NCRC 4 qubits, 2 layers",
            "ncrc_1q_L2": "NCRC 1 qubit, 2 layers", "ncrc_1q_L2_pruned": "NCRC 1 qubit, pruned", "ncrc_1q_L4": "NCRC 1 qubit, 4 layers",
            "ncrc_1q_L6": "NCRC 1 qubit, 6 layers", "ncrc_4q_L4_cheap": "NCRC 4 qubits, cheap training",
            "ncrc_1q_L2_pruned_cheap": "NCRC 1 qubit, cheap training"}
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for ax, p in zip(axes, (0.0, 0.05)):
        m, sd, colors = [], [], []
        for t in tags:
            v = [next(x for x in r["protocol"] if x["p"] == (0.0 if t == "logreg" else p))["AUPRC"] for r in raw(t)]
            m.append(np.mean(v)), sd.append(np.std(v, ddof=1))
            colors.append(MODELS.get(t, (None, "#C9BCEF"))[1])
        ax.barh(range(len(tags)), m, xerr=sd, capsize=3, color=colors, ecolor=INK)
        for i, v in enumerate(m):
            ax.text(0.01, i, f"{v:.3f}", va="center", color="white", fontsize=10, fontweight="bold")
        ax.set_yticks(range(len(tags)), list(tags.values()))
        ax.invert_yaxis()
        ax.set(xlabel="Test AUPRC — higher is better", title=f"Five seeds · p={p}", xlim=(0, 0.85))
    fig.suptitle("Every variant we ran (configuration chosen on validation; test shown for completeness)")
    fig.tight_layout()
    save(fig, "ablations.png")


def schedule_and_gradients():
    full, cheap = raw("ncrc_4q_L4_packed"), raw("ncrc_4q_L4_cheap_packed")
    a = [[next(x for x in r["protocol"] if x["p"] == 0.05)["AUPRC"] for r in runs] for runs in (full, cheap)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    axes[0].bar(["Full schedule", "Cheap schedule"], [np.mean(v) for v in a], yerr=[np.std(v, ddof=1) for v in a],
                capsize=5, color=["#7555DE", "#AE99E9"], ecolor=INK)
    axes[0].set(title="Test AUPRC at p=0.05\nHigher is better", ylim=(0, 0.85))
    ex = [full[0]["ledger"]["total_executions"] / 1e6, cheap[0]["ledger"]["total_executions"] / 1e6]
    axes[1].bar(["Full schedule", "Cheap schedule"], ex, color=["#7555DE", "#AE99E9"])
    for i, v in enumerate(ex):
        axes[1].text(i, v, f"{v:.2f} M", ha="center", va="bottom")
    axes[1].set(title="Executions per training run\nLower is cheaper", ylabel="Million executions")
    per = {"Original\nVQC": 41, "NCRC\nstandard": 81, "NCRC\npacked": 15, "NCRC\n1 qubit": 9}
    axes[2].bar(list(per), list(per.values()), color=["#A7A2B3", "#AE99E9", "#7555DE", "#7555DE"])
    for i, v in enumerate(per.values()):
        axes[2].text(i, v, str(v), ha="center", va="bottom")
    axes[2].set_title("Gradient executions per training example\nLower is cheaper; packed = identical results")
    fig.tight_layout()
    save(fig, "schedule_gradients.png")


def per_seed_shots(t):
    """Shot-test rows averaged over the shot draws of each seed, so the spread is over the 5 seeds."""
    rows = pd.DataFrame([{"seed": r["seed"], **x} for f in sorted((RESULTS_DIR / "shots").glob(f"{t}_seed[0-9].json"))
                         for r in [json.loads(f.read_text())] for x in r["rows"]])
    return rows.groupby(["seed", "p"]).mean(numeric_only=True).reset_index()


def shots():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, t in zip(axes[:2], ("ncrc_4q_L4", "baseline")):
        rows = per_seed_shots(t)
        rows = rows[rows.p.isin([0.0, 0.01, 0.05])]
        for col, label, color in (("AUPRC exact", "Exact (infinite shots)", "#AE99E9"), ("AUPRC shots", "1,024 shots", "#7555DE")):
            g = rows.groupby("p")[col].agg(["mean", "std"])
            ax.errorbar(range(3), g["mean"], yerr=g["std"], marker="o", capsize=4, color=color, label=label)
        ax.set_xticks(range(3), ["Clean", "p=0.01", "p=0.05"])
        ax.set(title=f"{MODELS[t][0]}: test AUPRC — higher is better", ylim=(0, 0.85))
        ax.legend(fontsize=9)
    for t in ("baseline", "ncrc_4q_L4"):
        rows = per_seed_shots(t)
        g = rows[rows.p.isin([0.0, 0.01, 0.05])].groupby("p")["Spearman rho shots vs clean"].agg(["mean", "std"])
        axes[2].errorbar(range(3), g["mean"], yerr=g["std"], marker="o", capsize=4, color=MODELS[t][1], label=MODELS[t][0])
    axes[2].set_xticks(range(3), ["Clean", "p=0.01", "p=0.05"])
    axes[2].set(title="Ranking kept with 1,024 shots\nHigher rho = better", ylim=(0, 1.05))
    axes[2].legend(fontsize=9)
    fig.suptitle("Realistic readout: every score estimated from 1,024 measurements · full test set · five seeds × five draws")
    fig.tight_layout()
    save(fig, "shots.png")


def cost():
    names = [MODELS[t][0].replace(" ", "\n", 1) for t in QUANTUM]
    loc = [raw(t)[0]["resources"]["noise_locations"] for t in QUANTUM]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    axes[0].bar(names, loc, color=[MODELS[t][1] for t in QUANTUM])
    for i, v in enumerate(loc):
        axes[0].text(i, v, str(v), ha="center", va="bottom")
    axes[0].set(title="Noise locations — lower is better", ylim=(0, max(loc) * 1.2))
    per = [41, 97, 15]
    axes[1].bar(names, per, color=[MODELS[t][1] for t in QUANTUM])
    for i, v in enumerate(per):
        axes[1].text(i, v, str(v), ha="center", va="bottom")
    axes[1].set(title="Executions per training example\nOurs with packed gradients", ylim=(0, 115))
    tot = [0] + [raw(t)[0]["ledger"]["total_executions"] / 1e6 for t in ("baseline", "ncrc_4q_L4_cnot", "ncrc_4q_L4_packed")]
    axes[2].bar(["Linear"] + names, tot, color=[c for _, c in MODELS.values()])
    for i, v in enumerate(tot):
        axes[2].text(i, v, f"{v:.2f}", ha="center", va="bottom")
    axes[2].set(title="Training + validation executions per seed\nLower is cheaper; linear uses zero", ylabel="Million executions")
    fig.tight_layout()
    save(fig, "cost.png")


def sweeps():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    for ax, mode, xmax, title in ((axes[0], "protocol", 0.05, "Official protocol: 21 noise levels, full test set"),
                                  (axes[1], "stress", 0.5, "Stress test: 26 levels, 1,024 shots, 500 fraud-rich rows")):
        rows = pd.DataFrame([{"model": r["model"], "seed": r["seed"], **x} for f in sorted((RESULTS_DIR / "sweep" / mode).glob("*.json"))
                             for r in [json.loads(f.read_text())] for x in r["rows"]])
        g = rows.groupby(["model", "seed", "p"]).AUPRC.mean().groupby(["model", "p"]).agg(["mean", "std"])
        for t, (name, color) in MODELS.items():
            d = g.loc[t]
            if t == "logreg":
                ax.axhline(d["mean"].iloc[0], color=color, ls=":", lw=2.5, label=name)
                continue
            ax.fill_between(d.index, d["mean"] - d["std"], d["mean"] + d["std"], color=color, alpha=0.2, lw=0)
            ax.plot(d.index, d["mean"], color=color, marker="o", ms=3.5, lw=2.2, label=name)
        ax.set(title=title, xlabel="Gate noise p", ylabel="Test AUPRC", xlim=(0, xmax), ylim=(0, 1.02))
    axes[0].legend(fontsize=10, loc="lower left")
    fig.suptitle("Every dot is a measured noise level · band = ± 1 std over five seeds")
    fig.tight_layout()
    save(fig, "sweeps.png")


def noise_types():
    df = pd.DataFrame([{"model": r["model"], **x} for f in sorted((RESULTS_DIR / "noise_types").glob("*.json"))
                       for r in [json.loads(f.read_text())] for x in r["rows"]])
    kinds = {"depolarizing": "Depolarizing", "amplitude_damping": "Amplitude\ndamping", "phase_damping": "Phase\ndamping",
             "bit_flip": "Bit flip", "hetero_depolarizing": "Uneven\ndepolarizing", "hetero_mixed": "Uneven mix"}
    w = df[np.isclose(df.p, 0.2)].groupby(["model", "kind"]).AUPRC.min().unstack()[list(kinds)].loc[QUANTUM]
    fig, ax = plt.subplots(figsize=(14, 4.6))
    x = np.arange(len(kinds))
    for i, t in enumerate(QUANTUM):
        ax.bar(x + (i - 1) * 0.27, w.loc[t], width=0.25, color=MODELS[t][1], label=MODELS[t][0])
        for xi, v in zip(x, w.loc[t]):
            ax.text(xi + (i - 1) * 0.27, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)
    ax.axhline(0.196, color="#555", ls="--", label="Random guessing (0.20)")
    ax.set_xticks(x, list(kinds.values()))
    ax.set(ylabel="Worst run, test AUPRC", ylim=(0, 1.08),
           title="Worst single run at mean noise 0.2 · 500 fraud-rich rows · exact · uneven: 25 random devices")
    ax.legend(fontsize=10, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    fig.tight_layout()
    save(fig, "noise_types.png")


def per_transaction():
    s = np.load(RESULTS_DIR / "slides" / "scores_seed0.npz")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    for ax, t in zip(axes, ("baseline", "ncrc_4q_L4")):
        x, y, fraud = s[f"{t}_clean"], s[f"{t}_noisy"], s["y"] == 1
        ax.scatter(x[~fraud], y[~fraud], s=16, color="#B9B3C7", alpha=0.7, label="Legitimate")
        ax.scatter(x[fraud], y[fraud], s=60, marker="*", color=MISSED, label="Fraud")
        ax.set(title=f"{MODELS[t][0]} · rank agreement ρ = {spearmanr(x, y).statistic:.3f}",
               xlabel="Score without noise", ylabel="Score at p = 0.05")
    axes[0].legend(fontsize=10)
    fig.suptitle("Each dot is one test transaction (seed 0, 500 fraud-rich rows): a straight line keeps the order")
    fig.tight_layout()
    save(fig, "per_transaction.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    df = protocol()
    bars_auprc_f1(df), headline_bars(df), delta_and_rho(df), ablations(), schedule_and_gradients(), shots(), cost()
    sweeps(), noise_types(), per_transaction()
    if (RESULTS_DIR / "demo" / "seed0.npz").exists():
        demo()
    print("written:", sorted(p.name for p in OUT.glob("*.png")))
