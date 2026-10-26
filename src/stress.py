"""High-noise stress test on a fraud-rich 500-row evaluation set, with a realistic 1,024-shot readout.

Beyond the official protocol (the headline tables stay exact-simulation, full test set). Here:
- evaluation set = all 98 test frauds + 402 random test legit (500 rows, ~20% fraud), like the fraud-rich
  Task 1 sample but drawn from unseen data; threshold from the matching 500-row validation set (99 frauds)
- gate noise up to p = 0.5, every score estimated from SHOTS_PER_EXECUTION shots, REPEATS shot draws
- models trained exactly as in the protocol (noiseless, 30 epochs); logistic regression is the classical reference

    python src/stress.py --model ncrc_4q_L4 --seed 0
"""
import argparse
import json

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from common import (RESULTS_DIR, SHOTS_PER_EXECUTION, choose_threshold, evaluate, load_data, make_model,
                    noisy_predict, predict, prepare_seed, train_vqc)
from models import MODELS

LEVELS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
N_EVAL = 500
REPEATS = 5


def fraud_rich(y, seed):
    """All frauds plus random legit rows, N_EVAL in total (fixed per seed)."""
    rng = np.random.default_rng(10_000 + seed)
    fraud = np.flatnonzero(y == 1)
    legit = rng.choice(np.flatnonzero(y == 0), size=N_EVAL - len(fraud), replace=False)
    return np.sort(np.concatenate([fraud, legit]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=[*MODELS, "logreg"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    d = prepare_seed(load_data(), a.seed)
    iv, it = fraud_rich(d["y_val"], a.seed), fraud_rich(d["y_test"], a.seed)
    Zv, yv, Zt, yt = d["Z_val"][iv], d["y_val"][iv], d["Z_test"][it], d["y_test"][it]
    rng = np.random.default_rng(1000 + a.seed)
    rows = []
    if a.model == "logreg":  # classical: no gates, so gate noise and shot noise do not apply
        ref = LogisticRegression(max_iter=1000).fit(d["Z_bal"], d["y_bal"])
        sv, st = ref.predict_proba(Zv)[:, 1], ref.predict_proba(Zt)[:, 1]
        r = evaluate(yt, st, choose_threshold(yv, sv))
        rows = [{"p": p, "repeat": 0, **r, "Spearman rho": 1.0} for p in LEVELS]
    else:
        M = MODELS[a.model]
        model = make_model(M["circuit"], M["n_wires"])
        w, _, _ = train_vqc(model, M["init"], d["Z_bal"], d["y_bal"], d["Z_val"], d["y_val"], a.seed)
        clean_t = predict(model, Zt, w)
        for p in LEVELS:
            ev = predict(model, Zv, w) if p == 0 else noisy_predict(Zv, w, p, M["circuit"], M["n_wires"])
            et = clean_t if p == 0 else noisy_predict(Zt, w, p, M["circuit"], M["n_wires"])
            for r in range(REPEATS):
                sv = rng.binomial(SHOTS_PER_EXECUTION, np.clip(ev, 0, 1)) / SHOTS_PER_EXECUTION
                st = rng.binomial(SHOTS_PER_EXECUTION, np.clip(et, 0, 1)) / SHOTS_PER_EXECUTION
                rows.append({"p": p, "repeat": r, **evaluate(yt, st, choose_threshold(yv, sv)),
                             "Spearman rho": spearmanr(clean_t, st).statistic,
                             "AUPRC exact": average_precision_score(yt, et)})
    path = RESULTS_DIR / "stress" / f"{a.model}_seed{a.seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": a.model, "seed": a.seed, "n_eval": N_EVAL, "fraud_rate": float(yt.mean()),
                                "shots": SHOTS_PER_EXECUTION, "rows": rows}, indent=1, default=float))
    print(path.name, [(r["p"], round(r["AUPRC"], 3)) for r in rows if r["repeat"] == 0])


if __name__ == "__main__":
    main()
