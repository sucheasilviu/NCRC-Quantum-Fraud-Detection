"""Dense noise sweep: every point is a real evaluation (no interpolation or smoothing).

    python src/sweep.py --mode protocol --model baseline --seed 0   # 21 levels p in [0, 0.05], full test set, exact
    python src/sweep.py --mode stress   --model ncrc_4q_L4 --seed 0 # 26 levels p in [0, 0.5], 500-row set, 1,024 shots

protocol mode follows evaluate_under_noise exactly (threshold from validation at the same p);
stress mode uses the fraud-rich 500-row sets of src/stress.py with a 1,024-shot readout and REPEATS draws, sampled as in
src/shots.py: one draw for a single readout, each wire separately for independent NCRC wires, whole bitstrings for
entangled NCRC.
Each model is trained once per seed exactly as in the protocol.
"""
import argparse
import json

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

from common import (RESULTS_DIR, SHOTS_PER_EXECUTION, choose_threshold, evaluate, load_data, make_model,
                    noisy_predict, predict, prepare_seed, train_vqc)
from models import MODELS
from shots import joint_shots, predict_probs, predict_wires
from stress import REPEATS, fraud_rich

GRIDS = {"protocol": np.round(np.linspace(0, 0.05, 21), 4), "stress": np.round(np.linspace(0, 0.5, 26), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=list(GRIDS))
    ap.add_argument("--model", required=True, choices=[*MODELS, "logreg"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    d = prepare_seed(load_data(), a.seed)
    Zv, yv, Zt, yt = d["Z_val"], d["y_val"], d["Z_test"], d["y_test"]
    if a.mode == "stress":
        iv, it = fraud_rich(yv, a.seed), fraud_rich(yt, a.seed)
        Zv, yv, Zt, yt = Zv[iv], yv[iv], Zt[it], yt[it]
    grid = [float(p) for p in GRIDS[a.mode]]
    rows, per_wire, joint = [], False, False
    if a.model == "logreg":  # classical: no gates, so the same scores at every p
        ref = LogisticRegression(max_iter=1000).fit(d["Z_bal"], d["y_bal"])
        r = evaluate(yt, ref.predict_proba(Zt)[:, 1], choose_threshold(yv, ref.predict_proba(Zv)[:, 1]))
        rows = [{"p": p, "repeat": 0, **r, "Spearman rho": 1.0} for p in grid]
    else:
        M = MODELS[a.model]
        model = make_model(M["circuit"], M["n_wires"])
        w, _, _ = train_vqc(model, M["init"], d["Z_bal"], d["y_bal"], d["Z_val"], d["y_val"], a.seed)
        clean_t = predict(model, Zt, w)
        rng = np.random.default_rng(1000 + a.seed)
        per_wire = a.mode == "stress" and M.get("wire_circuit") is not None and M["n_wires"] > 1
        joint = a.mode == "stress" and M.get("probs_circuit") is not None

        def shots(z):  # z: per-wire <Z_q> (rows, wires) -> score from each wire's own 1,024 outcomes
            ones = rng.binomial(SHOTS_PER_EXECUTION, np.clip((1 - z) / 2, 0, 1))
            return (1 - (1 - 2 * ones / SHOTS_PER_EXECUTION).mean(axis=1)) / 2

        for p in grid:
            if joint:
                Pv, Pt = (predict_probs(M["probs_circuit"], Z, w, M["n_wires"], p) for Z in (Zv, Zt))
                ones = np.array([bin(b).count("1") for b in range(Pv.shape[1])]) / M["n_wires"]
                ev, et = Pv @ ones, Pt @ ones
            elif per_wire:
                zv, zt = (predict_wires(M["wire_circuit"], Z, w, M["n_wires"], p) for Z in (Zv, Zt))
                ev, et = (1 - zv.mean(axis=1)) / 2, (1 - zt.mean(axis=1)) / 2
            else:
                ev = predict(model, Zv, w) if p == 0 else noisy_predict(Zv, w, p, M["circuit"], M["n_wires"])
                et = clean_t if p == 0 else noisy_predict(Zt, w, p, M["circuit"], M["n_wires"])
            if a.mode == "protocol":
                draws = [(ev, et)]
            elif joint:
                draws = [(joint_shots(Pv, rng), joint_shots(Pt, rng)) for _ in range(REPEATS)]
            elif per_wire:
                draws = [(shots(zv), shots(zt)) for _ in range(REPEATS)]
            else:
                draws = [(rng.binomial(SHOTS_PER_EXECUTION, np.clip(ev, 0, 1)) / SHOTS_PER_EXECUTION,
                          rng.binomial(SHOTS_PER_EXECUTION, np.clip(et, 0, 1)) / SHOTS_PER_EXECUTION) for _ in range(REPEATS)]
            for r, (sv, st) in enumerate(draws):
                rows.append({"p": p, "repeat": r, **evaluate(yt, st, choose_threshold(yv, sv)),
                             "Spearman rho": spearmanr(clean_t, st).statistic})
            print(f"p={p}: AUPRC {np.mean([x['AUPRC'] for x in rows if x['p'] == p]):.4f}", flush=True)
    path = RESULTS_DIR / "sweep" / a.mode / f"{a.model}_seed{a.seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mode": a.mode, "model": a.model, "seed": a.seed,
                                "shot_readout": "joint" if joint else "per_wire" if per_wire else "single", "rows": rows}, indent=1, default=float))


if __name__ == "__main__":
    main()
