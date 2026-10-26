"""Seed-0 per-transaction scores for the explanatory figures (PR curves, confusion matrices, threshold plots,
input-response slices). Same training and protocol as src/run.py; asserts the AUPRC matches results/raw.

    python src/demo_seed0.py     -> results/demo/seed0.npz
"""
import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from common import (RESULTS_DIR, choose_threshold, load_data, make_model, noisy_predict, predict, prepare_seed,
                    train_vqc)
from models import MODELS

SEED, LEVELS, SLICE_P = 0, [0.0, 0.01, 0.05], 0.05
TAGS = ["baseline", "ncrc_4q_L4_cnot", "ncrc_4q_L4"]


def main():
    torch.set_num_threads(4)
    d = prepare_seed(load_data(), SEED)
    axis = np.linspace(0, np.pi, 31)
    grid = np.tile(np.median(d["Z_bal"], axis=0), (axis.size ** 2, 1))
    grid[:, 0], grid[:, 1] = np.repeat(axis, axis.size), np.tile(axis, axis.size)  # row = feature 1, col = feature 2
    out = {"y_test": d["y_test"], "y_val": d["y_val"], "levels": np.array(LEVELS), "axis": axis}
    lr = LogisticRegression(max_iter=1000).fit(d["Z_bal"], d["y_bal"])
    out["logreg_val"], out["logreg_test"] = lr.predict_proba(d["Z_val"])[:, 1], lr.predict_proba(d["Z_test"])[:, 1]
    out["logreg_grid"] = lr.predict_proba(grid)[:, 1]
    for tag in TAGS:
        M = MODELS[tag]
        model = make_model(M["circuit"], M["n_wires"])
        w, _, _ = train_vqc(model, M["init"], d["Z_bal"], d["y_bal"], d["Z_val"], d["y_val"], SEED)
        raw = {r["p"]: r for r in json.loads((RESULTS_DIR / "raw" / f"{tag}_seed{SEED}.json").read_text())["protocol"]}
        for p in LEVELS:
            sv, st = ((predict(model, Z, w) if p == 0 else noisy_predict(Z, w, p, M["circuit"], M["n_wires"]))
                      for Z in (d["Z_val"], d["Z_test"]))
            assert abs(average_precision_score(d["y_test"], st) - raw[p]["AUPRC"]) < 1e-6, (tag, p)
            out[f"{tag}_val_{p}"], out[f"{tag}_test_{p}"] = sv, st
            print(tag, p, f"AUPRC {average_precision_score(d['y_test'], st):.4f} threshold {choose_threshold(d['y_val'], sv):.4f}", flush=True)
        out[f"{tag}_grid"] = noisy_predict(grid, w, SLICE_P, M["circuit"], M["n_wires"])
    (RESULTS_DIR / "demo").mkdir(exist_ok=True)
    np.savez_compressed(RESULTS_DIR / "demo" / "seed0.npz", **out)


if __name__ == "__main__":
    main()
