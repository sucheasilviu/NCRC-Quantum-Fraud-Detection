"""Finite-shot evaluation: every score re-estimated from SHOTS_PER_EXECUTION measurements.

Single-readout models: one Binomial draw. Independent NCRC wires: each wire sampled separately.
Entangled multi-wire NCRC: whole bitstrings sampled jointly (multinomial over all 2^n outcomes), since its wires are correlated.
"""
import argparse
import json

import numpy as np
import pennylane as qml
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from common import (RESULTS_DIR, SHOTS_PER_EXECUTION, load_data, make_model, noisy_predict, noisy_qnode,
                    predict, prepare_seed, train_vqc)
from models import MODELS

LEVELS = [0.0, 0.001, 0.01, 0.05]
REPEATS = 5


def predict_wires(circuit, Z, weights, n_wires, p=0.0, chunk=2000):
    """Per-wire <Z_q>, simulated with the protocol's noise model at p (exact density matrices)."""
    qnode = noisy_qnode(circuit, p, n_wires) if p else qml.QNode(circuit, qml.device("default.qubit", wires=n_wires), interface="torch")
    X = torch.tensor(Z)
    chunks = []
    with torch.no_grad():
        for k in range(0, len(X), chunk):
            out = qnode(X[k:k + chunk], weights)
            chunks.append(torch.stack(tuple(out) if isinstance(out, (tuple, list)) else (out,), dim=-1))
    return torch.cat(chunks).numpy()


def predict_probs(circuit, Z, weights, n_wires, p=0.0, chunk=2000):
    """Probabilities of all 2^n bitstrings, simulated with the protocol's noise model at p."""
    qnode = noisy_qnode(circuit, p, n_wires) if p else qml.QNode(circuit, qml.device("default.qubit", wires=n_wires), interface="torch")
    X = torch.tensor(Z)
    with torch.no_grad():
        return torch.cat([qnode(X[k:k + chunk], weights) for k in range(0, len(X), chunk)]).numpy()


def joint_shots(P, rng):
    """Score from SHOTS_PER_EXECUTION joint bitstrings: mean over shots of (number of 1s / n_wires)."""
    n = int(np.log2(P.shape[1]))
    ones = np.array([bin(b).count("1") for b in range(P.shape[1])]) / n
    P = np.clip(P, 0, None)
    return rng.multinomial(SHOTS_PER_EXECUTION, P / P.sum(axis=1, keepdims=True)) @ ones / SHOTS_PER_EXECUTION


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--pairwise-weight", type=float, default=None)
    a = ap.parse_args()
    M = MODELS[a.model]
    pairwise_weight = a.pairwise_weight if a.pairwise_weight is not None else M.get("default_pairwise_weight", 0.0)
    torch.set_num_threads(a.threads)
    d = prepare_seed(load_data(), a.seed)
    model = make_model(M["circuit"], M["n_wires"])
    weights, _, _ = train_vqc(model, M["init"], d["Z_bal"], d["y_bal"], d["Z_val"], d["y_val"], a.seed,
                              pairwise_weight=pairwise_weight)
    clean = predict(model, d["Z_test"], weights)
    rng = np.random.default_rng(1000 + a.seed)
    rows = []
    clean_z = (predict_wires(M["wire_circuit"], d["Z_test"], weights, M["n_wires"])
               if M.get("wire_circuit") else None)
    depths = [M["layers"]] * M["n_wires"] if clean_z is not None else None
    for p in LEVELS:
        exact = clean if p == 0.0 else noisy_predict(d["Z_test"], weights, p, M["circuit"], M["n_wires"])
        z = P = None
        if M.get("probs_circuit"):
            P = predict_probs(M["probs_circuit"], d["Z_test"], weights, M["n_wires"], p)
            n = M["n_wires"]
            # self-check: the bitstring distribution reproduces the protocol's score exactly
            assert np.allclose(P @ (np.array([bin(b).count("1") for b in range(2 ** n)]) / n), exact, atol=1e-9)
        if clean_z is not None:
            z = clean_z if p == 0.0 else predict_wires(M["wire_circuit"], d["Z_test"], weights, M["n_wires"], p)
            # self-check: the simulated shrink equals the covariance prediction (1 - 4p/3)^layers on every wire
            assert np.allclose(z, clean_z * np.asarray([(1 - 4 * p / 3) ** depth for depth in depths]), atol=1e-9)
        for repeat in range(REPEATS):
            if P is not None:
                estimated = joint_shots(P, rng)
            elif z is None:
                estimated = rng.binomial(SHOTS_PER_EXECUTION, np.clip(exact, 0, 1)) / SHOTS_PER_EXECUTION
            else:
                ones = rng.binomial(SHOTS_PER_EXECUTION, np.clip((1 - z) / 2, 0, 1))
                z_hat = 1 - 2 * ones / SHOTS_PER_EXECUTION
                estimated = (1 - z_hat.mean(axis=1)) / 2
            rows.append({"p": p, "repeat": repeat,
                         "AUPRC exact": average_precision_score(d["y_test"], exact),
                         "AUPRC shots": average_precision_score(d["y_test"], estimated),
                         "Spearman rho shots vs clean": spearmanr(clean, estimated).statistic})
    path = RESULTS_DIR / "shots" / f"{a.model}_seed{a.seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": a.model, "seed": a.seed, "shots": SHOTS_PER_EXECUTION,
                                "shot_method": "joint_bitstrings" if M.get("probs_circuit") else (
                                    "independent_per_wire_simulated" if M.get("wire_circuit") else "single_readout"),
                                "rows": rows}, indent=1, default=float))
    print(path.name, [(row["p"], round(row["AUPRC shots"], 3)) for row in rows if row["repeat"] == 0])


if __name__ == "__main__":
    main()
