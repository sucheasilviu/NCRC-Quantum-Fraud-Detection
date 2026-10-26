"""Does the result hold for other noise types, and for uneven noise? Every point is a real evaluation.

    python src/noise_types.py --model ncrc_4q_L4 --seed 0

Exact density-matrix scores (no shots) on the fraud-rich 500-row val/test sets of src/stress.py; threshold from
validation at the same noise. Noise kinds, each swept over the mean strength p in GRID:
  depolarizing, amplitude_damping, phase_damping, bit_flip: one channel, the same p after every gate on every wire
  hetero_depolarizing: depolarizing, but every noise location k gets its own p_k = 2 p u_k, u_k ~ U(0, 1)
  hetero_mixed: every location also gets its own channel type, drawn from the four above
Hetero kinds use DEVICES random "devices" per seed (the same u_k and types at every p), each kept separately.
"""
import argparse
import json

import numpy as np
import pennylane as qml
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

from common import (NATIVE_GATES, RESULTS_DIR, choose_threshold, evaluate, load_data, make_model, noisy_predict,
                    predict, prepare_seed, readout, train_vqc)
from models import MODELS
from stress import fraud_rich

GRID = np.round(np.linspace(0, 0.2, 9), 3)
CHANNELS = {"depolarizing": qml.DepolarizingChannel, "amplitude_damping": qml.AmplitudeDamping,
            "phase_damping": qml.PhaseDamping, "bit_flip": qml.BitFlip}
DEVICES = 5
MAX_LOCATIONS = 1000  # baseline has 126


@qml.transform
def add_noise(tape, specs):
    """After every gate, on every wire it touches, apply the next (channel, p) from specs."""
    ops, k = [], 0
    for op in tape.operations:
        ops.append(op)
        for w in op.wires:
            channel, p = specs[k]
            ops.append(channel(p, wires=w))
            k += 1
    return [tape.copy(operations=ops)], lambda res: res[0]


def noisy_scores(Z, w, circuit_fn, n_wires, specs):
    qnode = qml.QNode(circuit_fn, qml.device("default.mixed", wires=n_wires), interface="torch", diff_method="backprop")
    circuit = add_noise(qml.transforms.decompose(qnode, gate_set=NATIVE_GATES), specs=specs)
    with torch.no_grad():
        return readout(circuit(torch.tensor(Z), w)).numpy()


def noise_models(seed):
    """(kind, device, specs_for(p)) for every noise kind; hetero devices are fixed per seed."""
    out = [(kind, 0, lambda p, c=c: [(c, p)] * MAX_LOCATIONS) for kind, c in CHANNELS.items()]
    for d in range(DEVICES):
        rng = np.random.default_rng(20_000 + 100 * seed + d)
        u = rng.uniform(size=MAX_LOCATIONS)
        types = rng.choice(list(CHANNELS.values()), size=MAX_LOCATIONS)
        out.append(("hetero_depolarizing", d, lambda p, u=u: [(qml.DepolarizingChannel, 2 * p * x) for x in u]))
        out.append(("hetero_mixed", d, lambda p, u=u, t=types: [(c, 2 * p * x) for c, x in zip(t, u)]))
    return out


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
    rows = []
    if a.model == "logreg":  # classical: no gates, so the same scores under every noise
        ref = LogisticRegression(max_iter=1000).fit(d["Z_bal"], d["y_bal"])
        r = evaluate(yt, ref.predict_proba(Zt)[:, 1], choose_threshold(yv, ref.predict_proba(Zv)[:, 1]))
        rows = [{"kind": k, "device": 0, "p": float(p), **r, "Spearman rho": 1.0} for k in CHANNELS for p in GRID]
    else:
        M = MODELS[a.model]
        model = make_model(M["circuit"], M["n_wires"])
        w, _, _ = train_vqc(model, M["init"], d["Z_bal"], d["y_bal"], d["Z_val"], d["y_val"], a.seed)
        clean_v, clean_t = predict(model, Zv, w), predict(model, Zt, w)
        # self-check: uniform depolarizing through add_noise matches the protocol's qml.noise.insert
        ours = noisy_scores(Zt[:20], w, M["circuit"], M["n_wires"], [(qml.DepolarizingChannel, 0.05)] * MAX_LOCATIONS)
        assert np.allclose(ours, noisy_predict(Zt[:20], w, 0.05, M["circuit"], M["n_wires"]), atol=1e-9)
        for kind, dev, specs_for in noise_models(a.seed):
            for p in GRID:
                sv, st = (clean_v, clean_t) if p == 0 else (
                    noisy_scores(Z, w, M["circuit"], M["n_wires"], specs_for(float(p))) for Z in (Zv, Zt))
                rows.append({"kind": kind, "device": dev, "p": float(p), **evaluate(yt, st, choose_threshold(yv, sv)),
                             "Spearman rho": spearmanr(clean_t, st).statistic})
            print(f"{kind} dev{dev}: AUPRC " + " ".join(f"{x['AUPRC']:.3f}" for x in rows[-len(GRID):]), flush=True)
    path = RESULTS_DIR / "noise_types" / f"{a.model}_seed{a.seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": a.model, "seed": a.seed, "rows": rows}, indent=1, default=float))


if __name__ == "__main__":
    main()
