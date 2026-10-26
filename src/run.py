"""Run one model for one seed through the full Task 2 protocol and save results/raw/<tag>_seed<s>.json.

    python src/run.py --model baseline --seed 0
    python src/run.py --model ncrc_1q_L2 --seed 0 --val-fraction 0.2 --epochs 15 --tag ncrc_1q_L2_cheap
    python src/run.py --model logreg --seed 0
"""
import argparse
import json
import time

import pennylane as qml
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from common import (EPOCHS, RESULTS_DIR, SHOTS_PER_EXECUTION, PackedNCRCGradient, active_executions_per_sample, choose_threshold,
                    evaluate, gate_counts, load_data, make_model, noise_locations, prepare_seed, train_vqc,
                    evaluate_under_noise)
from models import MODELS

# robustness check beyond the protocol: amplitude damping (T1 decay) is NOT unitarily covariant
CHANNELS = {"depolarizing": qml.DepolarizingChannel, "amplitude_damping": qml.AmplitudeDamping}


def run_logreg(d):
    ref = LogisticRegression(max_iter=1000).fit(d["Z_bal"], d["y_bal"])
    val_s, test_s = ref.predict_proba(d["Z_val"])[:, 1], ref.predict_proba(d["Z_test"])[:, 1]
    # classical model: no noise, the same row is reported at every p
    return {"protocol": [{"p": 0.0, **evaluate(d["y_test"], test_s, choose_threshold(d["y_val"], val_s))}]}


def run_vqc(name, d, seed, epochs, val_fraction, validate_every, channel=qml.DepolarizingChannel,
            gradient_mode=None, pairwise_weight=0.0):
    M = MODELS[name]
    Z_tv, y_tv = d["Z_val"], d["y_val"]
    if val_fraction < 1.0:  # stratified validation subset used only for epoch selection during training
        Z_tv, _, y_tv, _ = train_test_split(Z_tv, y_tv, train_size=val_fraction, stratify=y_tv, random_state=seed)
    model = make_model(M["circuit"], M["n_wires"])
    gradient_backend = None
    if gradient_mode:
        if (M.get("architecture") != "ncrc" or M.get("entangle") or M["n_wires"] != 4
                or M["n_features"] != 4):
            raise ValueError("packed gradient modes require an independent four-wire, four-feature NCRC")
        gradient_backend = PackedNCRCGradient(
            M["n_wires"], M["layers"], M.get("prune_local_z", False), gradient_mode
        )
    t0 = time.time()
    weights, history, n_val = train_vqc(model, M["init"], d["Z_bal"], d["y_bal"], Z_tv, y_tv, seed,
                                        epochs=epochs, validate_every=validate_every,
                                        gradient_backend=gradient_backend, pairwise_weight=pairwise_weight)
    train_s = time.time() - t0
    protocol = evaluate_under_noise(M["circuit"], weights, d["Z_val"], d["Z_test"], d["y_val"], d["y_test"], n_wires=M["n_wires"], channel=channel)
    x0 = torch.tensor(d["Z_bal"][0])
    per_sample = (1 + 2 * gradient_backend.n_slots) if gradient_backend else active_executions_per_sample(
        M["circuit"], weights, d["Z_bal"][0], M["n_wires"]
    )
    train_exec = epochs * len(d["Z_bal"]) * per_sample
    val_exec = n_val * len(Z_tv)
    return {
        "protocol": protocol.to_dict("records"),
        "history": history.to_dict("records"),
        "resources": {
            "n_wires": M["n_wires"],
            "noise_locations": noise_locations(M["circuit"], x0, weights, n_wires=M["n_wires"]),
            "gate_counts": gate_counts(M["circuit"], x0, weights, n_wires=M["n_wires"]),
            "trainable_params": weights.numel(),
            "gradient_backend": gradient_mode or "pennylane-backprop",
            "gradient_coordinates": gradient_backend.n_slots if gradient_backend else None,
            "gradient_shift_groups": len(gradient_backend.shift_groups) if gradient_backend else None,
        },
        "ledger": {
            "executions_per_sample": per_sample, "epochs": epochs, "n_train": len(d["Z_bal"]),
            "val_rows": len(Z_tv), "val_passes": n_val,
            "train_executions": train_exec, "val_executions": val_exec,
            "total_executions": train_exec + val_exec,
            "total_shots": (train_exec + val_exec) * SHOTS_PER_EXECUTION,
        },
        "best_val_auprc_train_subset": max(h["validation AUPRC"] for h in history.to_dict("records") if h["validation AUPRC"] is not None),
        "train_seconds": train_s,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=[*MODELS, "logreg"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--val-fraction", type=float, default=1.0)
    ap.add_argument("--validate-every", type=int, default=1)
    ap.add_argument("--pairwise-weight", type=float, default=None,
                    help="Add this weight of within-minibatch pairwise logistic ranking loss to BCE")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--channel", choices=list(CHANNELS), default="depolarizing", help="noise channel after every gate (protocol: depolarizing)")
    ap.add_argument("--gradient-backend", choices=["standard", "euler", "parallel", "combined"], default=None,
                    help="NCRC shifts: Euler reduction (57), parallel wires (21), or both (15 executions/sample); "
                         "standard = plain parameter shift (overrides a model default)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    d = prepare_seed(load_data(), a.seed)
    pairwise_weight = a.pairwise_weight if a.pairwise_weight is not None else MODELS.get(a.model, {}).get("default_pairwise_weight", 0.0)
    if pairwise_weight < 0:
        ap.error("--pairwise-weight must be nonnegative")
    gradient_backend = None if a.gradient_backend == "standard" else (
        a.gradient_backend or MODELS.get(a.model, {}).get("default_gradient_backend"))
    if pairwise_weight and gradient_backend:
        ap.error("packed gradients (explicit or the model default) do not support the pairwise term yet; pass --gradient-backend standard")  # shortcut: packed backward is BCE-only
    out = run_logreg(d) if a.model == "logreg" else run_vqc(
        a.model, d, a.seed, a.epochs, a.val_fraction, a.validate_every, CHANNELS[a.channel],
        gradient_mode=gradient_backend,
        pairwise_weight=pairwise_weight,
    )
    effective_args = {**vars(a), "pairwise_weight": pairwise_weight, "gradient_backend": gradient_backend}
    out.update({"model": a.model, "tag": a.tag or a.model, "seed": a.seed, "args": effective_args})
    path = RESULTS_DIR / "raw" / f"{out['tag']}_seed{a.seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"{path.name}: " + ", ".join(f"p={r['p']}: AUPRC {r['AUPRC']:.3f}" for r in out["protocol"]))


if __name__ == "__main__":
    main()
