"""Task 2 circuits. Every entry of MODELS gives (circuit_fn, init, n_wires, n_features).

Main method: the Noise-Covariant Re-uploading Classifier (NCRC).
A single-qubit depolarizing channel commutes with every single-qubit unitary, D(U rho U^+) = U D(rho) U^+.
In a circuit with NO entangling gates every channel can therefore be pushed to the end of its wire, and the
whole noise model collapses to one Bloch-vector shrink (1 - 4p/3)^k per wire, k = gates on that wire.
If every wire carries the same number of gates and the readout is the mean of the Z_i, every score is mapped
by the SAME affine function at a given p: the ranking (AUPRC, AUROC, Spearman rho = 1) and, after the
protocol's threshold recalibration, the threshold metrics are exactly those of the noiseless circuit.
Expressivity comes from data re-uploading (Perez-Salinas et al. 2020) instead of entanglement: each layer is
one native Rot gate whose three angles are a trainable affine function of the encoded features.
"""
import numpy as np
import pennylane as qml
import torch

from common import N_QUBITS, init_weights, vqc


def ncrc_circuit(n_wires, layers, n_features, entangle=False, prune=False, measure_wires=False, measure_probs=False):
    def circuit(x, weights):
        # weights: (n_wires, layers, 3, n_features + 1) -> Rot angles = W x + b
        for layer in range(layers):
            for q in range(n_wires):
                w = weights[q, layer]
                angles = x @ w[:, :n_features].T + w[:, n_features]
                phi, theta, omega = angles[..., 0], angles[..., 1], angles[..., 2]
                # prune: Rot = RZ(omega) RY(theta) RZ(phi). The first RZ acts on |0> (a global phase) and the last
                # commutes with the Z readout, so fixing them to 0 keeps the same function class but saves 2
                # parameter-shift evaluations per wire. Only valid without entanglement.
                if prune and layer == 0:
                    phi = torch.zeros_like(theta)  # constant, so parameter-shift skips it
                if prune and layer == layers - 1:
                    omega = torch.zeros_like(theta)
                qml.Rot(phi, theta, omega, wires=q)
            # ablation only: a CNOT ring breaks noise covariance
            if entangle and n_wires > 1 and layer < layers - 1:
                for q in range(n_wires - 1):
                    qml.CNOT(wires=[q, q + 1])
        if measure_wires:
            return tuple(qml.expval(qml.PauliZ(q)) for q in range(n_wires))
        if measure_probs:  # all 2^n bitstrings: shots for entangled wires must be sampled jointly
            return qml.probs(wires=range(n_wires))
        if n_wires == 1:
            return qml.expval(qml.PauliZ(0))
        return qml.expval(qml.Hamiltonian([1.0 / n_wires] * n_wires, [qml.PauliZ(q) for q in range(n_wires)]))

    def init(seed):
        g = torch.Generator().manual_seed(seed)
        w = torch.empty(n_wires, layers, 3, n_features + 1, dtype=torch.float64)
        w[..., :n_features] = 0.5 * torch.randn(n_wires, layers, 3, n_features, generator=g, dtype=torch.float64)
        w[..., n_features] = 2 * np.pi * torch.rand(n_wires, layers, 3, generator=g, dtype=torch.float64)
        return torch.nn.Parameter(w)

    return circuit, init


def _ncrc(n_wires, layers, n_features=N_QUBITS, entangle=False, prune=False):
    circuit, init = ncrc_circuit(n_wires, layers, n_features, entangle, prune)
    wire_circuit = probs_circuit = None
    if not entangle:
        wire_circuit, _ = ncrc_circuit(n_wires, layers, n_features, entangle, prune, measure_wires=True)
    elif n_wires > 1:
        probs_circuit, _ = ncrc_circuit(n_wires, layers, n_features, entangle, prune, measure_probs=True)
    return {"circuit": circuit, "wire_circuit": wire_circuit, "probs_circuit": probs_circuit, "init": init, "n_wires": n_wires, "n_features": n_features,
            "architecture": "ncrc", "layers": layers, "entangle": entangle, "prune_local_z": prune}


MODELS = {
    "baseline": {"circuit": vqc, "init": init_weights, "n_wires": N_QUBITS, "n_features": N_QUBITS},
    "ncrc_1q_L2": _ncrc(1, 2),
    "ncrc_1q_L4": _ncrc(1, 4),
    "ncrc_1q_L6": _ncrc(1, 6),
    "ncrc_1q_L2_pruned": _ncrc(1, 2, prune=True),
    "ncrc_1q_L4_pruned": _ncrc(1, 4, prune=True),
    "ncrc_4q_L2": _ncrc(4, 2),
    "ncrc_4q_L4": {**_ncrc(4, 4, prune=True), "default_gradient_backend": "combined"},
    "ncrc_4q_L4_pairwise05": {**_ncrc(4, 4), "default_pairwise_weight": 0.5},
    "ncrc_4q_L4_cnot": _ncrc(4, 4, entangle=True),
}
