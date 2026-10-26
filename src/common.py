"""Shared Task 2 pipeline: data + common split, encoder, baseline VQC, training, noise model, evaluation protocol.

Ported from the Task 1 notebook (task1_guided_vqc_fraud_detection.ipynb, Parts 2-11) so that the
baseline and the protocol are identical to the reference; only file layout changed.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "creditcard.csv"
SPLIT_PATH = ROOT / "split_indices.npz"
RESULTS_DIR = ROOT / "results"

SPLIT_SEED = 42
N_QUBITS = 4
FEATURE_MAP_REPS = 2
ANSATZ_REPS = 3
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.05
SHOTS_PER_EXECUTION = 1024
NOISE_LEVELS = [0.001, 0.01, 0.05]
NATIVE_GATES = {"RX", "RY", "RZ", "Rot", "Hadamard", "CNOT", "CZ"}
SEEDS = [0, 1, 2, 3, 4]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------- data

def load_data():
    """Dataset and the common split. Creates split_indices.npz exactly as notebook Part 4.2 / 11.1 if missing."""
    df = pd.read_csv(DATA_PATH)
    df["Class"] = df["Class"].astype(int)
    X_all, y_all = df.drop(columns="Class"), df["Class"].to_numpy()
    if not SPLIT_PATH.exists():
        X_trval, X_te, y_trval, _ = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=SPLIT_SEED)
        X_tr, X_va, _, _ = train_test_split(X_trval, y_trval, test_size=0.25, stratify=y_trval, random_state=SPLIT_SEED)
        np.savez(SPLIT_PATH, train=X_tr.index.to_numpy(), val=X_va.index.to_numpy(), test=X_te.index.to_numpy())
    split = np.load(SPLIT_PATH)
    parts = {name: (X_all.loc[split[name]], y_all[split[name]]) for name in ["train", "val", "test"]}
    # official counts from the notebook's Check 4.1
    assert (len(parts["test"][1]), int(parts["test"][1].sum()), int(parts["val"][1].sum())) == (56962, 98, 99), "split differs from the common split"
    return parts


def undersample(X_part, y_part, seed):
    rng = np.random.default_rng(seed)
    fraud = np.flatnonzero(y_part == 1)
    legit = rng.choice(np.flatnonzero(y_part == 0), size=len(fraud), replace=False)
    keep = rng.permutation(np.concatenate([fraud, legit]))
    return X_part.iloc[keep], y_part[keep]


class AngleEncoder:
    """StandardScaler(Time, Amount) -> PCA -> MinMax to [0, pi]; fitted on the balanced training set only."""
    raw_columns = ["Time", "Amount"]

    def __init__(self, n_components, seed):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=seed)
        self.angles = MinMaxScaler(feature_range=(0, np.pi))

    def _scaled(self, frame):
        out = frame.copy()
        out[self.raw_columns] = self.scaler.transform(frame[self.raw_columns])
        return out.to_numpy()

    def fit(self, frame):
        self.scaler.fit(frame[self.raw_columns])
        self.angles.fit(self.pca.fit_transform(self._scaled(frame)))
        return self

    def transform(self, frame):
        return np.clip(self.angles.transform(self.pca.transform(self._scaled(frame))), 0, np.pi)


def prepare_seed(parts, seed, n_components=N_QUBITS):
    """Per-seed balanced training set and encoded angles (fit on training data only)."""
    X_train, y_train = parts["train"]
    X_bal, y_bal = undersample(X_train, y_train, seed)
    enc = AngleEncoder(n_components, seed).fit(X_bal)
    return {
        "Z_bal": enc.transform(X_bal), "y_bal": y_bal,
        "Z_val": enc.transform(parts["val"][0]), "y_val": parts["val"][1],
        "Z_test": enc.transform(parts["test"][0]), "y_test": parts["test"][1],
    }


# ---------------------------------------------------------------- baseline circuit (Task 1)

def feature_map(x):
    for _ in range(FEATURE_MAP_REPS):
        for i in range(N_QUBITS):
            qml.Hadamard(wires=i)
            qml.RZ(2 * x[..., i], wires=i)
        for i in range(N_QUBITS):
            for j in range(i + 1, N_QUBITS):
                qml.CNOT(wires=[i, j])
                qml.RZ(2 * (np.pi - x[..., i]) * (np.pi - x[..., j]), wires=j)
                qml.CNOT(wires=[i, j])


def ansatz(weights):
    for layer in range(ANSATZ_REPS + 1):
        for i in range(N_QUBITS):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)
        if layer < ANSATZ_REPS:
            for i in range(N_QUBITS - 1):
                qml.CNOT(wires=[i, i + 1])


def vqc(x, weights):
    feature_map(x)
    ansatz(weights)
    return qml.expval(qml.PauliZ(0))


def init_weights(seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.nn.Parameter(2 * np.pi * torch.rand(ANSATZ_REPS + 1, N_QUBITS, 2, generator=generator, dtype=torch.float64))


def readout(expval):
    return (1 - expval) / 2


def make_model(circuit_fn, n_wires=N_QUBITS):
    """Noiseless, backprop-differentiable model x, w -> fraud probability."""
    qnode = qml.QNode(circuit_fn, qml.device("default.qubit", wires=n_wires), interface="torch", diff_method="backprop")
    return lambda x, w: readout(qnode(x, w))


# ---------------------------------------------------------------- training

def predict(model, Z, weights, chunk=20000):
    X_t = torch.tensor(Z)
    with torch.no_grad():
        return torch.cat([model(X_t[k:k + chunk], weights) for k in range(0, len(X_t), chunk)]).numpy()


def train_vqc(model, init, Z_train, y_train, Z_valid, y_valid, seed, epochs=EPOCHS, lr=LEARNING_RATE,
              validate_every=1, gradient_backend=None, pairwise_weight=0.0):
    """Adam on BCE (+ optional pairwise ranking term), optional packed parameter-shift gradients;
    keeps the weights with the best validation AUPRC (notebook Part 7).
    validate_every > 1 scores validation less often (cheaper ledger); the last epoch is always scored."""
    set_seed(seed)
    weights = init(seed)
    optimizer = torch.optim.Adam([weights], lr=lr)
    loss_fn = torch.nn.BCELoss()
    X_t, y_t = torch.tensor(Z_train), torch.tensor(y_train, dtype=torch.float64)
    history, best_ap, best_weights, n_val = [], -1.0, None, 0
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(X_t))
        losses = []
        for k in range(0, len(X_t), BATCH_SIZE):
            idx = order[k:k + BATCH_SIZE]
            optimizer.zero_grad()
            if gradient_backend is None:
                scores = model(X_t[idx], weights).clamp(1e-7, 1 - 1e-7)
                labels = y_t[idx]
                loss = loss_fn(scores, labels)
                if pairwise_weight:
                    positive, negative = scores[labels == 1], scores[labels == 0]
                    if positive.numel() and negative.numel():
                        loss = loss + pairwise_weight * torch.nn.functional.softplus(
                            -(positive[:, None] - negative[None, :])
                        ).mean()
                loss.backward()
            else:
                loss = gradient_backend.backward(model, X_t[idx], y_t[idx], weights, loss_fn)
            optimizer.step()
            losses.append(loss.item())
        val_ap = None
        if epoch % validate_every == 0 or epoch == epochs:
            val_ap = average_precision_score(y_valid, predict(model, Z_valid, weights))
            n_val += 1
            if val_ap > best_ap:
                best_ap, best_weights = val_ap, weights.detach().clone()
        history.append({"epoch": epoch, "train loss": float(np.mean(losses)), "validation AUPRC": val_ap})
    return best_weights, pd.DataFrame(history), n_val


class PackedNCRCGradient:
    """Packed exact parameter shifts for independent four-wire affine-Rot NCRC models.

    The compiler fuses adjacent Euler Z angles and can shift matching coordinates
    across independent wires together. It keeps the original weight tensor and
    pulls measured derivatives back to those optimizer coordinates.
    """

    def __init__(self, n_wires, layers, prune, mode):
        if n_wires != 4 or mode not in {"parallel", "euler", "combined"}:
            raise ValueError("packed NCRC gradients require a four-wire NCRC")
        self.n_wires, self.layers, self.mode = n_wires, layers, mode
        if mode == "parallel":
            self.slots = [(l, a) for l in range(layers) for a in range(3)
                          if not (l == 0 and a == 0) and not (l == layers - 1 and a == 2)]
            self.shift_groups = [(None, k) for k in range(len(self.slots))]
        elif mode == "euler":
            self.n_slots = 2 * layers - 1
            self.shift_groups = [(q, k) for q in range(n_wires) for k in range(self.n_slots)]
        else:
            self.n_slots = 2 * layers - 1
            self.shift_groups = [(None, k) for k in range(self.n_slots)]
        self.n_slots = len(self.shift_groups)

    def coordinates(self, x, weights):
        if x.ndim == 1:
            angles = torch.einsum("qlaf,f->qla", weights[..., :x.numel()], x) + weights[..., x.numel()]
        else:
            n_features = x.shape[-1]
            angles = torch.einsum("qlaf,bf->bqla", weights[..., :n_features], x)
            angles = angles + weights[..., n_features].unsqueeze(0)
        if self.mode == "parallel":
            if x.ndim == 1:
                return torch.stack([angles[:, l, a] for l, a in self.slots], dim=1)
            return torch.stack([angles[:, :, l, a] for l, a in self.slots], dim=-1)
        if x.ndim == 1:
            theta = angles[:, :, 1]
            beta = angles[:, :-1, 2] + angles[:, 1:, 0]
            return torch.cat((theta, beta), dim=1)
        theta = angles[..., 1]
        beta = angles[..., :-1, 2] + angles[..., 1:, 0]
        return torch.cat((theta, beta), dim=-1)

    def circuit(self, coords):
        batched = coords.ndim == 3
        for q in range(self.n_wires):
            if self.mode == "parallel":
                by_layer = [[None, None, None] for _ in range(self.layers)]
                for k, (layer, axis) in enumerate(self.slots):
                    by_layer[layer][axis] = coords[:, q, k] if batched else coords[q, k]
                for layer in range(self.layers):
                    theta = by_layer[layer][1]
                    phi = torch.zeros_like(theta) if by_layer[layer][0] is None else by_layer[layer][0]
                    omega = torch.zeros_like(theta) if by_layer[layer][2] is None else by_layer[layer][2]
                    qml.Rot(phi, theta, omega, wires=q)
            else:
                for layer in range(self.layers):
                    theta = coords[:, q, layer] if batched else coords[q, layer]
                    zero = torch.zeros_like(theta)
                    if layer < self.layers - 1:
                        omega = coords[:, q, self.layers + layer] if batched else coords[q, self.layers + layer]
                    else:
                        omega = zero
                    qml.Rot(zero, theta, omega, wires=q)
        return tuple(qml.expval(qml.PauliZ(q)) for q in range(self.n_wires))

    def backward(self, model, X, y, weights, loss_fn):
        p = getattr(model, "_gradient_noise", 0.0)
        device = qml.device("default.mixed" if p else "default.qubit", wires=self.n_wires)
        if p:
            device = qml.noise.insert(device, qml.DepolarizingChannel, p, position="all")
        qnode = qml.QNode(self.circuit, device, interface="torch", diff_method=None)

        coords = self.coordinates(X, weights)
        with torch.no_grad():
            z = torch.stack(qnode(coords.detach()), dim=-1)
            dz = []
            for group_wire, k in self.shift_groups:
                delta = torch.zeros_like(coords)
                if group_wire is None:
                    delta[:, :, k] = np.pi / 2
                else:
                    delta[:, group_wire, k] = np.pi / 2
                plus = torch.stack(qnode((coords + delta).detach()), dim=-1)
                minus = torch.stack(qnode((coords - delta).detach()), dim=-1)
                dz.append((plus - minus) / 2)
            dz = torch.stack(dz, dim=-1)

        z = z.detach().requires_grad_(True)
        scores = (1 - z.mean(dim=-1)) / 2
        loss = loss_fn(scores.clamp(1e-7, 1 - 1e-7), y)
        dloss_dz, = torch.autograd.grad(loss, z)
        dloss_dc = torch.zeros_like(coords)
        for j, (group_wire, coordinate) in enumerate(self.shift_groups):
            contribution = dloss_dz * dz[:, :, j]
            if group_wire is None:
                dloss_dc[:, :, coordinate] += contribution
            else:
                dloss_dc[:, group_wire, coordinate] += contribution[:, group_wire]
        coords.backward(dloss_dc)
        return loss.detach()


# ---------------------------------------------------------------- noise model + protocol (notebook Parts 10-11)

def noisy_qnode(circuit_fn, p, n_wires=N_QUBITS, channel=qml.DepolarizingChannel):
    noisy_device = qml.noise.insert(qml.device("default.mixed", wires=n_wires), channel, p, position="all")
    qnode = qml.QNode(circuit_fn, noisy_device, interface="torch", diff_method="backprop")
    return qml.transforms.decompose(qnode, gate_set=NATIVE_GATES)


def make_noisy_model(circuit_fn, p, n_wires=N_QUBITS, channel=qml.DepolarizingChannel):
    if p == 0:
        model = make_model(circuit_fn, n_wires=n_wires)
    else:
        qnode = noisy_qnode(circuit_fn, p, n_wires=n_wires, channel=channel)
        model = lambda x, weights: readout(qnode(x, weights))
    model._gradient_noise = p
    return model


def noisy_predict(Z, weights, p, circuit_fn, n_wires=N_QUBITS, chunk=2000, channel=qml.DepolarizingChannel):
    noisy_circuit = noisy_qnode(circuit_fn, p, n_wires, channel)
    X_t = torch.tensor(Z)
    with torch.no_grad():
        return torch.cat([readout(noisy_circuit(X_t[k:k + chunk], weights)) for k in range(0, len(X_t), chunk)]).numpy()


def noise_locations(circuit_fn, *args, n_wires=N_QUBITS):
    qnode = qml.transforms.decompose(qml.QNode(circuit_fn, qml.device("default.qubit", wires=n_wires)), gate_set=NATIVE_GATES)
    return sum(size * count for size, count in qml.specs(qnode)(*args)["resources"].gate_sizes.items())


def gate_counts(circuit_fn, *args, n_wires=N_QUBITS):
    qnode = qml.transforms.decompose(qml.QNode(circuit_fn, qml.device("default.qubit", wires=n_wires)), gate_set=NATIVE_GATES)
    return dict(qml.specs(qnode)(*args)["resources"].gate_types)


def choose_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return thresholds[np.argmax(f1[:-1])]


def evaluate(y_true, scores, threshold):
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "AUPRC": average_precision_score(y_true, scores),
        "AUROC": roc_auc_score(y_true, scores),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "F1": f1_score(y_true, predicted, zero_division=0),
        "MCC": matthews_corrcoef(y_true, predicted),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


def evaluate_under_noise(circuit_fn, weights, Z_valid, Z_test, y_valid, y_test, n_wires=N_QUBITS, noise_levels=NOISE_LEVELS,
                         channel=qml.DepolarizingChannel):
    clean = qml.QNode(circuit_fn, qml.device("default.qubit", wires=n_wires), interface="torch")
    with torch.no_grad():
        clean_val = readout(clean(torch.tensor(Z_valid), weights)).numpy()
        clean_test = readout(clean(torch.tensor(Z_test), weights)).numpy()
    rows = []
    for p in [0.0, *noise_levels]:
        if p == 0.0:
            val_scores, test_scores = clean_val, clean_test
        else:
            val_scores = noisy_predict(Z_valid, weights, p, circuit_fn, n_wires, channel=channel)
            test_scores = noisy_predict(Z_test, weights, p, circuit_fn, n_wires, channel=channel)
        threshold = choose_threshold(y_valid, val_scores)
        rows.append({"p": p, **evaluate(y_test, test_scores, threshold),
                     "Spearman rho": spearmanr(clean_test, test_scores).statistic, "threshold": threshold})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- hardware cost ledger

def active_executions_per_sample(circuit_fn, weights, x, n_wires=N_QUBITS):
    """Executions PennyLane's parameter-shift actually needs per training example (counts only parameters
    in the readout's light cone), measured with qml.Tracker on a shot device (notebook Part 8)."""
    dev = qml.device("default.qubit", wires=n_wires)
    qnode = qml.set_shots(qml.QNode(circuit_fn, dev, interface="torch", diff_method="parameter-shift"), shots=SHOTS_PER_EXECUTION)
    w = weights.detach().clone().requires_grad_(True)
    with qml.Tracker(dev) as tracker:
        readout(qnode(torch.tensor(x), w)).backward()
    return tracker.totals["executions"]
