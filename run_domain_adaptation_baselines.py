"""
R2.6 — Domain-adaptation baselines for the zero-shot chemistry transfer task.

Reviewer 2 (comment 6) noted that the zero-shot baselines (LSTM/GRU/etc.) were
not designed for cross-chemistry transfer, so the comparison may understate
what conventional methods can do. This script evaluates domain-adaptation
variants of the same baselines under the IDENTICAL protocol used by
`src/recreate_zeroshot_baseline.py` (train: LCO — NASA/CALCE/Oxford;
test: Panasonic 18650PF), on the IDENTICAL 500-sample evaluation subset
(seed 123), so results are directly comparable to Table tab:zeroshot.

Methods:
  1. Zero-shot re-evaluation (sanity anchor — should match paper numbers).
  2. CORAL feature alignment (Sun & Saenko 2016): whiten source features and
     re-color with the target covariance estimated from UNLABELED adaptation
     samples, then train the baseline on the aligned source.
  3. MMD alignment: encoder trained with source supervision + RBF-kernel MMD
     penalty between source and unlabeled-target encodings.
  4. Few-shot fine-tuning: train on source, then fine-tune on a small labeled
     target budget (200 samples, ~5% of the target pool, disjoint from the
     evaluation subset).

Run:  PYTHONPATH=. arch -arm64 python3 run_domain_adaptation_baselines.py
Output: reports/domain_adaptation_baselines.json
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import random
from pathlib import Path

import numpy as np

from src.recreate_zeroshot_baseline import (
    RUL_SCALE_CYCLES,
    load_zeroshot_samples,
    samples_to_arrays,
    standardize_fit_transform,
    standardize_transform,
    mean_absolute_error_np,
    r2_score_np,
)

EVAL_SEED = 123          # must match recreate_zeroshot_baseline.load_zeroshot_samples
EVAL_SIZE = 500          # paper evaluation subset size
FEWSHOT_BUDGET = 200     # labeled target samples for fine-tuning (~5% of pool)
TRAIN_EPOCHS = 200       # same as recreate_zeroshot_baseline
FT_EPOCHS = 50
FT_LR = 1e-4
MMD_LAMBDA = 1.0


def set_seed(seed: int = 42) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def load_split():
    """Load LCO source and Panasonic target; reproduce the paper's 500-sample
    eval subset and use the remaining target samples as the adaptation pool."""
    train_samples, target_samples = load_zeroshot_samples(
        max_train_samples=4000, max_test_samples=0
    )
    n = len(target_samples)
    rng = np.random.default_rng(EVAL_SEED)
    eval_idx = set(rng.choice(n, size=EVAL_SIZE, replace=False).tolist())
    eval_samples = [target_samples[i] for i in sorted(eval_idx)]
    adapt_samples = [target_samples[i] for i in range(n) if i not in eval_idx]
    return train_samples, eval_samples, adapt_samples


def train_model(model, X, soh, rul_norm, epochs=TRAIN_EPOCHS, lr=1e-3):
    import torch
    import torch.nn as nn
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        soh_pred, rul_pred = model(X)
        loss = crit(soh_pred, soh) + crit(rul_pred, rul_norm)
        loss.backward()
        opt.step()
    return model


def evaluate(model, X_eval, soh_eval, rul_cycles_eval):
    import torch
    model.eval()
    with torch.no_grad():
        soh_pred, rul_pred = model(X_eval)
    soh_pred = soh_pred.numpy()
    rul_pred_cycles = rul_pred.numpy() * RUL_SCALE_CYCLES
    return {
        "soh_mae": float(mean_absolute_error_np(soh_eval, soh_pred) * 100.0),
        "soh_r2": float(r2_score_np(soh_eval, soh_pred)),
        "rul_mae": float(np.mean(np.abs(rul_pred_cycles - rul_cycles_eval))),
    }


def coral_transform(X_src: np.ndarray, X_tgt: np.ndarray) -> np.ndarray:
    """CORAL: align second-order statistics of source to target."""
    eps = 1e-3
    d = X_src.shape[1]
    cov_s = np.cov(X_src, rowvar=False) + eps * np.eye(d)
    cov_t = np.cov(X_tgt, rowvar=False) + eps * np.eye(d)

    def mat_pow(C, p):
        vals, vecs = np.linalg.eigh(C)
        vals = np.clip(vals, eps, None)
        return (vecs * (vals ** p)) @ vecs.T

    return (X_src - X_src.mean(0)) @ mat_pow(cov_s, -0.5) @ mat_pow(cov_t, 0.5) + X_tgt.mean(0)


def rbf_mmd(x, y, sigmas=(1.0, 2.0, 4.0, 8.0)):
    import torch

    def gram(a, b):
        d2 = torch.cdist(a, b) ** 2
        return sum(torch.exp(-d2 / (2 * s * s)) for s in sigmas)

    return gram(x, x).mean() + gram(y, y).mean() - 2 * gram(x, y).mean()


class MMDRegressor:
    """MLP encoder + heads, trained with source MSE + MMD(source, target)."""

    def __init__(self, input_dim, hidden_dim=128):
        import torch.nn as nn
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.soh_head = nn.Linear(hidden_dim, 1)
        self.rul_head = nn.Linear(hidden_dim, 1)

    def parameters(self):
        for m in (self.encoder, self.soh_head, self.rul_head):
            yield from m.parameters()

    def __call__(self, x):
        z = self.encoder(x)
        return self.soh_head(z).squeeze(-1), self.rul_head(z).squeeze(-1)

    def train_mode(self):
        for m in (self.encoder, self.soh_head, self.rul_head):
            m.train()

    def eval_mode(self):
        for m in (self.encoder, self.soh_head, self.rul_head):
            m.eval()


def train_mmd(model, X_src, soh_src, rul_src, X_tgt, epochs=TRAIN_EPOCHS,
              lr=1e-3, batch=256, lam=MMD_LAMBDA):
    import torch
    import torch.nn as nn
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    n_s, n_t = X_src.shape[0], X_tgt.shape[0]
    g = torch.Generator().manual_seed(42)
    model.train_mode()
    for _ in range(epochs):
        si = torch.randint(0, n_s, (batch,), generator=g)
        ti = torch.randint(0, n_t, (batch,), generator=g)
        xs, xt = X_src[si], X_tgt[ti]
        opt.zero_grad()
        zs = model.encoder(xs)
        zt = model.encoder(xt)
        soh_pred = model.soh_head(zs).squeeze(-1)
        rul_pred = model.rul_head(zs).squeeze(-1)
        loss = (crit(soh_pred, soh_src[si]) + crit(rul_pred, rul_src[si])
                + lam * rbf_mmd(zs, zt))
        loss.backward()
        opt.step()
    return model


def evaluate_mmd(model, X_eval, soh_eval, rul_cycles_eval):
    import torch
    model.eval_mode()
    with torch.no_grad():
        soh_pred, rul_pred = model(X_eval)
    soh_pred = soh_pred.numpy()
    rul_pred_cycles = rul_pred.numpy() * RUL_SCALE_CYCLES
    return {
        "soh_mae": float(mean_absolute_error_np(soh_eval, soh_pred) * 100.0),
        "soh_r2": float(r2_score_np(soh_eval, soh_pred)),
        "rul_mae": float(np.mean(np.abs(rul_pred_cycles - rul_cycles_eval))),
    }


def main():
    import torch
    from src.sota_baseline_comparison import (
        LSTMBaseline, TransformerBaseline, MLPBaseline
    )

    set_seed(42)
    print("Loading LCO source / Panasonic target splits...")
    train_samples, eval_samples, adapt_samples = load_split()

    X_src, soh_src, rul_norm_src, _, _ = samples_to_arrays(train_samples)
    X_eval, soh_eval, _, rul_cycles_eval, _ = samples_to_arrays(eval_samples)
    X_adapt, soh_adapt, rul_norm_adapt, _, _ = samples_to_arrays(adapt_samples)

    print(f"source={len(X_src)}  eval={len(X_eval)}  adapt pool={len(X_adapt)}")

    # Labeled few-shot budget drawn from the adaptation pool (never from eval)
    rng = np.random.default_rng(7)
    fs_idx = rng.choice(len(X_adapt), size=FEWSHOT_BUDGET, replace=False)

    # Source-fitted standardization, as in the original protocol
    X_src_s, mean, std = standardize_fit_transform(X_src)
    X_eval_s = standardize_transform(X_eval, mean, std)
    X_adapt_s = standardize_transform(X_adapt, mean, std)

    t = lambda a: torch.tensor(a, dtype=torch.float32)
    results = {"protocol": {
        "source": "LCO (NASA, CALCE, Oxford), 4000 samples",
        "target": "Panasonic 18650PF (NCA), 500-sample eval subset (seed 123)",
        "adapt_pool": f"{len(X_adapt)} target samples disjoint from eval",
        "fewshot_budget": FEWSHOT_BUDGET,
        "mmd_lambda": MMD_LAMBDA,
        "ft_epochs": FT_EPOCHS,
        "ft_lr": FT_LR,
    }}

    arch_factories = {
        "LSTM": lambda: LSTMBaseline(X_src.shape[1]),
        "Transformer": lambda: TransformerBaseline(X_src.shape[1]),
        "MLP": lambda: MLPBaseline(X_src.shape[1]),
    }

    for name, factory in arch_factories.items():
        # 1) Zero-shot anchor
        set_seed(42)
        model = train_model(factory(), t(X_src_s), t(soh_src), t(rul_norm_src))
        results[f"{name} (zero-shot)"] = evaluate(model, t(X_eval_s), soh_eval, rul_cycles_eval)
        print(name, "zero-shot:", results[f"{name} (zero-shot)"])

        # 2) CORAL-aligned source training (unlabeled target adaptation)
        set_seed(42)
        X_src_coral = coral_transform(X_src_s, X_adapt_s).astype(np.float32)
        model = train_model(factory(), t(X_src_coral), t(soh_src), t(rul_norm_src))
        results[f"{name} + CORAL"] = evaluate(model, t(X_eval_s), soh_eval, rul_cycles_eval)
        print(name, "+CORAL:", results[f"{name} + CORAL"])

        # 3) Few-shot fine-tuning on 200 labeled target samples
        set_seed(42)
        model = train_model(factory(), t(X_src_s), t(soh_src), t(rul_norm_src))
        model = train_model(
            model,
            t(X_adapt_s[fs_idx]), t(soh_adapt[fs_idx]), t(rul_norm_adapt[fs_idx]),
            epochs=FT_EPOCHS, lr=FT_LR,
        )
        results[f"{name} + few-shot FT"] = evaluate(model, t(X_eval_s), soh_eval, rul_cycles_eval)
        print(name, "+few-shot FT:", results[f"{name} + few-shot FT"])

    # 4) MMD-aligned MLP (unlabeled target adaptation)
    set_seed(42)
    mmd_model = MMDRegressor(X_src.shape[1])
    train_mmd(mmd_model, t(X_src_s), t(soh_src), t(rul_norm_src), t(X_adapt_s))
    results["MLP + MMD"] = evaluate_mmd(mmd_model, t(X_eval_s), soh_eval, rul_cycles_eval)
    print("MLP +MMD:", results["MLP + MMD"])

    out = Path("reports/domain_adaptation_baselines.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}")

    print(f"\n{'Method':<28}{'SOH MAE':>10}{'SOH R2':>10}{'RUL MAE':>12}")
    for k, v in results.items():
        if k == "protocol":
            continue
        print(f"{k:<28}{v['soh_mae']:>9.2f}%{v['soh_r2']:>10.3f}{v['rul_mae']:>10.1f} cy")
    print(f"{'HERO (paper, same eval)':<28}{'0.74':>9}%{'0.990':>10}{'44.0':>10} cy")


if __name__ == "__main__":
    main()
