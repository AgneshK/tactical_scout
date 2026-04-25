#!/usr/bin/env python3
"""
tune_model.py — Optuna hyperparameter search for TacticalAutoencoder.

Searches latent_dim (4–16), learning rate, hidden layer sizes, dropout,
weight decay, batch size, and optimizer to minimise validation MSE.

EYE TEST WEIGHTING
------------------
Vanilla MSE treats all ~200 FBRef features equally, so a goalkeeper save
rate has the same pull on the embedding as progressive carries.  Here,
scout-relevant attacking and creativity features get 2x reconstruction
weight so the latent space reflects tactical function — what scouts
actually observe on the pitch — rather than raw statistical volume.

ARCHITECTURE IMPROVEMENT
------------------------
Replaced ReLU with GELU (smoother gradients on normalised tabular data)
and added optional dropout for regularisation.  Hidden dim sizes are now
tuned instead of being fixed at 32→16.

Usage (run from project root or backend/):
    python backend/tune_model.py
    python backend/tune_model.py --n-trials 100
    python backend/tune_model.py --n-trials 75 --timeout 3600
    python backend/tune_model.py --study-name my_run   # resume a study
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import joblib
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import PowerTransformer, RobustScaler
from torch.utils.data import DataLoader, TensorDataset, random_split

# ── Paths ──────────────────────────────────────────────────────────────────────

_HERE         = Path(__file__).resolve().parent
DATA_PATH     = _HERE / ".." / "data" / "players_data-2024_2025.csv"
ARTIFACTS_DIR = _HERE / "artifacts"

# ── Constants ──────────────────────────────────────────────────────────────────

TRIAL_EPOCHS = 80     # max epochs per Optuna trial (early stopping cuts this short)
FINAL_EPOCHS = 200    # epochs for the champion model on the full dataset
PATIENCE     = 12     # early-stopping patience — epochs without val improvement
VAL_FRAC     = 0.15   # fraction of players held out for validation during search
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)  # suppress per-trial noise

# ── Scout-relevant feature weighting ("eye test") ──────────────────────────────
# These are the metrics that scouts can visually validate — goal threat, shot
# creation, ball progression, and key passing.  Upweighting their reconstruction
# error biases the latent space toward tactical function over volume stats, so
# two players with similar *roles* end up closer than two players who simply
# touched the ball the same number of times.

_SCOUT_FEATURES = [
    "Goals", "Assists", "xG", "xAG", "npxG",
    "PrgC",  "PrgP",   "KP", "SCA",  "GCA",
    "Sh",    "SoT",
]


def build_feature_weights(feature_cols: list) -> torch.Tensor:
    weights = torch.ones(len(feature_cols))
    for i, col in enumerate(feature_cols):
        if any(s in col for s in _SCOUT_FEATURES):
            weights[i] = 2.0
    return weights.to(DEVICE)


def weighted_mse(pred: torch.Tensor,
                 target: torch.Tensor,
                 weights: torch.Tensor) -> torch.Tensor:
    return ((pred - target).pow(2) * weights).mean()


# ── Model ──────────────────────────────────────────────────────────────────────

class TacticalAutoencoder(nn.Module):
    """
    Symmetric autoencoder with parametric hidden dims and GELU activations.

    Encoder: num_features → hidden_1 → hidden_2 → latent_dim
    Decoder: latent_dim   → hidden_2 → hidden_1 → num_features
    """

    def __init__(self, num_features: int, latent_dim: int,
                 hidden_1: int, hidden_2: int, dropout: float = 0.0):
        super().__init__()
        act = nn.GELU()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, hidden_1), act, nn.Dropout(dropout),
            nn.Linear(hidden_1,     hidden_2), act, nn.Dropout(dropout),
            nn.Linear(hidden_2,     latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,   hidden_2), act, nn.Dropout(dropout),
            nn.Linear(hidden_2,     hidden_1), act, nn.Dropout(dropout),
            nn.Linear(hidden_1,     num_features),
        )

    def forward(self, x: torch.Tensor) -> tuple:
        z = self.encoder(x)
        return z, self.decoder(z)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_and_preprocess() -> tuple:
    """
    Mirrors the preprocessing in autoencoder.py exactly.
    Returns (tensor, player_info, weights, pt, scaler, feature_cols).
    """
    df = pd.read_csv(DATA_PATH)
    df = df[~df["Pos"].str.contains("GK", na=False)]
    df = df[df["Min"] > 500].reset_index(drop=True)

    for col in ["Goals", "Assists"]:
        if col in df.columns:
            df[col] = (df[col] / df["Min"]) * 90

    _meta    = ["Player", "Pos", "Squad", "Comp"]
    _display = ["Age", "Nation", "MP", "Min", "Goals", "Assists",
                "xG", "xAG", "PrgC", "PrgP", "Tkl", "KP"]
    player_info = df[_meta + [c for c in _display if c in df.columns]]

    feat_df   = (df.select_dtypes(include=["float64", "int64"])
                   .drop(columns=["Min"], errors="ignore")
                   .fillna(0))
    feat_cols = feat_df.columns.tolist()

    pt     = PowerTransformer(method="yeo-johnson")
    scaler = RobustScaler()
    scaled = scaler.fit_transform(pt.fit_transform(feat_df))

    tensor  = torch.tensor(scaled, dtype=torch.float32)
    weights = build_feature_weights(feat_cols)

    n_scout = int((weights > 1).sum().item())
    log.info(
        "Dataset ready: %d players · %d features · %d scout-weighted (2×)",
        len(df), len(feat_cols), n_scout,
    )
    return tensor, player_info, weights, pt, scaler, feat_cols


# ── Trial training ─────────────────────────────────────────────────────────────

def run_training(
    tensor: torch.Tensor,
    weights: torch.Tensor,
    params: dict,
    epochs: int,
    patience: int,
    val_split: bool = True,
    trial: "optuna.trial.Trial | None" = None,
) -> tuple:
    """
    Train one configuration. Returns (best_val_loss, trained_model).
    Supports early stopping and Optuna median pruning.
    """
    num_features = tensor.shape[1]
    latent_dim   = params["latent_dim"]
    hidden_1     = params["hidden_1"]
    hidden_2     = params["hidden_2"]
    dropout      = params["dropout"]
    lr           = params["lr"]
    weight_decay = params["weight_decay"]
    batch_size   = params["batch_size"]
    opt_name     = params["optimizer"]

    n     = len(tensor)
    n_val = max(1, int(n * VAL_FRAC)) if val_split else 0
    n_tr  = n - n_val

    if val_split:
        tr_ds, val_ds = random_split(
            TensorDataset(tensor, tensor), [n_tr, n_val],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        tr_ds = val_ds = TensorDataset(tensor, tensor)

    tr_loader  = DataLoader(tr_ds,  batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = TacticalAutoencoder(
        num_features, latent_dim, hidden_1, hidden_2, dropout
    ).to(DEVICE)

    opt_cls   = optim.AdamW if opt_name == "adamw" else optim.Adam
    optimizer = opt_cls(model.parameters(), lr=lr, weight_decay=weight_decay)
    # CosineAnnealing smoothly decays LR to lr*0.01 over the training horizon
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_val   = float("inf")
    best_state = None
    stale      = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, _ in tr_loader:
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()
            _, recon = model(batch_x)
            weighted_mse(recon, batch_x, weights).backward()
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(DEVICE)
                _, recon = model(batch_x)
                val_loss += weighted_mse(recon, batch_x, weights).item()
        val_loss /= len(val_loader)

        # Track best and early-stop
        if val_loss < best_val - 1e-6:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale      = 0
        else:
            stale += 1
            if stale >= patience:
                break

        # Optuna pruning signal
        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    return best_val, model


# ── Optuna objective ───────────────────────────────────────────────────────────

def make_objective(tensor: torch.Tensor, weights: torch.Tensor):
    """Closure — captures pre-loaded data so each trial doesn't re-read the CSV."""

    def objective(trial: optuna.trial.Trial) -> float:
        # ── Required parameters (per user spec) ───────────────────────────────
        latent_dim = trial.suggest_int("latent_dim", 4, 16)
        lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        # ── Architecture ──────────────────────────────────────────────────────
        hidden_1 = trial.suggest_int("hidden_1", 32, 128, step=16)
        # hidden_2 is suggested over a fixed range then clamped so that
        # hidden_2 < hidden_1 (progressive compression is enforced).
        hidden_2_raw = trial.suggest_int("hidden_2_raw", 16, 64, step=8)
        hidden_2     = max(8, min(hidden_2_raw, hidden_1 - 8))

        # ── Regularisation ────────────────────────────────────────────────────
        dropout      = trial.suggest_float("dropout", 0.0, 0.35)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        batch_size   = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        optimizer    = trial.suggest_categorical("optimizer", ["adam", "adamw"])

        params = dict(
            latent_dim=latent_dim, hidden_1=hidden_1, hidden_2=hidden_2,
            dropout=dropout, lr=lr, weight_decay=weight_decay,
            batch_size=batch_size, optimizer=optimizer,
        )

        val_loss, _ = run_training(
            tensor, weights, params,
            epochs=TRIAL_EPOCHS, patience=PATIENCE,
            val_split=True, trial=trial,
        )
        return val_loss

    return objective


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna hyperparameter search for TacticalAutoencoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-trials",   type=int,  default=50,
                   help="Number of Optuna trials")
    p.add_argument("--timeout",    type=int,  default=None,
                   help="Hard time limit in seconds (runs alongside --n-trials)")
    p.add_argument("--study-name", type=str,  default="tactical_ae",
                   help="Study name — reuse to resume a previous search")
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    log.info("Device : %s", DEVICE)
    log.info("Trials : %d%s", args.n_trials,
             f"  |  Timeout: {args.timeout}s" if args.timeout else "")

    tensor, player_info, weights, pt, scaler, feat_cols = load_and_preprocess()

    # ── Build study ────────────────────────────────────────────────────────────
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner  = optuna.pruners.MedianPruner(
        n_startup_trials=10,   # first 10 trials run fully — build baseline
        n_warmup_steps=20,     # don't prune before epoch 20 (model needs warm-up)
        interval_steps=5,      # evaluate pruning every 5 epochs
    )
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    # Seed with the exact baseline from autoencoder.py so the search always has
    # a known-good reference point as trial #0.
    study.enqueue_trial({
        "latent_dim": 8, "lr": 1e-3,
        "hidden_1": 32, "hidden_2_raw": 16,
        "dropout": 0.0, "weight_decay": 1e-6,
        "batch_size": 32, "optimizer": "adam",
    })

    log.info("Starting search — baseline (autoencoder.py defaults) enqueued as trial #0")
    t0 = time.perf_counter()

    study.optimize(
        make_objective(tensor, weights),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    elapsed   = time.perf_counter() - t0
    completed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    pruned    = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    log.info("Search done in %.0fs — %d completed, %d pruned", elapsed, completed, pruned)

    # ── Results ────────────────────────────────────────────────────────────────
    best = study.best_trial
    hidden_2_final = max(8, min(best.params["hidden_2_raw"], best.params["hidden_1"] - 8))

    print("\n" + "═" * 58)
    print("  BEST TRIAL")
    print("═" * 58)
    print(f"  Trial #      : {best.number}")
    print(f"  Val MSE      : {best.value:.6f}")
    print(f"  latent_dim   : {best.params['latent_dim']}")
    print(f"  lr           : {best.params['lr']:.2e}")
    print(f"  hidden_1     : {best.params['hidden_1']}")
    print(f"  hidden_2     : {hidden_2_final}")
    print(f"  dropout      : {best.params['dropout']:.3f}")
    print(f"  weight_decay : {best.params['weight_decay']:.2e}")
    print(f"  batch_size   : {best.params['batch_size']}")
    print(f"  optimizer    : {best.params['optimizer']}")

    trials_df    = study.trials_dataframe(attrs=("number", "value", "state"))
    completed_df = trials_df[trials_df["state"] == "COMPLETE"].nsmallest(5, "value")
    print("\n  TOP 5 TRIALS")
    print("  " + "─" * 40)
    for _, row in completed_df.iterrows():
        marker = " ◀ best" if int(row["number"]) == best.number else ""
        print(f"  #{int(row['number']):>3}  val_mse = {row['value']:.6f}{marker}")
    print("═" * 58 + "\n")

    # ── Retrain champion on the full dataset ───────────────────────────────────
    best_params = {
        "latent_dim":   best.params["latent_dim"],
        "lr":           best.params["lr"],
        "hidden_1":     best.params["hidden_1"],
        "hidden_2":     hidden_2_final,
        "dropout":      best.params["dropout"],
        "weight_decay": best.params["weight_decay"],
        "batch_size":   best.params["batch_size"],
        "optimizer":    best.params["optimizer"],
    }

    log.info(
        "Retraining champion on full dataset for %d epochs "
        "(latent_dim=%d, lr=%.2e, hidden=%d→%d→%d) …",
        FINAL_EPOCHS,
        best_params["latent_dim"], best_params["lr"],
        best_params["hidden_1"], best_params["hidden_2"],
        best_params["latent_dim"],
    )

    _, final_model = run_training(
        tensor, weights, best_params,
        epochs=FINAL_EPOCHS,
        patience=FINAL_EPOCHS,  # disabled — run the full budget on the full set
        val_split=False,
    )

    # ── Save artifacts (same format as autoencoder.py) ─────────────────────────
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    final_model.eval()
    with torch.no_grad():
        embeddings = final_model.encoder(tensor.to(DEVICE)).cpu()

    torch.save(embeddings,       ARTIFACTS_DIR / "embeddings.pt")
    joblib.dump(player_info,     ARTIFACTS_DIR / "player_info.pkl")
    with open(ARTIFACTS_DIR / "best_params.json", "w") as f:
        json.dump({**best_params, "val_mse": best.value, "trial": best.number,
                   "n_features": tensor.shape[1], "n_players": len(player_info)}, f, indent=2)

    log.info("Artifacts saved to %s", ARTIFACTS_DIR)
    log.info("  embeddings.pt  — shape %s", list(embeddings.shape))
    log.info("  player_info.pkl — %d players", len(player_info))
    log.info("  best_params.json")


if __name__ == "__main__":
    main()
