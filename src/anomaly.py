from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib


class FeatureAutoencoder(nn.Module):
    def __init__(self, n_features: int, latent_dim: int = 8):
        super().__init__()
        h = max(16, min(64, n_features * 2))
        self.encoder = nn.Sequential(nn.Linear(n_features, h), nn.ReLU(), nn.Linear(h, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, h), nn.ReLU(), nn.Linear(h, n_features))

    def forward(self, x):
        return self.decoder(self.encoder(x))


@dataclass
class AnomalyArtifact:
    feature_columns: list[str]
    mean: np.ndarray
    std: np.ndarray
    threshold: float
    checkpoint: str


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"bearing_id", "time_idx", "rul_raw", "source_file"}
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in excluded]


def train_autoencoder(df: pd.DataFrame, out_path: Path, epochs: int = 20, batch_size: int = 256, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    feats = feature_columns(df)
    # Only healthy early-life observations are used for fitting.
    frac = df.groupby("bearing_id")["time_idx"].transform(lambda s: s / max(float(s.max()), 1.0))
    healthy = df[frac <= 0.20].copy()
    X = healthy[feats].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-6
    Xn = (X - mean) / std
    loader = DataLoader(TensorDataset(torch.tensor(Xn)), batch_size=batch_size, shuffle=True)

    model = FeatureAutoencoder(len(feats))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for (xb,) in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        history.append(total / max(len(healthy), 1))

    model.eval()
    with torch.no_grad():
        errs = []
        for (xb,) in DataLoader(TensorDataset(torch.tensor(Xn)), batch_size=1024):
            errs.extend(torch.mean((model(xb) - xb) ** 2, dim=1).numpy())
    threshold = float(np.quantile(errs, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "n_features": len(feats), "latent_dim": 8}, out_path)
    meta = {"features": feats, "mean": mean.tolist(), "std": std.tolist(), "threshold": threshold,
            "history": history, "healthy_fraction": 0.20}
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return model, meta


def score_autoencoder(df: pd.DataFrame, model: FeatureAutoencoder, meta: dict) -> pd.DataFrame:
    X = df[meta["features"]].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
    X = (X - np.asarray(meta["mean"], dtype=np.float32)) / (np.asarray(meta["std"], dtype=np.float32) + 1e-6)
    with torch.no_grad():
        rec = model(torch.tensor(X)).numpy()
    out = df[["bearing_id", "time_idx", "rul_raw"]].copy()
    out["reconstruction_error"] = np.mean((X - rec) ** 2, axis=1)
    out["anomaly"] = out["reconstruction_error"] > float(meta["threshold"])
    out["anomaly_score"] = np.minimum(out["reconstruction_error"] / max(float(meta["threshold"]), 1e-8), 3.0) / 3.0
    return out


def train_isolation_forest(df: pd.DataFrame, out_path: Path, seed: int = 42):
    feats = feature_columns(df)
    frac = df.groupby("bearing_id")["time_idx"].transform(lambda s: s / max(float(s.max()), 1.0))
    healthy = df[frac <= 0.20]
    X = healthy[feats].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
    scaler = StandardScaler().fit(X)
    model = IsolationForest(n_estimators=250, contamination="auto", random_state=seed, n_jobs=-1)
    model.fit(scaler.transform(X))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "features": feats}, out_path)
    return model
