"""
MachSense Core Engine — Unified backend supporting the Streamlit Web Dashboard,
validation audits, neural inference, and headless CLI training.

All primary modeling, feature extraction, sequence windowing, training loops,
and visualization routines are also self-contained within each Jupyter Notebook:
  - notebooks/01_Paderborn_End_to_End.ipynb
  - notebooks/02_FEMTO_End_to_End.ipynb
  - notebooks/03_CMAPSS_End_to_End.ipynb
"""

from __future__ import annotations
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]

# ==============================================================================
# 1. Utilities
# ==============================================================================

def seed_everything(seed: int = 42):
    """Seed random number generators across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(obj: Any, path: Path | str):
    """Serialize object to JSON with indentation."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def require_free_space(path: Path | str, required_gb: float = 1.0):
    """Verify minimum available disk space before starting downloads or extraction."""
    free_gb = shutil.disk_usage(path).free / 1e9
    if free_gb < required_gb:
        raise RuntimeError(
            f"Insufficient disk space: {free_gb:.2f} GB available; need at least {required_gb:.2f} GB."
        )


def load_config() -> dict:
    """Load default configuration YAML if present."""
    cfg_path = ROOT / "configs" / "default.yaml"
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        'window': 32, 'stride': 4, 'hidden_dim': 64, 'layers': 3,
        'dropout': 0.15, 'condition_dim': 16, 'learning_rate': 0.001,
        'weight_decay': 0.0001, 'lambda_health': 1.0, 'lambda_anomaly': 0.20,
        'lambda_monotonic': 0.0, 'lambda_consistency': 0.50
    }


# ==============================================================================
# 2. Feature Standardization & Quality Audits
# ==============================================================================

ID_CANDIDATES = ["bearing_id", "bearing", "run_id", "run", "unit", "id"]
TIME_CANDIDATES = ["time_idx", "time", "timestamp", "cycle", "sample", "index"]
RUL_CANDIDATES = ["rul", "rul_norm", "remaining_life", "remaining_useful_life"]
HEALTH_CANDIDATES = ["health", "health_index", "hi"]
SPEED_CANDIDATES = ["speed", "rpm", "shaft_speed", "motor_speed"]
LOAD_CANDIDATES = ["load", "load_n", "force", "torque"]


def _first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}
    return next((lower[c.lower()] for c in candidates if c.lower() in lower), None)


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names for bearing ID, temporal index, speed, and load."""
    out = df.copy()
    idc = _first_col(out, ID_CANDIDATES)
    tc = _first_col(out, TIME_CANDIDATES)
    rc = _first_col(out, RUL_CANDIDATES)
    hc = _first_col(out, HEALTH_CANDIDATES)
    sc = _first_col(out, SPEED_CANDIDATES)
    lc = _first_col(out, LOAD_CANDIDATES)

    out['bearing_id'] = out[idc].astype(str) if idc else out.get('__source', pd.Series(['run_01'] * len(out))).map(lambda x: str(Path(str(x)).stem))
    out['time_idx'] = np.arange(len(out)) if tc is None else pd.to_numeric(out[tc], errors='coerce')
    
    if 'rul_raw' not in out.columns or out['rul_raw'].isna().all():
        if rc is not None and not out[rc].isna().all():
            out['rul_raw'] = pd.to_numeric(out[rc], errors='coerce')
        else:
            out['rul_raw'] = out.groupby('bearing_id')['time_idx'].transform(lambda s: s.max() - s)
    else:
        out['rul_raw'] = pd.to_numeric(out['rul_raw'], errors='coerce')

    if hc is not None:
        out['health_raw'] = pd.to_numeric(out[hc], errors='coerce')

    out['speed'] = 0.0 if sc is None else pd.to_numeric(out[sc], errors='coerce')
    out['load'] = 0.0 if lc is None else pd.to_numeric(out[lc], errors='coerce')

    for c in out.columns:
        if c not in {'bearing_id', '__source', 'split'}:
            try:
                out[c] = pd.to_numeric(out[c])
            except (TypeError, ValueError):
                pass
    return out


def choose_numeric_features(df: pd.DataFrame) -> List[str]:
    """Select pure input features, excluding target variables, IDs, and conditions."""
    excluded = {
        'time_idx', 'cycle', 'rul_raw', 'rul_norm', 'health', 'health_raw',
        'anomaly', 'anomaly_raw', 'speed', 'load', 'bearing_id', 'unit',
        '__source', 'split', 'rul_at_end'
    }
    feats = [c for c in df.select_dtypes(include=[np.number]).columns if c not in excluded]
    if not feats:
        raise RuntimeError("No model input features remain after target/ID filtering.")
    return feats


def condition_columns(df: pd.DataFrame, dataset_name: str) -> List[str]:
    """Identify operating condition columns (speed/load or turbofan settings)."""
    if str(dataset_name).lower().startswith('cmapss'):
        cols = [c for c in ('setting_1', 'setting_2', 'setting_3') if c in df.columns]
        if len(cols) != 3:
            raise ValueError(f"C-MAPSS condition columns missing: {cols}")
        return cols
    if 'speed' not in df.columns:
        df['speed'] = 0.0
    if 'load' not in df.columns:
        df['load'] = 0.0
    return ['speed', 'load']


def audit_table(df: pd.DataFrame, name: str, min_runs: int = 2, min_rows: int = 64) -> dict:
    """Audit dataset consistency, check for finite values, and confirm run integrity."""
    required = {'bearing_id', 'time_idx'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing required columns {sorted(missing)}")
    if len(df) < min_rows:
        raise ValueError(f"{name}: only {len(df)} rows; expected at least {min_rows}")
    if df.bearing_id.astype(str).nunique() < min_runs:
        raise ValueError(f"{name}: not enough independent runs (found {df.bearing_id.nunique()})")
    if df.time_idx.isna().any():
        raise ValueError(f"{name}: NaN found in time_idx")

    if 'split' in df.columns and (df['split'].astype(str).str.lower() == 'train').any():
        check_df = df[df['split'].astype(str).str.lower() == 'train']
    else:
        check_df = df

    cols_to_check = [c for c in check_df.select_dtypes(include='number').columns if c not in {'health_raw', 'rul_at_end'}]
    num = check_df[cols_to_check].to_numpy()
    if not np.isfinite(num).all():
        raise ValueError(f"{name}: NaN or Inf exists in numeric columns")

    dup = int(df.duplicated(['bearing_id', 'time_idx']).sum())
    if dup:
        raise ValueError(f"{name}: {dup} duplicate run/time rows detected")

    return {
        'name': name,
        'rows': len(df),
        'runs': int(df.bearing_id.astype(str).nunique()),
        'columns': len(df.columns),
        'missing_values': int(check_df[cols_to_check].isna().sum().sum())
    }


def assert_no_leakage(features: List[str], forbidden: tuple = ('rul', 'health', 'anomaly', 'stage', 'time_idx', 'bearing_id', 'unit', 'split', 'cycle')):
    """Strict assertion preventing target/temporal variables from leaking into model inputs."""
    bad = [c for c in features if any(x == c.lower() or c.lower().startswith(x + '_') for x in forbidden)]
    if bad:
        raise ValueError(f"Potential target/ID leakage in input features: {bad}")


def check_checkpoint(path: Path | str) -> dict:
    """Verify serialized PyTorch checkpoint format, weight keys, and statistics."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {p}")
    s = torch.load(p, map_location='cpu', weights_only=False)
    required = ['model_state', 'features', 'condition_cols', 'feature_mean', 'feature_std', 'cond_mean', 'cond_std', 'config']
    missing = [k for k in required if k not in s]
    if missing:
        raise ValueError(f"Checkpoint missing required keys: {missing}")
    if len(s['features']) != len(s['feature_mean']) or len(s['features']) != len(s['feature_std']):
        raise ValueError("Feature statistic dimensions do not match feature list length")
    if len(s['condition_cols']) != len(s['cond_mean']) or len(s['condition_cols']) != len(s['cond_std']):
        raise ValueError("Condition statistic dimensions do not match condition list length")
    return s


# ==============================================================================
# 3. MachSense Neural Architecture
# ==============================================================================

class ConditionFiLM(nn.Module):
    """Feature-wise Linear Modulation conditioning on operating regime."""
    def __init__(self, condition_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(condition_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden * 2)
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.net(c)


class TemporalBlock(nn.Module):
    """Dilated residual 1D temporal convolution block."""
    def __init__(self, channels: int, dilation: int, dropout: float = 0.15):
        super().__init__()
        pad = dilation
        g = 8 if channels >= 8 else 1
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=pad, dilation=dilation)
        self.norm1 = nn.GroupNorm(g, channels)
        self.norm2 = nn.GroupNorm(g, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dropout(F.gelu(self.norm1(self.conv1(x))))
        y = self.dropout(F.gelu(self.norm2(self.conv2(y))))
        return x + y


class MachSenseNet(nn.Module):
    """
    Multi-Task Condition-Aware Temporal Convolutional Network.
    Outputs:
      - Health Index: Continuous degradation state [0, 1]
      - Quantile RUL: Probabilistic Remaining Useful Life [p10, p50, p90]
      - Feature Reconstruction: Autoencoder decoder for unsupervised anomaly scoring
    """
    def __init__(self, n_features: int, n_conditions: int, hidden: int = 64, layers: int = 3, dropout: float = 0.15, cond_dim: int = 16):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU()
        )
        self.film = ConditionFiLM(n_conditions, hidden)
        self.cond_proj = nn.Sequential(
            nn.Linear(n_conditions, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.GELU()
        )
        self.tcn = nn.ModuleList([TemporalBlock(hidden, 2**i, dropout) for i in range(layers)])
        self.pool = nn.Sequential(
            nn.Linear(hidden + cond_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.health = nn.Sequential(
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
        self.rul = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 3),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_features)
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> dict:
        h = self.input_proj(x)
        gamma, beta = self.film(condition).chunk(2, dim=-1)
        h = h * (1 + 0.1 * torch.tanh(gamma)) + 0.1 * torch.tanh(beta)
        h = h.transpose(1, 2)
        for block in self.tcn:
            h = block(h)
        h = h.transpose(1, 2)
        context = h[:, -1]
        cond = self.cond_proj(condition[:, -1])
        z = self.pool(torch.cat([context, cond], dim=-1))
        return {
            'health': self.health(z).squeeze(-1),
            'rul': self.rul(z),
            'reconstruction': self.decoder(z),
            'latent': z
        }


# ==============================================================================
# 4. Multi-Task Losses & Sequence Dataset
# ==============================================================================

def quantile_loss(pred: torch.Tensor, target: torch.Tensor, taus: tuple = (0.1, 0.5, 0.9)) -> torch.Tensor:
    """Asymmetric pinball quantile loss for confidence intervals."""
    target = target.unsqueeze(1).expand_as(pred)
    tau = torch.tensor(taus, dtype=pred.dtype, device=pred.device).view(1, -1)
    err = target - pred
    return torch.maximum(tau * err, (tau - 1) * err).mean()


def machsense_loss(outputs: dict, target_rul: torch.Tensor, target_health: torch.Tensor, target_x_last: torch.Tensor,
                   *, lh: float = 1.0, la: float = 0.2, lm: float = 0.0, lc: float = 0.5) -> Tuple[torch.Tensor, dict]:
    """Unified loss combining Quantile Pinball, Health Regression, and Reconstruction MSE."""
    q = quantile_loss(outputs['rul'], target_rul)
    h = F.smooth_l1_loss(outputs['health'], target_health)
    recon = F.mse_loss(outputs['reconstruction'], target_x_last)
    consistency = F.smooth_l1_loss(outputs['health'], outputs['rul'][:, 1].detach())
    total = q + lh * h + la * recon + lc * consistency
    return total, {
        'quantile': float(q.detach()),
        'health': float(h.detach()),
        'reconstruction': float(recon.detach()),
        'consistency': float(consistency.detach())
    }


def build_targets(df: pd.DataFrame, train_rul_scale: float = None) -> pd.DataFrame:
    """Construct normalized ground-truth Remaining Useful Life and Health targets."""
    d = df.copy()
    d['rul_raw'] = pd.to_numeric(d.get('rul_raw', np.nan), errors='coerce')
    miss = d.groupby('bearing_id')['rul_raw'].transform(lambda s: s.notna().mean() if len(s) else 0) < 0.2
    if miss.any():
        inferred = d.groupby('bearing_id')['time_idx'].transform(lambda s: s.max() - s)
        d.loc[miss, 'rul_raw'] = inferred.loc[miss]
    scale = float(train_rul_scale if train_rul_scale is not None else max(d['rul_raw'].max(), 1.0))
    d['rul_norm'] = (d['rul_raw'] / max(scale, 1e-8)).clip(0, 1)
    d['health'] = d['rul_norm']
    return d


class SequenceDataset(Dataset):
    """Generates sliding time-series windows (WINDOW, N_FEATURES) per run/bearing."""
    def __init__(self, df: pd.DataFrame, feature_cols: list, condition_cols: list,
                 window: int = 32, stride: int = 4, fit_stats: tuple = None, fit_cond_stats: tuple = None):
        self.feature_cols = feature_cols
        self.condition_cols = condition_cols
        self.window = int(window)
        self.samples = []
        d = df.sort_values(['bearing_id', 'time_idx']).reset_index(drop=True)
        
        Xraw = d[feature_cols].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
        Xraw = np.nan_to_num(Xraw, nan=0.0, posinf=0.0, neginf=0.0)
        if fit_stats is None:
            self.mean = Xraw.mean(0)
            self.std = Xraw.std(0) + 1e-6
        else:
            self.mean, self.std = fit_stats
        X = (Xraw - self.mean) / self.std

        Craw = d[condition_cols].apply(pd.to_numeric, errors='coerce').to_numpy(np.float32)
        Craw = np.nan_to_num(Craw, nan=0.0, posinf=0.0, neginf=0.0)
        if fit_cond_stats is None:
            self.cond_mean = Craw.mean(0)
            self.cond_std = Craw.std(0) + 1e-6
        else:
            self.cond_mean, self.cond_std = fit_cond_stats
        C = (Craw - self.cond_mean) / self.cond_std

        for bid, idxs in d.groupby('bearing_id', sort=False).groups.items():
            idxs = np.asarray(list(idxs), dtype=int)
            idxs = idxs[np.argsort(d.loc[idxs, 'time_idx'].to_numpy())]
            for end in range(self.window - 1, len(idxs), max(int(stride), 1)):
                sel = idxs[end - self.window + 1 : end + 1]
                last = sel[-1]
                self.samples.append((
                    X[sel], C[sel],
                    float(d.loc[last, 'rul_norm']),
                    float(d.loc[last, 'health']),
                    str(bid),
                    float(d.loc[last, 'time_idx'])
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int):
        x, c, r, h, bid, t = self.samples[i]
        return {
            'x': torch.tensor(x, dtype=torch.float32),
            'condition': torch.tensor(c, dtype=torch.float32),
            'rul': torch.tensor(r, dtype=torch.float32),
            'health': torch.tensor(h, dtype=torch.float32),
            'x_last': torch.tensor(x[-1], dtype=torch.float32),
            'bearing_id': bid,
            'time_idx': torch.tensor(t, dtype=torch.float32)
        }


# ==============================================================================
# 5. Prescriptive Rules & Inference Engine
# ==============================================================================

def anomaly_probability(error: float, threshold: float, scale: float) -> float:
    """Map reconstruction MSE to continuous anomaly likelihood using calibrated baseline."""
    z = (error - threshold) / max(scale, 1e-6)
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -10, 10))))


def lifecycle_stage(health: float, anomaly_prob: float) -> str:
    """Categorize machine operational lifecycle stage."""
    if health >= 80 and anomaly_prob < 30:
        return "Nominal / Baseline"
    elif health >= 50 and anomaly_prob < 60:
        return "Early Degradation"
    elif health >= 20:
        return "Accelerated Wear"
    else:
        return "Critical Pre-Failure"


def maintenance_action(health: float, rul_p10: float, rul_p50: float, anomaly_prob: float) -> str:
    """Recommend actionable prescriptive maintenance decision."""
    if health < 25 or anomaly_prob > 80:
        return "IMMEDIATE SHUTDOWN & INSPECTION REQUIRED"
    elif health < 50 or rul_p10 < 20:
        return "Schedule replacement during next planned maintenance window"
    elif anomaly_prob > 50:
        return "Increase vibration sampling frequency & monitor thermal trend"
    return "Normal operation; continue regular monitoring"


def load_checkpoint(dataset_name: str) -> Tuple[Path, dict]:
    """Load serialized checkpoint for a benchmark dataset."""
    path = ROOT / "artifacts" / "checkpoints" / f"machsense_{dataset_name}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path, torch.load(path, map_location="cpu", weights_only=False)


def predict_series(df: pd.DataFrame, checkpoint: dict, model: nn.Module = None) -> pd.DataFrame:
    """
    Run temporal sliding-window inference across an ordered machine trajectory.
    Computes:
      - Health index trajectory (0-100%)
      - Probabilistic RUL bounds (p10, p50, p90 in native units)
      - Reconstruction anomaly scores & probabilities
      - Lifecycle stages and maintenance actions
    """
    d = standardize(df).sort_values(['bearing_id', 'time_idx']).reset_index(drop=True)
    if len(d) == 0:
        return pd.DataFrame()

    features = checkpoint['features']
    cond_cols = checkpoint['condition_cols']
    missing = [c for c in features + cond_cols if c not in d.columns]
    if missing:
        raise ValueError(f"Missing required model input columns: {missing}")

    X = d[features].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
    X = (X - np.asarray(checkpoint['feature_mean'])) / (np.asarray(checkpoint['feature_std']) + 1e-6)
    C = d[cond_cols].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
    C = (C - np.asarray(checkpoint['cond_mean'])) / (np.asarray(checkpoint['cond_std']) + 1e-6)

    cfg = checkpoint['config']
    w = int(cfg['window'])
    if model is None:
        model = MachSenseNet(
            len(features), len(cond_cols),
            hidden=cfg['hidden_dim'], layers=cfg['layers'],
            dropout=cfg['dropout'], cond_dim=cfg['condition_dim']
        )
        model.load_state_dict(checkpoint['model_state'])
        model.eval()

    pad_len = max(0, w - len(d))
    if pad_len > 0:
        X_padded = np.pad(X, ((pad_len, 0), (0, 0)), mode='edge')
        C_padded = np.pad(C, ((pad_len, 0), (0, 0)), mode='edge')
        indices = [len(d) - 1]
    else:
        X_padded, C_padded = X, C
        indices = list(range(w - 1, len(d)))

    if not indices:
        return pd.DataFrame()

    batch_X = torch.stack([torch.from_numpy(X_padded[idx + pad_len - w + 1 : idx + pad_len + 1]) for idx in indices])
    batch_C = torch.stack([torch.from_numpy(C_padded[idx + pad_len - w + 1 : idx + pad_len + 1]) for idx in indices])

    rows = []
    with torch.no_grad():
        out = model(batch_X, batch_C)
        health_arr = out['health'].squeeze(-1).numpy() if out['health'].ndim > 1 else out['health'].numpy()
        rul_arr = out['rul'].numpy()
        rec_arr = torch.mean((out['reconstruction'] - batch_X[:, -1, :]) ** 2, dim=-1).numpy()
        thresh = checkpoint.get('anomaly_threshold', 0.02)
        ascale = checkpoint.get('anomaly_scale', 0.01)
        rscale = checkpoint.get('rul_scale', 1.0)

        for i, end in enumerate(indices):
            h = float(health_arr[i])
            q = rul_arr[i]
            rec = float(rec_arr[i])
            a = anomaly_probability(rec, thresh, ascale)
            rows.append({
                'index': end,
                'bearing_id': str(d.loc[end, 'bearing_id']),
                'time_idx': d.loc[end, 'time_idx'],
                'health': 100 * h,
                'rul_p10': 100 * float(q[0]),
                'rul_p50': 100 * float(q[1]),
                'rul_p90': 100 * float(q[2]),
                'anomaly': 100 * a,
                'stage': lifecycle_stage(100 * h, 100 * a),
                'action': maintenance_action(100 * h, float(q[0]), float(q[1]), 100 * a),
                'rul_p50_native': float(q[1] * rscale)
            })
    return pd.DataFrame(rows)


def train_machsense(df: pd.DataFrame, dataset_name: str, epochs: int = 40, batch_size: int = 32, seed: int = 42, window: int = None):
    """Headless CLI training utility (mirrors notebook training for scripts)."""
    seed_everything(seed)
    cfg = load_config()
    cfg['window'] = int(window or cfg['window'])

    d = standardize(df)
    name = dataset_name.lower()
    if name == 'femto' and 'split' in d.columns:
        train = d[d['split'].astype(str).str.lower().eq('train')].copy()
        if train.empty:
            raise ValueError('FEMTO training split is empty')
        d = train
    elif name.startswith('cmapss') and 'split' in d.columns:
        d = d[d['split'].eq('train')].copy()

    bearings = sorted(d.bearing_id.astype(str).unique())
    if len(bearings) < 2:
        raise ValueError(f"Need at least 2 independent runs; found {len(bearings)}")

    val_bid = bearings[-1]
    train_df = d[d.bearing_id.astype(str) != val_bid].copy()
    val_df = d[d.bearing_id.astype(str) == val_bid].copy()
    scale = float(train_df.groupby('bearing_id')['rul_raw'].max().max())

    train_df = build_targets(train_df, scale)
    val_df = build_targets(val_df, scale)
    merged = pd.concat([train_df, val_df], ignore_index=True)
    features = choose_numeric_features(merged)
    cond_cols = condition_columns(train_df, dataset_name)

    train_ds = SequenceDataset(train_df, features, cond_cols, cfg['window'], cfg['stride'])
    val_ds = SequenceDataset(val_df, features, cond_cols, cfg['window'], cfg['stride'],
                             fit_stats=(train_ds.mean, train_ds.std),
                             fit_cond_stats=(train_ds.cond_mean, train_ds.cond_std))

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MachSenseNet(len(features), len(cond_cols), hidden=cfg['hidden_dim'], layers=cfg['layers'], dropout=cfg['dropout'], cond_dim=cfg['condition_dim'])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])

    ckpt_dir = ROOT / 'artifacts' / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / f'machsense_{dataset_name}.pt'

    best = float('inf')
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for b in train_dl:
            opt.zero_grad()
            out = model(b['x'], b['condition'])
            loss, _ = machsense_loss(out, b['rul'], b['health'], b['x_last'], lh=cfg['lambda_health'], la=cfg['lambda_anomaly'], lm=cfg['lambda_monotonic'], lc=cfg['lambda_consistency'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(b['rul'])

        model.eval()
        vals, ys, ps = [], [], []
        with torch.no_grad():
            for b in val_dl:
                out = model(b['x'], b['condition'])
                loss, _ = machsense_loss(out, b['rul'], b['health'], b['x_last'], lh=cfg['lambda_health'], la=cfg['lambda_anomaly'], lm=cfg['lambda_monotonic'], lc=cfg['lambda_consistency'])
                vals.append(loss.item() * len(b['rul']))
                ys.extend(b['rul'].numpy().tolist())
                ps.extend(out['rul'][:, 1].numpy().tolist())

        val_loss = sum(vals) / len(val_ds)
        rmse = float(np.sqrt(np.mean((np.asarray(ys) - np.asarray(ps))**2)))
        history.append({'epoch': ep, 'train_loss': total / len(train_ds), 'val_loss': val_loss, 'val_rul_rmse_norm': rmse})

        if val_loss < best:
            best = val_loss
            torch.save({
                'model_state': model.state_dict(),
                'features': features,
                'condition_cols': cond_cols,
                'feature_mean': train_ds.mean,
                'feature_std': train_ds.std,
                'cond_mean': train_ds.cond_mean,
                'cond_std': train_ds.cond_std,
                'config': cfg,
                'dataset': dataset_name,
                'validation_bearing': val_bid,
                'rul_scale': scale
            }, ckpt)

        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"[{dataset_name}] epoch={ep:03d} train={total/len(train_ds):.4f} val={val_loss:.4f} normalized_RMSE={rmse:.4f}")

    # Calibrate anomaly threshold
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model_state'])
    model.eval()
    healthy_err = []
    with torch.no_grad():
        for b in DataLoader(train_ds, batch_size=256, shuffle=False):
            keep = b['rul'] > 0.8
            if keep.any():
                out = model(b['x'], b['condition'])
                healthy_err.extend(torch.mean((out['reconstruction'] - b['x_last'])**2, dim=1)[keep].numpy().tolist())

    state['anomaly_threshold'] = float(np.quantile(healthy_err, 0.95)) if healthy_err else 0.02
    state['anomaly_scale'] = float(max(np.std(healthy_err), 1e-5)) if healthy_err else 0.01
    torch.save(state, ckpt)

    metrics = {
        'dataset': dataset_name,
        'validation_bearing': val_bid,
        'n_train_bearings': len(bearings) - 1,
        'features': features,
        'condition_cols': cond_cols,
        'history': history,
        'best_val_loss': best,
        'rul_scale': scale,
        'checkpoint': str(ckpt)
    }
    save_json(metrics, ROOT / 'artifacts' / f'metrics_{dataset_name}.json')

    # Export check sample
    cols = ['bearing_id', 'time_idx'] + cond_cols + features
    sample = val_df[cols].copy().tail(160)
    p = ROOT / 'artifacts' / 'check_samples'
    p.mkdir(parents=True, exist_ok=True)
    sample_out = p / f'{dataset_name}_validation_input.csv'
    sample.to_csv(sample_out, index=False)
    return ckpt, metrics, sample_out
