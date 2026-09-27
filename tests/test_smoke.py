import numpy as np
import pandas as pd
import torch
from pathlib import Path
from src.model import MachSenseNet
from src.features import condition_columns, choose_numeric_features, standardize
from src.validation import audit_table, assert_no_leakage, check_checkpoint
from src.experiment import predict_series, load_checkpoint

def test_dynamic_model():
    m = MachSenseNet(6, 3, hidden=32, layers=2, cond_dim=8)
    o = m(torch.randn(4, 16, 6), torch.randn(4, 16, 3))
    assert o['rul'].shape == (4, 3)
    assert o['health'].shape == (4,)
    assert o['reconstruction'].shape == (4, 6)

def test_schema():
    d = pd.DataFrame({
        'bearing_id': ['a'] * 40 + ['b'] * 40,
        'time_idx': list(range(40)) * 2,
        'rul_raw': np.tile(np.arange(39, -1, -1), 2),
        'f1': np.random.rand(80),
        'f2': np.random.rand(80),
        'speed': 1.0,
        'load': 2.0
    })
    audit = audit_table(d, 'demo')
    assert audit['runs'] == 2
    assert audit['missing_values'] == 0
    assert condition_columns(d, 'paderborn') == ['speed', 'load']
    assert_no_leakage(['f1', 'f2'])

def test_checkpoints_and_predict():
    for ds in ['paderborn', 'cmapss']:
        ckpt_path = Path(f'artifacts/checkpoints/machsense_{ds}.pt')
        if ckpt_path.exists():
            ckpt = check_checkpoint(ckpt_path)
            assert 'model_state' in ckpt
            assert len(ckpt['features']) > 0
            assert len(ckpt['condition_cols']) > 0
            
            # Test inference with dummy dataframe
            dummy = pd.DataFrame({
                'bearing_id': ['test_unit'] * 10,
                'time_idx': list(range(10))
            })
            for f in ckpt['features']:
                dummy[f] = np.random.randn(10)
            for c in ckpt['condition_cols']:
                dummy[c] = 1.0
            pred = predict_series(dummy, ckpt)
            assert not pred.empty
            assert 'health' in pred.columns
            assert 'rul_p50' in pred.columns
            assert 'anomaly' in pred.columns
            assert 'stage' in pred.columns
            assert 'action' in pred.columns
