from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

def audit_table(df,name,min_runs=2,min_rows=64):
    required={'bearing_id','time_idx','rul_raw'}; missing=required-set(df.columns)
    if missing: raise ValueError(f'{name}: missing columns {sorted(missing)}')
    if len(df)<min_rows: raise ValueError(f'{name}: only {len(df)} rows')
    if df.bearing_id.astype(str).nunique()<min_runs: raise ValueError(f'{name}: not enough independent runs')
    if df.time_idx.isna().any(): raise ValueError(f'{name}: NaN time_idx')
    if 'split' in df.columns and (df['split'].astype(str).str.lower() == 'train').any():
        check_df = df[df['split'].astype(str).str.lower() == 'train']
    else:
        check_df = df
    num=check_df.select_dtypes(include='number').to_numpy()
    if not np.isfinite(num).all(): raise ValueError(f'{name}: NaN/Inf exists in numeric columns')
    dup=int(df.duplicated(['bearing_id','time_idx']).sum())
    if dup: raise ValueError(f'{name}: {dup} duplicate run/time rows')
    return {'name':name,'rows':len(df),'runs':int(df.bearing_id.astype(str).nunique()),'columns':len(df.columns),'missing_values':int(check_df.isna().sum().sum())}

def assert_no_leakage(features,forbidden=('rul','health','anomaly','stage','time_idx','bearing_id','unit','split','cycle')):
    bad=[c for c in features if any(x==c.lower() or c.lower().startswith(x+'_') for x in forbidden)]
    if bad: raise ValueError(f'Potential target/ID leakage: {bad}')

def check_checkpoint(path: Path | str):
    path = Path(path)
    if not path.exists(): raise FileNotFoundError(path)
    s=torch.load(path,map_location='cpu',weights_only=False)
    required=['model_state','features','condition_cols','feature_mean','feature_std','cond_mean','cond_std','config']
    missing=[k for k in required if k not in s]
    if missing: raise ValueError(f'Checkpoint missing {missing}')
    if len(s['features'])!=len(s['feature_mean']) or len(s['features'])!=len(s['feature_std']): raise ValueError('Feature statistic dimensions do not match')
    if len(s['condition_cols'])!=len(s['cond_mean']) or len(s['condition_cols'])!=len(s['cond_std']): raise ValueError('Condition statistic dimensions do not match')
    return s
