from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ID_CANDIDATES=["bearing_id","bearing","run_id","run","unit","id"]
TIME_CANDIDATES=["time_idx","time","timestamp","cycle","sample","index"]
RUL_CANDIDATES=["rul","rul_norm","remaining_life","remaining_useful_life"]
HEALTH_CANDIDATES=["health","health_index","hi"]
SPEED_CANDIDATES=["speed","rpm","shaft_speed","motor_speed"]
LOAD_CANDIDATES=["load","load_n","force","torque"]

def first_col(df,candidates):
    lower={str(c).lower():c for c in df.columns}
    return next((lower[c.lower()] for c in candidates if c.lower() in lower),None)

def read_any(path:Path):
    if path.suffix.lower()=='.parquet': return pd.read_parquet(path)
    if path.suffix.lower()=='.csv': return pd.read_csv(path)
    if path.suffix.lower()=='.json': return pd.read_json(path,lines=True)
    raise ValueError(f'Unsupported file: {path}')

def load_and_standardize(input_dir):
    frames=[]
    for p in sorted(Path(input_dir).rglob('*')):
        if p.suffix.lower() not in {'.csv','.parquet','.json'}: continue
        try:
            d=read_any(p)
            if len(d)==0: continue
            d.columns=[str(c).strip() for c in d.columns]
            d['__source']=p.as_posix(); frames.append(d)
        except Exception:
            continue
    if not frames: raise FileNotFoundError(f'No readable tables under {input_dir}')
    return pd.concat(frames,ignore_index=True,sort=False)

def standardize(df):
    out=df.copy()
    idc=first_col(out,ID_CANDIDATES); tc=first_col(out,TIME_CANDIDATES); rc=first_col(out,RUL_CANDIDATES); hc=first_col(out,HEALTH_CANDIDATES); sc=first_col(out,SPEED_CANDIDATES); lc=first_col(out,LOAD_CANDIDATES)
    out['bearing_id']=out[idc].astype(str) if idc else out['__source'].map(lambda x:Path(x).stem)
    out['time_idx']=np.arange(len(out)) if tc is None else pd.to_numeric(out[tc],errors='coerce')
    if 'rul_raw' not in out.columns:
        out['rul_raw']=np.nan if rc is None else pd.to_numeric(out[rc],errors='coerce')
    else:
        out['rul_raw']=pd.to_numeric(out['rul_raw'],errors='coerce')
    if 'health_raw' not in out.columns:
        if hc is not None:
            out['health_raw']=pd.to_numeric(out[hc],errors='coerce')
    else:
        out['health_raw']=pd.to_numeric(out['health_raw'],errors='coerce')
    if 'speed' not in out.columns:
        out['speed']=0.0 if sc is None else pd.to_numeric(out[sc],errors='coerce')
    else:
        out['speed']=pd.to_numeric(out['speed'],errors='coerce')
    if 'load' not in out.columns:
        out['load']=0.0 if lc is None else pd.to_numeric(out[lc],errors='coerce')
    else:
        out['load']=pd.to_numeric(out['load'],errors='coerce')
    for c in out.columns:
        if c not in {'bearing_id','__source'}:
            try:
                out[c]=pd.to_numeric(out[c])
            except (TypeError, ValueError):
                pass
    return out

def choose_numeric_features(df):
    excluded={'time_idx','cycle','rul_raw','rul_norm','health','health_raw','anomaly','anomaly_raw','speed','load','bearing_id','unit','__source','split','rul_at_end'}
    feats=[c for c in df.select_dtypes(include=[np.number]).columns if c not in excluded]
    if not feats: raise RuntimeError('No model input features remain after target/ID filtering.')
    return feats

def condition_columns(df,dataset_name):
    if dataset_name.lower().startswith('cmapss'):
        cols=[c for c in ('setting_1','setting_2','setting_3') if c in df.columns]
        if len(cols)!=3: raise ValueError(f'C-MAPSS condition columns missing: {cols}')
        return cols
    if 'speed' not in df.columns: df['speed']=0.0
    if 'load' not in df.columns: df['load']=0.0
    return ['speed','load']
