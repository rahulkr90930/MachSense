from __future__ import annotations
import json, shutil, subprocess, zipfile
from pathlib import Path
from urllib.request import Request, urlopen
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
ROOT=Path(__file__).resolve().parents[1]
PADERBORN_REPO='https://github.com/alireza-javanmardi/bearing-RUL.git'
PADERBORN_B01_URL='https://zenodo.org/records/10868257/files/B01.zip?download=1'
FEMTO_URL='https://phm-datasets.s3.amazonaws.com/NASA/10.%20FEMTO%20Bearing.zip'
CMAPSS_BASE_URL='https://raw.githubusercontent.com/PunVas/nasa-c-mapss/main'

def require_free_space(path,required_gb):
    free=shutil.disk_usage(path).free/1e9
    if free<required_gb: raise RuntimeError(f'Insufficient disk space: {free:.2f} GB free; need at least {required_gb:.2f} GB.')

def download_file(url,dest,chunk_size=1024*1024,retries=3):
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and dest.stat().st_size>0:
        print(f'[skip] {dest.name} ({dest.stat().st_size/1e9:.2f} GB)'); return dest
    last=None
    for attempt in range(1,retries+1):
        try:
            req=Request(url,headers={'User-Agent':'MachSense/1.0'})
            with urlopen(req,timeout=60) as r:
                total=int(r.headers.get('Content-Length') or 0)
                with open(dest,'wb') as f, tqdm(total=total,unit='B',unit_scale=True,desc=dest.name) as bar:
                    while True:
                        chunk=r.read(chunk_size)
                        if not chunk: break
                        f.write(chunk); bar.update(len(chunk))
            if dest.stat().st_size==0: raise RuntimeError('Downloaded zero-byte file')
            return dest
        except Exception as exc:
            last=exc
            if dest.exists(): dest.unlink()
            print(f'[retry {attempt}/{retries}] {exc}')
    raise RuntimeError(f'Failed to download {url}') from last

def safe_extract(zip_path,out_dir):
    out_dir.mkdir(parents=True,exist_ok=True); marker=out_dir/'.extracted'
    if marker.exists(): return out_dir
    with zipfile.ZipFile(zip_path) as zf:
        bad=zf.testzip()
        if bad: raise RuntimeError(f'Corrupt ZIP member: {bad}')
        root=out_dir.resolve()
        for info in zf.infolist():
            target=(out_dir/info.filename).resolve()
            if root not in target.parents and target!=root: raise RuntimeError(f'Unsafe ZIP path: {info.filename}')
        zf.extractall(out_dir)
    
    while True:
        zips = list(out_dir.rglob('*.zip'))
        if not zips: break
        for z in zips:
            try:
                with zipfile.ZipFile(z) as zf2:
                    zf2.extractall(z.parent)
            except Exception:
                pass
            z.unlink()
        
    marker.write_text('ok',encoding='utf-8'); return out_dir

def clone_paderborn(dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and not (dest/'.git').exists(): shutil.rmtree(dest)
    if not (dest/'.git').exists():
        try:
            subprocess.run(['git','clone','--depth','1',PADERBORN_REPO,str(dest)],check=True)
        except Exception as exc:
            archive=dest.parent/'bearing-RUL-main.zip'; tmp=dest.parent/'_paderborn_repo'
            download_file('https://github.com/alireza-javanmardi/bearing-RUL/archive/refs/heads/main.zip',archive)
            if tmp.exists(): shutil.rmtree(tmp)
            safe_extract(archive,tmp); roots=[p for p in tmp.iterdir() if p.is_dir()]
            if not roots: raise RuntimeError('Could not unpack Paderborn repository') from exc
            shutil.move(str(roots[0]),str(dest)); shutil.rmtree(tmp,ignore_errors=True)
    return dest

def file_stats(root):
    files=[p for p in Path(root).rglob('*') if p.is_file()]; total=sum(p.stat().st_size for p in files)
    return {'root':str(root),'files':len(files),'bytes':total,'gb':total/1e9}

def build_paderborn_features(pader_root, out_path=None, fft_bands_count=20):
    if out_path and Path(out_path).exists():
        print(f"[skip] Using existing features from {Path(out_path).name}")
        return pd.read_parquet(out_path)
    import re
    pader_root = Path(pader_root)
    ldm_root = pader_root / 'data' / 'LDM'
    if not ldm_root.exists(): ldm_root = pader_root / 'LDM'
    if not ldm_root.exists(): ldm_root = pader_root
    fft_dir = ldm_root / 'fft'
    op_dir = ldm_root / 'op'
    temp_dir = ldm_root / 'temp'
    if not (fft_dir.exists() and op_dir.exists() and temp_dir.exists()):
        raise FileNotFoundError(f'Missing LDM directories (fft, op, temp) under {ldm_root}')
    def get_bid(p):
        m = re.search(r'(B\d{2})', Path(p).stem.upper())
        return m.group(1) if m else None
    fft_files = {get_bid(p): p for p in fft_dir.glob('*.npy') if get_bid(p)}
    op_files = {get_bid(p): p for p in op_dir.glob('*.csv') if get_bid(p)}
    temp_files = {get_bid(p): p for p in temp_dir.glob('*.csv') if get_bid(p)}
    bearings = sorted(set(fft_files) & set(op_files) & set(temp_files))
    if len(bearings) < 2:
        raise ValueError(f'Fewer than 2 bearings found across fft/op/temp in {ldm_root}')
    frames = []
    for b in bearings:
        fft = np.load(fft_files[b]).astype(np.float32)
        op = pd.read_csv(op_files[b])
        temp = pd.read_csv(temp_files[b])
        if 'Time' in op.columns: op = op.drop(columns=['Time'])
        if 'Time' in temp.columns: temp = temp.drop(columns=['Time'])
        n = len(fft)
        if not (n == len(op) == len(temp)):
            raise ValueError(f'{b}: modality length mismatch: FFT={n}, OP={len(op)}, TEMP={len(temp)}')
        bins_per_band = fft.shape[1] // fft_bands_count
        fft_bands = np.abs(fft[:, :fft_bands_count * bins_per_band].reshape(n, fft_bands_count, bins_per_band)).mean(axis=2)
        fft_df = pd.DataFrame(fft_bands, columns=[f'fft_band_{i:02d}' for i in range(fft_bands_count)])
        temp_df = temp.copy()
        temp_df.columns = [f'temp_{i+1}' for i in range(temp.shape[1])]
        d = pd.concat([fft_df.reset_index(drop=True), op.reset_index(drop=True), temp_df.reset_index(drop=True)], axis=1)
        d['bearing_id'] = b
        d['time_idx'] = np.arange(n, dtype=np.int64)
        d['rul_raw'] = (n - 1 - d['time_idx']).astype(float)
        speed_cols = [c for c in op.columns if 'setspeed' in c.lower().replace(' ', '') or 'speed' in c.lower()]
        load_cols = [c for c in op.columns if 'setstatload' in c.lower().replace(' ', '') or 'load' in c.lower()]
        d['speed'] = pd.to_numeric(op[speed_cols[0]] if speed_cols else 0.0, errors='coerce')
        d['load'] = pd.to_numeric(op[load_cols[0]] if load_cols else 0.0, errors='coerce')
        frames.append(d)
    out_df = pd.concat(frames, ignore_index=True)
    out_df = out_df.sort_values(['bearing_id', 'time_idx']).reset_index(drop=True)
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(out_path, index=False)
    return out_df

def _rms_feature(x,prefix):
    x=np.asarray(x,dtype=float); x=x[np.isfinite(x)]
    if x.size<8: return {f'{prefix}_rms':0.0}
    x=x-x.mean(); return {f'{prefix}_rms':float(np.sqrt(np.mean(x*x)))}

def _read_acc_csv(path):
    d=pd.read_csv(path,header=None,sep=None,engine='python').apply(pd.to_numeric,errors='coerce').dropna(how='all')
    a=d.iloc[:,0].to_numpy(); b=d.iloc[:,1].to_numpy() if d.shape[1]>1 else a
    return a,b

def discover_femto_runs(root):
    runs=[]
    p_root = Path(root)
    has_full_test = any('full_test_set' in [p.name.lower() for p in d.parents] for d in p_root.rglob('Bearing*_*'))
    for d in sorted(p_root.rglob('Bearing*_*')):
        files=list(d.glob('acc_*.csv'))
        if not files: continue
        names={p.name.lower() for p in d.parents}
        if has_full_test and 'test_set' in names and 'full_test_set' not in names:
            continue
        split='train' if ('learning_set' in names or 'training_set' in names) else ('test' if any(x in names for x in ('test_set','validation_set','full_test_set')) else 'unknown')
        runs.append((d.name,d,split))
    return runs

def _femto_condition(bid):
    if str(bid).startswith('Bearing1'): return 1800.0,4000.0
    if str(bid).startswith('Bearing2'): return 1650.0,4200.0
    if str(bid).startswith('Bearing3'): return 1500.0,5000.0
    return 0.0,0.0

def build_femto_features(root,out_path,max_bearings=None,training_only=True):
    if out_path and Path(out_path).exists() and not max_bearings:
        print(f"[skip] Using existing features from {Path(out_path).name}")
        d = pd.read_parquet(out_path)
        return d[d['split'] == 'train'].copy() if (training_only and 'split' in d.columns) else d
    runs=discover_femto_runs(root)
    if max_bearings: runs=runs[:max_bearings]
    if not runs: raise FileNotFoundError(f'No FEMTO bearing folders under {root}')
    rows=[]
    for bid,run,split in tqdm(runs,desc='FEMTO feature extraction'):
        files=sorted(run.glob('acc_*.csv')); speed,load=_femto_condition(bid); n=len(files)
        for i,p in enumerate(files):
            h,v=_read_acc_csv(p)
            rows.append({'bearing_id':bid,'time_idx':i,'split':split,'rul_raw':float(n-1-i) if split=='train' else np.nan,'speed':speed,'load':load,**_rms_feature(h,'h'),**_rms_feature(v,'v')})
    d=pd.DataFrame(rows); out_path.parent.mkdir(parents=True,exist_ok=True); d.to_parquet(out_path,index=False)
    return d[d['split'] == 'train'].copy() if (training_only and 'split' in d.columns) else d

def read_cmapss_file(path):
    raw=pd.read_csv(path,sep=r'\s+',header=None,engine='python').dropna(axis=1,how='all')
    if raw.shape[1]!=26: raise ValueError(f'Expected 26 columns in {path}, found {raw.shape[1]}')
    raw.columns=['unit','cycle','setting_1','setting_2','setting_3']+[f'sensor_{i}' for i in range(1,22)]
    return raw

def build_cmapss_fd001(raw_dir,out_train,out_test,out_rul):
    out_train, out_test, out_rul = Path(out_train), Path(out_test), Path(out_rul)
    if out_train.exists() and out_test.exists() and out_rul.exists():
        print(f"[skip] Using existing C-MAPSS features from {out_train.name}")
        return pd.read_parquet(out_train), pd.read_parquet(out_test)
    tr=read_cmapss_file(raw_dir/'train_FD001.txt'); te=read_cmapss_file(raw_dir/'test_FD001.txt'); true=pd.read_csv(raw_dir/'RUL_FD001.txt',sep=r'\s+',header=None,names=['rul'],engine='python')
    tr['bearing_id']=tr['unit'].astype(str); tr['time_idx']=tr['cycle']; tr['split']='train'; tr['rul_raw']=tr.groupby('unit')['cycle'].transform('max')-tr['cycle']
    te['bearing_id']='test_'+te['unit'].astype(str); te['time_idx']=te['cycle']; te['split']='test'; te['rul_raw']=np.nan
    if len(true)!=te['unit'].nunique(): raise ValueError('RUL_FD001 does not match test engine count')
    truth=dict(zip(sorted(te['unit'].unique()),true.rul.astype(float))); te['rul_at_end']=te['unit'].map(truth)
    train_out=tr.drop(columns=['unit']); test_out=te.drop(columns=['unit'])
    out_train.parent.mkdir(parents=True,exist_ok=True); train_out.to_parquet(out_train,index=False); test_out.to_parquet(out_test,index=False); true.to_csv(out_rul,index=False)
    return train_out,test_out

def download_cmapss_fd001(dest):
    dest.mkdir(parents=True,exist_ok=True)
    for name in ('train_FD001.txt','test_FD001.txt','RUL_FD001.txt'): download_file(f'{CMAPSS_BASE_URL}/{name}',dest/name)

build_cmapss_features = build_cmapss_fd001

def create_manifest(entries,path):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(entries,indent=2),encoding='utf-8')
