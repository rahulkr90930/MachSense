from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from .features import standardize, choose_numeric_features, condition_columns
from .dataset import build_targets, SequenceDataset
from .model import MachSenseNet
from .losses import machsense_loss
from .utils import seed_everything, save_json
ROOT=Path(__file__).resolve().parents[1]

def load_config():
    with open(ROOT/'configs/default.yaml',encoding='utf-8') as f: return yaml.safe_load(f)

def split_for_training(df,dataset_name):
    d=standardize(df); name=dataset_name.lower()
    if name=='femto' and 'split' in d.columns:
        train=d[d['split'].astype(str).str.lower().eq('train')].copy()
        if train.empty: raise ValueError('FEMTO training split is empty')
        return train
    if name.startswith('cmapss') and 'split' in d.columns:
        return d[d['split'].eq('train')].copy()
    return d

def train_machsense(df,dataset_name,epochs=40,batch_size=32,seed=42,window=None):
    seed_everything(seed); cfg=load_config(); cfg['window']=int(window or cfg['window'])
    d=split_for_training(df,dataset_name); bearings=sorted(d.bearing_id.astype(str).unique())
    if len(bearings)<2: raise ValueError(f'Need at least 2 independent runs; found {len(bearings)}')
    val_bid=bearings[-1]; train_df=d[d.bearing_id.astype(str)!=val_bid].copy(); val_df=d[d.bearing_id.astype(str)==val_bid].copy()
    scale=float(train_df.groupby('bearing_id')['rul_raw'].max().max())
    train_df=build_targets(train_df,scale); val_df=build_targets(val_df,scale)
    merged=pd.concat([train_df,val_df],ignore_index=True); features=choose_numeric_features(merged); cond_cols=condition_columns(train_df,dataset_name)
    train_ds=SequenceDataset(train_df,features,cond_cols,cfg['window'],cfg['stride'])
    val_ds=SequenceDataset(val_df,features,cond_cols,cfg['window'],cfg['stride'],fit_stats=(train_ds.mean,train_ds.std),fit_cond_stats=(train_ds.cond_mean,train_ds.cond_std))
    if not train_ds or not val_ds: raise RuntimeError('Not enough sequential samples; reduce WINDOW')
    train_dl=DataLoader(train_ds,batch_size=batch_size,shuffle=True); val_dl=DataLoader(val_ds,batch_size=batch_size,shuffle=False)
    model=MachSenseNet(len(features),len(cond_cols),hidden=cfg['hidden_dim'],layers=cfg['layers'],dropout=cfg['dropout'],cond_dim=cfg['condition_dim'])
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    ckpt_dir=ROOT/'artifacts'/'checkpoints'; ckpt_dir.mkdir(parents=True,exist_ok=True); ckpt=ckpt_dir/f'machsense_{dataset_name}.pt'
    best=float('inf'); history=[]
    for ep in range(1,epochs+1):
        model.train(); total=0.0
        for b in train_dl:
            opt.zero_grad(); out=model(b['x'],b['condition']); loss,_=machsense_loss(out,b['rul'],b['health'],b['x_last'],lh=cfg['lambda_health'],la=cfg['lambda_anomaly'],lm=cfg['lambda_monotonic'],lc=cfg['lambda_consistency']); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); total+=loss.item()*len(b['rul'])
        model.eval(); vals=[]; ys=[]; ps=[]
        with torch.no_grad():
            for b in val_dl:
                out=model(b['x'],b['condition']); loss,_=machsense_loss(out,b['rul'],b['health'],b['x_last'],lh=cfg['lambda_health'],la=cfg['lambda_anomaly'],lm=cfg['lambda_monotonic'],lc=cfg['lambda_consistency']); vals.append(loss.item()*len(b['rul'])); ys.extend(b['rul'].numpy().tolist()); ps.extend(out['rul'][:,1].numpy().tolist())
        val_loss=sum(vals)/len(val_ds); rmse=float(np.sqrt(np.mean((np.asarray(ys)-np.asarray(ps))**2)))
        history.append({'epoch':ep,'train_loss':total/len(train_ds),'val_loss':val_loss,'val_rul_rmse_norm':rmse})
        if val_loss<best:
            best=val_loss; torch.save({'model_state':model.state_dict(),'features':features,'condition_cols':cond_cols,'feature_mean':train_ds.mean,'feature_std':train_ds.std,'cond_mean':train_ds.cond_mean,'cond_std':train_ds.cond_std,'config':cfg,'dataset':dataset_name,'validation_bearing':val_bid,'rul_scale':scale},ckpt)
        if ep==1 or ep%5==0 or ep==epochs: print(f'[{dataset_name}] epoch={ep:03d} train={total/len(train_ds):.4f} val={val_loss:.4f} normalized_RMSE={rmse:.4f}')
    state=torch.load(ckpt,map_location='cpu',weights_only=False); model.load_state_dict(state['model_state']); model.eval(); healthy_err=[]
    with torch.no_grad():
        for b in DataLoader(train_ds,batch_size=256,shuffle=False):
            keep=b['rul']>0.8
            if keep.any():
                out=model(b['x'],b['condition']); healthy_err.extend(torch.mean((out['reconstruction']-b['x_last'])**2,dim=1)[keep].numpy().tolist())
    state['anomaly_threshold']=float(np.quantile(healthy_err,0.95)) if healthy_err else 0.02; state['anomaly_scale']=float(max(np.std(healthy_err),1e-5)) if healthy_err else 0.01; torch.save(state,ckpt)
    metrics={'dataset':dataset_name,'validation_bearing':val_bid,'n_train_bearings':len(bearings)-1,'features':features,'condition_cols':cond_cols,'history':history,'best_val_loss':best,'rul_scale':scale,'checkpoint':str(ckpt)}
    save_json(metrics,ROOT/'artifacts'/f'metrics_{dataset_name}.json'); sample_path=export_validation_sample(val_df,state,dataset_name); return ckpt,metrics,sample_path

def export_validation_sample(val_df,checkpoint,dataset_name):
    cols=['bearing_id','time_idx']+checkpoint['condition_cols']+checkpoint['features']; sample=val_df[cols].copy().tail(160)
    p=ROOT/'artifacts'/'check_samples'; p.mkdir(parents=True,exist_ok=True); out=p/f'{dataset_name}_validation_input.csv'; sample.to_csv(out,index=False); return out

def load_checkpoint(dataset_name):
    path=ROOT/'artifacts'/'checkpoints'/f'machsense_{dataset_name}.pt'
    if not path.exists(): raise FileNotFoundError(f'No checkpoint: {path}')
    return path,torch.load(path,map_location='cpu',weights_only=False)

def predict_series(df,checkpoint,model=None):
    from .rules import anomaly_probability,lifecycle_stage,maintenance_action
    d=standardize(df).sort_values(['bearing_id','time_idx']).reset_index(drop=True)
    if len(d)==0: return pd.DataFrame()
    features=checkpoint['features']; cond_cols=checkpoint['condition_cols']
    missing=[c for c in features+cond_cols if c not in d.columns]
    if missing: raise ValueError(f'Missing model columns: {missing}')
    X=d[features].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float32)
    X=(X-np.asarray(checkpoint['feature_mean']))/(np.asarray(checkpoint['feature_std'])+1e-6)
    C=d[cond_cols].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float32)
    C=(C-np.asarray(checkpoint['cond_mean']))/(np.asarray(checkpoint['cond_std'])+1e-6)
    cfg=checkpoint['config']
    w=int(cfg['window'])
    if model is None:
        model=MachSenseNet(len(features),len(cond_cols),hidden=cfg['hidden_dim'],layers=cfg['layers'],dropout=cfg['dropout'],cond_dim=cfg['condition_dim'])
        model.load_state_dict(checkpoint['model_state'])
        model.eval()

    pad_len=max(0,w-len(d))
    if pad_len>0:
        X_padded=np.pad(X,((pad_len,0),(0,0)),mode='edge')
        C_padded=np.pad(C,((pad_len,0),(0,0)),mode='edge')
        indices=[len(d)-1]
    else:
        X_padded=X
        C_padded=C
        indices=list(range(w-1,len(d)))

    if not indices: return pd.DataFrame()

    batch_X=torch.stack([torch.from_numpy(X_padded[idx+pad_len-w+1:idx+pad_len+1]) for idx in indices])
    batch_C=torch.stack([torch.from_numpy(C_padded[idx+pad_len-w+1:idx+pad_len+1]) for idx in indices])

    rows=[]
    with torch.no_grad():
        out=model(batch_X,batch_C)
        health_arr=out['health'].squeeze(-1).numpy() if out['health'].ndim>1 else out['health'].numpy()
        rul_arr=out['rul'].numpy()
        rec_arr=torch.mean((out['reconstruction']-batch_X[:,-1,:])**2,dim=-1).numpy()
        thresh=checkpoint.get('anomaly_threshold',0.02)
        ascale=checkpoint.get('anomaly_scale',0.01)
        rscale=checkpoint.get('rul_scale',1.0)

        for i,end in enumerate(indices):
            h=float(health_arr[i])
            q=rul_arr[i]
            rec=float(rec_arr[i])
            a=anomaly_probability(rec,thresh,ascale)
            rows.append({
                'index':end,
                'bearing_id':str(d.loc[end,'bearing_id']),
                'time_idx':d.loc[end,'time_idx'],
                'health':100*h,
                'rul_p10':100*float(q[0]),
                'rul_p50':100*float(q[1]),
                'rul_p90':100*float(q[2]),
                'anomaly':100*a,
                'stage':lifecycle_stage(h,a),
                'action':maintenance_action(h,float(q[0]),float(q[1]),a),
                'rul_p50_native':float(q[1]*rscale)
            })
    return pd.DataFrame(rows)
