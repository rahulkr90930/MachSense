from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def build_targets(df,train_rul_scale=None):
    d=df.copy(); d['rul_raw']=pd.to_numeric(d.get('rul_raw',np.nan),errors='coerce')
    miss=d.groupby('bearing_id')['rul_raw'].transform(lambda s:s.notna().mean() if len(s) else 0)<0.2
    if miss.any():
        inferred=d.groupby('bearing_id')['time_idx'].transform(lambda s:s.max()-s)
        d.loc[miss,'rul_raw']=inferred.loc[miss]
    scale=float(train_rul_scale if train_rul_scale is not None else max(d['rul_raw'].max(),1.0))
    d['rul_norm']=(d['rul_raw']/max(scale,1e-8)).clip(0,1)
    d['health']=d['rul_norm']
    return d

class SequenceDataset(Dataset):
    def __init__(self,df,feature_cols,condition_cols,window=32,stride=4,fit_stats=None,fit_cond_stats=None):
        self.feature_cols=feature_cols; self.condition_cols=condition_cols; self.window=int(window); self.samples=[]
        d=df.sort_values(['bearing_id','time_idx']).reset_index(drop=True)
        Xraw=d[feature_cols].replace([np.inf,-np.inf],np.nan).to_numpy(np.float32)
        Xraw=np.nan_to_num(Xraw,nan=0,posinf=0,neginf=0)
        if fit_stats is None: self.mean=Xraw.mean(0); self.std=Xraw.std(0)+1e-6
        else: self.mean,self.std=fit_stats
        X=(Xraw-self.mean)/self.std
        Craw=d[condition_cols].apply(pd.to_numeric,errors='coerce').to_numpy(np.float32); Craw=np.nan_to_num(Craw,nan=0,posinf=0,neginf=0)
        if fit_cond_stats is None: self.cond_mean=Craw.mean(0); self.cond_std=Craw.std(0)+1e-6
        else: self.cond_mean,self.cond_std=fit_cond_stats
        C=(Craw-self.cond_mean)/self.cond_std
        for bid,idxs in d.groupby('bearing_id',sort=False).groups.items():
            idxs=np.asarray(list(idxs),dtype=int); idxs=idxs[np.argsort(d.loc[idxs,'time_idx'].to_numpy())]
            for end in range(self.window-1,len(idxs),max(int(stride),1)):
                sel=idxs[end-self.window+1:end+1]; last=sel[-1]
                self.samples.append((X[sel],C[sel],float(d.loc[last,'rul_norm']),float(d.loc[last,'health']),str(bid),float(d.loc[last,'time_idx'])))
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        x,c,r,h,bid,t=self.samples[i]
        return {'x':torch.tensor(x,dtype=torch.float32),'condition':torch.tensor(c,dtype=torch.float32),'rul':torch.tensor(r,dtype=torch.float32),'health':torch.tensor(h,dtype=torch.float32),'x_last':torch.tensor(x[-1],dtype=torch.float32),'bearing_id':bid,'time_idx':torch.tensor(t,dtype=torch.float32)}
