from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
class ConditionFiLM(nn.Module):
    def __init__(self,condition_dim,hidden):
        super().__init__(); self.net=nn.Sequential(nn.Linear(condition_dim,hidden),nn.GELU(),nn.Linear(hidden,hidden*2))
    def forward(self,c): return self.net(c)
class TemporalBlock(nn.Module):
    def __init__(self,channels,dilation,dropout):
        super().__init__(); pad=dilation; g=8 if channels>=8 else 1
        self.conv1=nn.Conv1d(channels,channels,3,padding=pad,dilation=dilation); self.conv2=nn.Conv1d(channels,channels,3,padding=pad,dilation=dilation)
        self.norm1=nn.GroupNorm(g,channels); self.norm2=nn.GroupNorm(g,channels); self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        y=self.dropout(F.gelu(self.norm1(self.conv1(x)))); y=self.dropout(F.gelu(self.norm2(self.conv2(y)))); return x+y
class MachSenseNet(nn.Module):
    def __init__(self,n_features,n_conditions,hidden=64,layers=3,dropout=0.15,cond_dim=16):
        super().__init__(); self.input_proj=nn.Sequential(nn.Linear(n_features,hidden),nn.LayerNorm(hidden),nn.GELU()); self.film=ConditionFiLM(n_conditions,hidden); self.cond_proj=nn.Sequential(nn.Linear(n_conditions,cond_dim),nn.LayerNorm(cond_dim),nn.GELU()); self.tcn=nn.ModuleList([TemporalBlock(hidden,2**i,dropout) for i in range(layers)]); self.pool=nn.Sequential(nn.Linear(hidden+cond_dim,hidden),nn.GELU(),nn.Dropout(dropout)); self.health=nn.Sequential(nn.Linear(hidden,1),nn.Sigmoid()); self.rul=nn.Sequential(nn.Linear(hidden,hidden//2),nn.GELU(),nn.Linear(hidden//2,3),nn.Sigmoid()); self.decoder=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,n_features))
    def forward(self,x,condition):
        h=self.input_proj(x); gamma,beta=self.film(condition).chunk(2,dim=-1); h=h*(1+0.1*torch.tanh(gamma))+0.1*torch.tanh(beta); h=h.transpose(1,2)
        for block in self.tcn: h=block(h)
        h=h.transpose(1,2); context=h[:,-1]; cond=self.cond_proj(condition[:,-1]); z=self.pool(torch.cat([context,cond],dim=-1)); return {'health':self.health(z).squeeze(-1),'rul':self.rul(z),'reconstruction':self.decoder(z),'latent':z}
