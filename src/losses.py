from __future__ import annotations
import torch
import torch.nn.functional as F

def quantile_loss(pred,target,taus=(0.1,0.5,0.9)):
    target=target.unsqueeze(1).expand_as(pred); tau=torch.tensor(taus,dtype=pred.dtype,device=pred.device).view(1,-1); err=target-pred
    return torch.maximum(tau*err,(tau-1)*err).mean()
def machsense_loss(outputs,target_rul,target_health,target_x_last,*,lh=1.0,la=0.2,lm=0.0,lc=0.5):
    q=quantile_loss(outputs['rul'],target_rul); h=F.smooth_l1_loss(outputs['health'],target_health); recon=F.mse_loss(outputs['reconstruction'],target_x_last); consistency=F.smooth_l1_loss(outputs['health'],outputs['rul'][:,1].detach())
    total=q+lh*h+la*recon+lc*consistency
    return total,{'quantile':float(q.detach()),'health':float(h.detach()),'reconstruction':float(recon.detach()),'consistency':float(consistency.detach())}
