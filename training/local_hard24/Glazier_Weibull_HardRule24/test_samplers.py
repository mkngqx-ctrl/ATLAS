"""Compare all samplers with an exactly enumerated history-dependent model."""
import numpy as np
import torch
from autoregressive_samplers import METHODS,sample_curve,weighted_km

def probabilities(h):
    h=np.asarray(h)
    p=np.tile([.4,.25,.15,.12,.08],(len(h),1))
    odd=((h==1).sum(1)%2)==1
    p[odd]=[.15,.2,.3,.25,.1]
    if h.shape[1]>=4:
        mask=h[:,-1]==2
        p[mask,3]+=p[mask,2];p[mask,2]=0
    return p

class Toy(torch.nn.Module):
    START_TOKEN=5
    def next_state_logits(self,h,x):
        p=probabilities(h.cpu().numpy())
        return torch.tensor(np.log(p,where=p>0,out=np.full_like(p,-np.inf)),device=x.device,dtype=torch.float64)

def exact(prefix,T=6):
    histories=[([5]+list(prefix),1.)];s=1.;curve=[]
    for t in range(len(prefix)+1,T+1):
        risk=sum(w for h,w in histories);death=0.;nxt=[]
        for h,w in histories:
            p=probabilities([h])[0];death+=w*p[3]
            for state in range(3):
                if p[state]>0:nxt.append((h+[state],w*p[state]))
        s*=1-death/risk;curve.append(s);histories=nxt
    return np.array(curve)

def run_tests():
    records=[];meta=dict(T_K=6,device='cpu',x_mean=0.,x_sd=1.)
    for prefix in [(),(1,2)]:
        truth=exact(prefix)
        for method in METHODS:
            curve,diag=sample_curve(Toy(),meta,0.,M=10000,method=method,seed=1357,prefix=prefix)
            err=float(abs(curve-truth).max())
            assert err<.025,(method,prefix,err)
            assert (np.diff(curve)<=1e-10).all()
            records.append(dict(method=method,L=len(prefix),max_error=err))
    np.testing.assert_allclose(weighted_km(np.array([1,2,3]),np.array([True,False,True]),np.ones(3),3),[2/3,2/3,0])
    print('14 exact-enumeration sampler checks and weighted KM check passed')
    return records

if __name__=='__main__':run_tests()
