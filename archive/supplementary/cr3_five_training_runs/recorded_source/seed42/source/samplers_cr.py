"""Seven autoregressive rollout estimators for survival and two joint CIFs.

Unweighted/weighted Aalen-Johansen updates preserve S+F_A+F_B=1.
No disease graph, death preference, or covariate effect is imposed on learned
next-state logits. Each rollout uses its own full generated prefix.
"""
import math
import numpy as np
import torch
from sfsa_cr import Scenario,initial,transitions,observed,validate_curves

METHODS=('iid','stratified','sobol','antithetic','importance','weighted_branching','conditional_mc')


def uniforms(method,M,T,seed):
    rng=np.random.default_rng(seed)
    if method=='sobol':return torch.quasirandom.SobolEngine(T,scramble=True,seed=int(seed)).draw(M).double().numpy()
    if method=='stratified':return np.stack([rng.permutation((np.arange(M)+rng.random(M))/M) for _ in range(T)],axis=1)
    if method=='antithetic':
        u=rng.random((math.ceil(M/2),T));return np.concatenate([u,1-u])[:M]
    return rng.random((M,T))


def weighted_aj(duration,cause,weights,H=30,L=0):
    """Deaths precede same-time censor removal; empty risk sets carry curves."""
    duration=np.asarray(duration);cause=np.asarray(cause);weights=np.asarray(weights)
    if not (duration.shape==cause.shape==weights.shape) or (weights<0).any() or not np.isfinite(weights).all():
        raise ValueError('Invalid observation weights')
    s=1.;f=np.zeros(2);curves=[]
    for t in range(L+1,H+1):
        risk=weights[duration>=t].sum()
        if risk>0:
            h=np.array([weights[(duration==t)&(cause==k)].sum()/risk for k in (1,2)])
            f+=s*h;s*=max(0.,1-h.sum())
        curves.append([s,*f])
    out=np.asarray(curves).T;validate_curves(out);return out


class Oracle:
    """True observed generator, used only for sampling diagnostics/evaluation."""
    def __init__(self,scenario=None):self.scenario=scenario or Scenario()
    def probabilities(self,x,hist):
        if hist.shape[1]==1:return observed(initial(x),self.scenario.censor_probability)
        p=observed(transitions(x),self.scenario.censor_probability)
        last=hist[:,-1];result=p[np.arange(len(x)),last.clip(max=6)]
        terminal=last>=7
        if terminal.any():result[terminal]=np.eye(10)[last[terminal]]
        return result


def probabilities(model,x,hist,batch_size):
    if isinstance(model,Oracle):return model.probabilities(x,hist)
    device=next(model.parameters()).device;pieces=[]
    for start in range(0,len(hist),batch_size):
        h=torch.tensor(hist[start:start+batch_size],device=device,dtype=torch.long)
        xx=torch.tensor(x[start:start+batch_size],device=device,dtype=torch.float32)
        logits=model.next_state_logits(h,xx) if hasattr(model,'next_state_logits') else model.forward_teacher_forced(h,xx)[:,-1,:]
        pieces.append(logits.softmax(-1).double().cpu().numpy())
    return np.concatenate(pieces)


def sample_curves(model,meta,x,M=10000,method='iid',seed=42,prefix=(),batch_size=512,death_tilt=2.):
    if method not in METHODS or M<2 or death_tilt<=0:raise ValueError('Invalid sampler configuration')
    x=np.asarray(x,dtype=float);prefix=np.asarray(prefix,dtype=int)
    if x.shape!=(3,) or prefix.ndim!=1 or not np.isin(prefix,np.arange(7)).all():raise ValueError('Invalid covariates/prefix')
    H=int(meta['horizon']);L=len(prefix)
    if L>=H:raise ValueError('No future grid')
    hist=np.tile(np.r_[10,prefix],(M,1))
    xx=np.tile((x-np.asarray(meta['x_mean']))/np.asarray(meta['x_sd']),(M,1))
    u=uniforms(method,M,H-L,seed);rng=np.random.default_rng(seed+19)
    duration=np.full(M,H);cause=np.zeros(M,dtype=int);active=np.ones(M,dtype=bool)
    logweights=np.zeros(M);weights=np.ones(M)/M
    s=1.;f=np.zeros(2);curves=[];min_ess=float(M)
    if not isinstance(model,Oracle):model.eval()
    with torch.no_grad():
        for j,t in enumerate(range(L+1,H+1)):
            p=probabilities(model,xx,hist,batch_size)
            # Remove floating-point accumulation drift before inverse CDF.
            if not np.isfinite(p).all() or (p<0).any() or (p.sum(1)<=0).any():raise FloatingPointError('Invalid next-state distribution')
            p=p/p.sum(1,keepdims=True)
            if method in ('weighted_branching','conditional_mc'):
                h=weights@p[:,7:9];f+=s*h;s*=max(0.,1-h.sum());curves.append([s,*f])
                live=p[:,:7].sum(1);parent_mass=weights*live
                if parent_mass.sum()==0:
                    curves.extend([[s,*f]]*(H-t));break
                parent_mass/=parent_mass.sum()
                min_ess=min(min_ess,float(1/(parent_mass@parent_mass)))
                if method=='weighted_branching':
                    mass=(weights[:,None]*p[:,:7]).ravel();mass/=mass.sum()
                    branches=rng.choice(M*7,size=M,p=mass)
                    parents=branches//7;nxt=branches%7;hist=hist[parents]
                    weights=np.ones(M)/M
                else:
                    weights=parent_mass
                    q=np.divide(p[:,:7],live[:,None],out=np.full((M,7),1/7),where=live[:,None]>0)
                    cdf=q.cumsum(1);cdf[:,-1]=1.
                    nxt=(u[:,j,None]>=cdf).sum(1)
            else:
                q=p.copy()
                if method=='importance':q[:,7:9]*=death_tilt;q/=q.sum(1,keepdims=True)
                cdf=q.cumsum(1);cdf[:,-1]=1.
                nxt=(u[:,j,None]>=cdf).sum(1)
                if method=='importance':
                    indices=np.flatnonzero(active)
                    logweights[indices]+=np.log(p[indices,nxt[indices]])-np.log(q[indices,nxt[indices]])
                ending=active&(nxt>=7)
                duration[ending]=t;cause[ending]=np.where(nxt[ending]==7,1,np.where(nxt[ending]==8,2,0))
                nxt[~active]=hist[~active,-1];active[ending]=False
            hist=np.column_stack([hist,nxt])
    if method in ('weighted_branching','conditional_mc'):
        out=np.asarray(curves).T;validate_curves(out)
        return out,dict(min_ESS=min_ess,budget_type='M live particles; weighted AJ')
    weights=np.exp(logweights-logweights.max());ess=float(weights.sum()**2/(weights@weights))
    return weighted_aj(duration,cause,weights,H,L),dict(min_ESS=ess,budget_type='M complete observed trajectories; weighted AJ' if method=='importance' else 'M complete observed trajectories; AJ')
