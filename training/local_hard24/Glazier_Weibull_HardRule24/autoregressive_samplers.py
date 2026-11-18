"""Alternative samplers for the learned five-state Glazier distribution.

No hard death rule is imposed on model predictions: that rule must be learned
from the training cohort. States 3 and 4 terminate an observed trajectory.
Weighted methods return a product-limit curve, not ordinary unweighted KM.
"""
import math
import numpy as np
import torch

METHODS=('iid','stratified','sobol','antithetic','importance',
         'weighted_branching','conditional_mc')

def uniforms(method,M,T,seed):
    rng=np.random.default_rng(seed)
    if method=='sobol':
        # Exact requested count, including M=10000. Powers-of-two balance
        # guarantees do not apply to this non-power-of-two prefix.
        return torch.quasirandom.SobolEngine(T,scramble=True,seed=int(seed)).draw(M).double().numpy()
    if method=='stratified':
        return np.stack([rng.permutation((np.arange(M)+rng.random(M))/M) for _ in range(T)],axis=1)
    if method=='antithetic':
        u=rng.random((math.ceil(M/2),T))
        return np.concatenate([u,1-u])[:M]
    return rng.random((M,T))

def weighted_km(duration,event,weights,T=30,L=0):
    """Frequency/importance weighted product-limit point estimator."""
    out=np.ones(T-L);s=1.
    for j,t in enumerate(range(L+1,T+1)):
        risk=weights[duration>=t].sum()
        deaths=weights[(duration==t)&event].sum()
        if risk>0:s*=max(0.,1-deaths/risk)
        out[j]=s
    return out

def _probabilities(model,x,hist,batch_size):
    pieces=[]
    for start in range(0,len(hist),batch_size):
        h=hist[start:start+batch_size]
        xx=x[start:start+batch_size]
        if hasattr(model,'next_state_logits'):logits=model.next_state_logits(h,xx)
        else:logits=model.forward_teacher_forced(h,xx)[:,-1,:]
        pieces.append(torch.softmax(logits,dim=-1).double().cpu().numpy())
    return np.concatenate(pieces)

def sample_curve(model,meta,x,M=10000,method='iid',seed=42,prefix=(),
                 batch_size=512,death_tilt=2.):
    if method not in METHODS:raise ValueError(method)
    if M<2 or death_tilt<=0:raise ValueError('Invalid budget or proposal tilt')
    prefix=np.asarray(prefix,dtype=np.int64)
    if np.any((prefix<0)|(prefix>=3)):raise ValueError('Prefix must be live and uncensored')
    L=len(prefix);T=int(meta['T_K']);device=meta['device']
    if L>=T:raise ValueError('No future horizons')
    hist=np.tile(np.r_[model.START_TOKEN,prefix],(M,1))
    hist=torch.tensor(hist,dtype=torch.long,device=device)
    xx=torch.full((M,),float((x-meta['x_mean'])/meta['x_sd']),device=device)
    u=uniforms(method,M,T-L,seed);rng=np.random.default_rng(seed+19)
    duration=np.full(M,T);event=np.zeros(M,dtype=bool);active=np.ones(M,dtype=bool)
    logweights=np.zeros(M);particle_weights=np.ones(M)/M
    survival=1.;curves=[];min_ess=float(M)
    model.eval()
    with torch.no_grad():
        for j,t in enumerate(range(L+1,T+1)):
            p=_probabilities(model,xx,hist,batch_size)
            if method in ('weighted_branching','conditional_mc'):
                hazard=float(particle_weights@p[:,3]);survival*=max(0.,1-hazard)
                curves.append(survival)
                live=p[:,:3].sum(1)
                if method=='weighted_branching':
                    # Enumerate all three live branches, then resample M
                    # histories with probabilities proportional to branch mass.
                    mass=(particle_weights[:,None]*p[:,:3]).reshape(-1)
                    if mass.sum()==0:curves.extend([survival]*(T-t));break
                    branch=rng.choice(M*3,size=M,p=mass/mass.sum())
                    parents=branch//3;nxt=branch%3
                    hist=hist[torch.tensor(parents,device=device)]
                    particle_weights=np.ones(M)/M
                else:
                    # Integrate death/censoring at each step and sample only
                    # the live transition; retain changing risk-set weights.
                    particle_weights*=live
                    if particle_weights.sum()==0:curves.extend([survival]*(T-t));break
                    particle_weights/=particle_weights.sum()
                    q=np.divide(p[:,:3],live[:,None],out=np.full((M,3),1/3),where=live[:,None]>0)
                    nxt=np.minimum((u[:,j,None]>np.cumsum(q,1)).sum(1),2)
                min_ess=min(min_ess,float(1/(particle_weights@particle_weights)))
            else:
                q=p.copy()
                if method=='importance':
                    q[:,3]*=death_tilt;q/=q.sum(1,keepdims=True)
                nxt=np.minimum((u[:,j,None]>np.cumsum(q,1)).sum(1),4)
                if method=='importance':
                    indices=np.arange(M)
                    logweights[active]+=np.log(p[indices[active],nxt[active]])-np.log(q[indices[active],nxt[active]])
                terminal=active&(nxt>=3)
                duration[terminal]=t;event[terminal]=nxt[terminal]==3
                # Later samples after absorption do not enter the likelihood
                # or observation; pin them to the terminal state.
                nxt[~active]=hist[:,-1].cpu().numpy()[~active]
                active[terminal]=False
            hist=torch.cat([hist,torch.tensor(nxt[:,None],device=device)],dim=1)
    if method in ('weighted_branching','conditional_mc'):
        return np.asarray(curves),dict(min_particle_ESS=min_ess,budget_type='M live particles; weighted product limit')
    weights=np.exp(logweights-logweights.max())
    ess=float(weights.sum()**2/(weights@weights))
    return weighted_km(duration,event,weights,T,L),dict(trajectory_ESS=ess,budget_type='M trajectories')
