"""Synthetic two-disease SFSA. All coefficients are benchmark design choices.

No t=20/t=24 changes. Each step proposes a biological state, then independently
censors surviving patients. A same-step death is recorded before censoring.
"""
from dataclasses import dataclass, asdict
import numpy as np

STATES = ('Healthy','Sick_A','Sick_B','Sick_AB','Sicker_A','Sicker_B','Sicker_AB',
          'Death_A','Death_B','Censored')
H, A, B, AB, AA, BB, AABB, DA, DB, C = range(10)
N_LIVE, N_STATES, START = 7, 10, 10
COVARIATES = ('Calcium','Age','Drug')
TARGETS = ('Survival','CIF_Death_A','CIF_Death_B')


@dataclass(frozen=True)
class Scenario:
    horizon: int = 30
    censor_probability: float = .008
    calcium_min: float = 8.
    calcium_max: float = 13.
    age_min: float = 30.
    age_max: float = 90.
    drug_min: float = 0.
    drug_max: float = 1.
    version: str = 'cr_3cov_v1'

    def __post_init__(self):
        if self.horizon < 1 or not 0 <= self.censor_probability < 1:
            raise ValueError('Invalid horizon or censoring probability')
        if not (8 <= self.calcium_min < self.calcium_max <= 13 and
                30 <= self.age_min < self.age_max <= 90 and
                0 <= self.drug_min < self.drug_max <= 1):
            raise ValueError('Ranges must stay within the documented synthetic domain')

    def to_dict(self): return asdict(self)


def check_x(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim==1: x=x[None,:]
    if x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all():
        raise ValueError('Expected finite patient-by-3 covariates')
    if ((x < [8,30,0]) | (x > [13,90,1])).any():
        raise ValueError('Covariates outside the synthetic benchmark domain')
    return x


def effects(x):
    x=check_x(x)
    calcium=((x[:,0]-8.)/5.)**2.5
    age=np.maximum(x[:,1]-50.,0.)/40.
    return calcium,age,x[:,2]


def initial(x):
    """Distribution of s_1 given baseline covariates; no fictitious s_0."""
    ca,age,_=effects(x)
    p=np.zeros((len(ca),9))
    p[:,A]=.08+.10*ca; p[:,B]=.08+.10*age
    p[:,AB]=.025+.05*ca*age
    p[:,AA]=.01+.02*ca; p[:,BB]=.01+.02*age
    p[:,AABB]=.005+.015*ca*age
    p[:,H]=1-p.sum(1)
    return p


def transitions(x):
    """Biological transition rows: seven live states -> seven live/two death.

    Calcium acts through a bounded power response; age through ReLU. Drug
    moves self-transition mass to recovery, leaving other row entries fixed.
    These are synthetic effects, not clinical dose-response assumptions.
    """
    ca,age,drug=effects(x); n=len(ca)
    q=np.zeros((n,7,9),dtype=np.float64)
    q[:,H,A]=.02+.10*ca+.025*age
    q[:,H,B]=.02+.08*age+.01*ca
    q[:,H,AB]=.005+.02*ca*age
    q[:,A,H]=.08/(1+ca+.25*age)
    q[:,A,AA]=.04+.20*ca+.05*age
    q[:,A,AB]=.025+.065*age
    q[:,B,H]=.08/(1+age+.25*ca)
    q[:,B,BB]=.04+.18*age+.04*ca
    q[:,B,AB]=.025+.06*ca
    q[:,AB,A]=.025; q[:,AB,B]=.025
    q[:,AB,AABB]=.05+.15*ca+.12*age
    q[:,AA,A]=.09
    q[:,AA,AABB]=.025+.055*age
    q[:,AA,DA]=.025+.14*ca+.08*age
    q[:,BB,B]=.06+.35*drug
    q[:,BB,AABB]=.025+.055*ca
    q[:,BB,DB]=.02+.10*age+.03*ca
    q[:,AABB,AB]=.04+.30*drug
    q[:,AABB,DB]=.02+.06*age+.025*ca
    q[:,AABB,DA]=1.8*q[:,AABB,DB]+.03*ca+.015
    for s in range(7):q[:,s,s]=1-q[:,s,:].sum(1)
    if (q < 0).any():raise ArithmeticError('Negative transition mass')
    return q


def observed(p,censor_probability):
    """Independent censor removal after the biological transition/death."""
    out=np.zeros((*p.shape[:-1],10),dtype=np.float64)
    out[...,:7]=p[...,:7]*(1-censor_probability)
    out[...,7:9]=p[...,7:9]
    out[...,9]=p[...,:7].sum(-1)*censor_probability
    return out


def generate(n=50000,seed=34001,scenario=None):
    scenario=scenario or Scenario()
    if n<100:raise ValueError('At least 100 patients required')
    rng=np.random.default_rng(seed)
    x=np.column_stack([rng.uniform(getattr(scenario,k+'_min'),getattr(scenario,k+'_max'),n)
                       for k in ('calcium','age','drug')])
    q=observed(transitions(x),scenario.censor_probability)
    first=observed(initial(x),scenario.censor_probability)
    states=np.empty((n,scenario.horizon),dtype=np.int64)
    previous=np.full(n,START,dtype=np.int64)
    duration=np.full(n,scenario.horizon,dtype=np.int64)
    cause=np.zeros(n,dtype=np.int64)
    for j in range(scenario.horizon):
        active=previous<7 if j else np.ones(n,dtype=bool)
        ids=np.flatnonzero(active)
        p=first[ids] if j==0 else q[ids,previous[ids]]
        nxt=np.minimum((rng.random((len(ids),1))>=p.cumsum(1)).sum(1),9)
        previous[ids]=nxt;states[:,j]=previous
        ended=ids[nxt>=7]
        duration[ended]=j+1
        cause[ended]=np.where(previous[ended]==7,1,np.where(previous[ended]==8,2,0))
    order=np.random.default_rng(seed+1).permutation(n)
    return dict(covariates=x,states=states,duration=duration,cause=cause,
                train_index=order[:int(.8*n)],val_index=order[int(.8*n):int(.9*n)],
                test_index=order[int(.9*n):])


def reference_curves(x,prefix=None,landmark=0,scenario=None):
    """Exact forward recursion for conditional [S,F_A,F_B]; no Monte Carlo.

    Returned grid is landmark+1,...,horizon. Independent censoring cancels from
    the event-first population KM/AJ hazard; the biological recursion is equal
    to that observed-process functional in this specifically defined generator.
    """
    scenario=scenario or Scenario();x=check_x(x);L=int(landmark)
    if L<0 or L>=scenario.horizon:raise ValueError('Invalid landmark')
    q=transitions(x); mass=np.zeros((len(x),9)); curves=[]
    if L:
        prefix=np.asarray(prefix)
        if prefix.shape!=(len(x),L) or not np.isin(prefix,np.arange(7)).all():
            raise ValueError('Supply exactly the observed live prefix')
        mass[np.arange(len(x)),prefix[:,-1]]=1.
    for t in range(L+1,scenario.horizon+1):
        if t==1:mass=initial(x)
        else:
            nxt=np.einsum('ns,nsd->nd',mass[:,:7],q)
            nxt[:,7:9]+=mass[:,7:9];mass=nxt
        curves.append(np.column_stack([mass[:,:7].sum(1),mass[:,7],mass[:,8]]))
    return np.stack(curves,axis=-1)


def validate_curves(curves,atol=2e-6):
    curves=np.asarray(curves)
    if curves.ndim not in (2,3) or curves.shape[-2]!=3 or not np.isfinite(curves).all():
        raise ValueError('Expected finite [...,3,time] survival/CIF curves')
    if curves.min() < -atol or curves.max()>1+atol:raise ValueError('Curve out of range')
    np.testing.assert_allclose(curves.sum(-2),1.,atol=atol,rtol=0)
    if (np.diff(curves[...,0,:],axis=-1)>atol).any() or (np.diff(curves[...,1:,:],axis=-1)<-atol).any():
        raise ValueError('Survival/CIF monotonicity violation')
