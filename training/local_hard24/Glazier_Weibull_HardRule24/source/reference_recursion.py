"""Deterministic reference functionals for the preserved, discrete SFSA.

This reproduces the *population limit of the legacy event-first KM/AJ
estimator*, not an assertion about independently censored latent event times.
Only NumPy and the standard library are required. It does not train models.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np

SCENARIOS = {
    'ReLU': dict(mean=70., sd=20., low=30., high=120., p=.0011),
    'Weibull': dict(mean=10., sd=1.5, low=7., high=13., p=.003),
}

def covariate_F(x, scenario):
    if scenario == 'ReLU':
        return .75 if x <= 70 else -.012*x+1.59
    return .2+1.5*(x/10)**4*np.exp(-(x/10)**5)

def base_matrix(x, scenario):
    f=covariate_F(x,scenario); p=SCENARIOS[scenario]['p']
    return np.array([
        [f-p/3-.001,(1-f)/2-p/3,(1-f)/2-p/3,.001,p],
        [2*f/3-p/3,(1-f)/3-p/3-.002,(2-f)/3-p/3,.002,p],
        [0,f/8-2*p/3+.306,.69-f/8-p/3,.004,p],
    ])

def state_space(kind, horizon=30):
    if kind=='base':
        return [(0,0),(1,0)]+[(2,r) for r in range(1,5)]
    return [(s,u) for s in range(3) for u in range(1,horizon+1)]

def kernels(x, scenario, kind, horizon=30, constant_regime=False):
    states=state_space(kind,horizon); index={z:i for i,z in enumerate(states)}
    initial=np.zeros(len(states))
    for s,p in enumerate([2/3,1/4,1/12]):
        initial[index[(s,1 if kind=='semi' or s==2 else 0)]]=p
    k=[]; baseline=base_matrix(x,scenario)
    for t in range(2,horizon+1):
        q=np.zeros((len(states),len(states))); death=np.zeros(len(states)); drop=np.zeros(len(states))
        ref=5 if constant_regime or t<20 else 3
        for i,(s,u) in enumerate(states):
            row=baseline[s].copy()
            if kind=='semi' and s in (1,2):
                donors=[0,1] if s==1 else [1,2]
                dest=2 if s==1 else 3
                lam,nu,cap=(.35,2.5,.90) if s==1 else (.97,3.,.99)
                omega=row[donors].sum(); extra=min(lam*(u/ref)**nu,cap,.95*omega)
                if omega>0: row[donors]*=(omega-extra)/omega
                row[dest]+=extra
            death[i]=row[3]; drop[i]=row[4]
            for dest in range(3):
                if kind=='base':
                    duration=(u+1 if s==2 else 1) if dest==2 else 0
                    if dest==2 and duration>=ref:
                        death[i]+=row[dest]; continue
                else:
                    duration=min(u+1,horizon) if dest==s else 1
                if kind=='semi' and t>=24 and s==2 and dest==2:
                    death[i]+=row[dest]
                    continue
                q[i,index[(dest,duration)]]+=row[dest]
        if not np.allclose(q.sum(1)+death+drop,1,atol=2e-14):
            raise AssertionError('probability conservation')
        if min(q.min(),death.min(),drop.min()) < -1e-14:
            raise AssertionError('negative probability')
        k.append((q,death,drop))
    return states,initial,k

def forward(initial,k):
    masses=[None,initial.copy()]; surv=[1.,1.]; cur=initial.copy()
    for q,d,c in k:
        risk=cur.sum(); event=cur@d
        surv.append(surv[-1]*(1-event/risk) if risk else surv[-1])
        cur=cur@q; masses.append(cur.copy())
    return masses,np.asarray(surv)

def conditional_curves(k,L,indices):
    n=k[0][0].shape[0]; cur=np.eye(n)[indices]; surv=np.ones(len(indices)); curves=[]
    for q,d,c in k[L-1:]:
        risk=cur.sum(1); ev=cur@d
        frac=np.divide(ev,risk,out=np.zeros_like(ev),where=risk>0)
        surv*=1-frac; curves.append(surv.copy()); cur=cur@q
    return np.asarray(curves).T

def weighted_median(y,w):
    order=np.argsort(y,axis=0,kind='stable')
    ys=np.take_along_axis(y,order,axis=0)
    ws=np.take_along_axis(np.broadcast_to(w[:,None],y.shape),order,axis=0)
    pos=np.argmax(np.cumsum(ws,axis=0)>=.5*w.sum(),axis=0)
    return ys[pos,np.arange(y.shape[1])]

def risks_at_x(x,scenario,kind,landmarks=(5,10,15,20,25),constant_regime=False):
    states,init,k=kernels(x,scenario,kind,constant_regime=constant_regime)
    masses,marginal=forward(init,k); results=[]
    for L in landmarks:
        idx=np.flatnonzero(masses[L]>0); weights=masses[L][idx]; eligible=weights.sum(); weights=weights/eligible
        curves=conditional_curves(k,L,idx); median=weighted_median(curves,weights)
        risk=(weights[:,None]*abs(curves-median)).sum(0)
        state_risk=np.zeros(curves.shape[1])
        for s in range(3):
            take=np.array([states[i][0]==s for i in idx])
            if take.any():
                med=weighted_median(curves[take],weights[take])
                state_risk+=(weights[take,None]*abs(curves[take]-med)).sum(0)
        ratio=marginal[L+1:]/marginal[L]
        ratio_risk=(weights[:,None]*abs(curves-ratio)).sum(0)
        results.append(dict(L=L,eligible=eligible,floor=float(risk.mean()),state_floor=float(state_risk.mean()),ratio=float(ratio_risk.mean()),classes=len(idx)))
    return results

def quadrature(scenario,n):
    s=SCENARIOS[scenario]; a,b=s['low'],s['high']
    # Split the ReLU kink so smooth quadrature is used on each interval.
    intervals=[(a,70.),(70.,b)] if scenario=='ReLU' else [(a,b)]
    z,w=np.polynomial.legendre.leggauss(n)
    xs=[]; ws=[]
    normalizer=.5*(math.erf((b-s['mean'])/(s['sd']*math.sqrt(2)))-math.erf((a-s['mean'])/(s['sd']*math.sqrt(2))))
    for lo,hi in intervals:
        x=lo+(z+1)*(hi-lo)/2
        density=np.exp(-.5*((x-s['mean'])/s['sd'])**2)/(s['sd']*math.sqrt(2*math.pi)*normalizer)
        xs.extend(x);ws.extend(w*(hi-lo)/2*density)
    return np.asarray(xs),np.asarray(ws)

def integrated(scenario,kind,n=64,constant_regime=False):
    xs,ws=quadrature(scenario,n); accum=np.zeros((5,5))
    for x,w in zip(xs,ws):
        for j,r in enumerate(risks_at_x(x,scenario,kind,constant_regime=constant_regime)):
            a=r['eligible'];accum[j]+=[w*a,w*a*r['floor'],w*a*r['state_floor'],w*a*r['ratio'],w*r['floor']]
    rows=[]
    for j,L in enumerate([5,10,15,20,25]):
        a=accum[j,0]
        rows.append(dict(scenario=scenario,generator=kind,landmark=L,eligible_probability=a,
                         history_blind_reference_floor=accum[j,1]/a,current_state_reference_floor=accum[j,2]/a,
                         ratio_reference_error=accum[j,3]/a,baseline_standardized_floor=accum[j,4]/ws.sum(),
                         quadrature_order=n,constant_regime=constant_regime))
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='reference_results');p.add_argument('--order',type=int,default=64)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for scenario in SCENARIOS:
        for kind in ['base','semi']:
            rows+=integrated(scenario,kind,args.order)
            print(scenario,kind,flush=True)
    with (out/f'reference_floors_{args.order}.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'estimand.json').write_text(json.dumps(dict(target='Population event-first KM reference functional of preserved observed generator; latent survival not asserted',population='alive and uncensored at landmark',horizons='L+1 through 30 equally weighted',classes='all positive-probability reachable augmented states',outer_integration='Gauss-Legendre with truncated-normal density; ReLU interval split at 70'),indent=2))
if __name__=='__main__':main()
