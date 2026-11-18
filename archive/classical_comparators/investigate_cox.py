"""Exploratory stratified landmark Cox supermodels; NumPy-only Breslow fitting.

Archived inputs are read only. No oracle curves are used in fitting or selection.
Landmarks use separate calendar-time baseline hazards; regression effects are
shared with linear landmark interactions. This is a pilot specification, not a
claim of correctly specified proportional hazards in these discrete benchmarks.
"""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

from workspace_paths import BENCHMARKS
BASE = BENCHMARKS
OUT = Path(__file__).resolve().parent / 'pilot_results'
SPECS = {
    'Hard24': ('Hard24/42', [0, 20, 23, 24], 3,
               '676d66bf67e67a52dba3be50cfe7925d3b49f8f5e9ec9876913ed45146e03c54'),
    'CR3': ('CR3/42', [0, 10, 20], 7,
            'db08a45363fd873f7d62d5736e49fbb86df4f199109d56a5b7d683c38d8f2c15'),
}


def run_length(states, landmark):
    if landmark == 0:
        return np.zeros(len(states))
    length = np.ones(len(states), dtype=float)
    same = np.ones(len(states), dtype=bool)
    for j in range(landmark - 2, -1, -1):
        same &= states[:, j] == states[:, landmark - 1]
        length += same
    return length


class Features:
    def __init__(self, c, n_live):
        x = c['covariates'] if 'covariates' in c else c['covariate'][:, None]
        self.x = x
        self.mu = x[c['train_index']].mean(0)
        self.sd = x[c['train_index']].std(0)
        self.n_live = n_live

    def raw(self, c, ids, landmark):
        x = (self.x[ids] - self.mu) / self.sd
        state = c['states'][ids, landmark - 1] if landmark else np.full(len(ids), -1)
        # State 0 is the reference among observed live states. At baseline all
        # state indicators are zero; baseline is a separate hazard stratum.
        indicators = np.column_stack([state == j for j in range(1, self.n_live)])
        length = np.log1p(run_length(c['states'][ids], landmark))[:, None]
        z = np.column_stack([x, x*x, indicators, length, indicators*length])
        return np.column_stack([z, z * (landmark / 30.)])


def strata(c, ids, landmarks, features, cause):
    result = []
    for landmark in landmarks:
        take = ids if landmark == 0 else ids[c['states'][ids, landmark-1] < features.n_live]
        t = c['duration'][take]
        e = (c['event'][take] == 1 if 'event' in c
             else (c['cause'][take] > 0 if cause == 0 else c['cause'][take] == cause))
        result.append(dict(landmark=landmark, ids=take, x=features.raw(c, take, landmark), t=t, e=e))
    return result


def objective(beta, data, ridge, hessian=True):
    p = len(beta)
    loss = .5*ridge*(beta@beta)
    grad = ridge*beta.copy()
    hess = ridge*np.eye(p)
    for d in data:
        x,t,e = d['x'],d['t'],d['e']
        eta = x@beta
        shift = eta.max()
        w = np.exp(eta-shift)
        # Group moments by observed exit time, then form reverse risk sums.
        s0 = np.bincount(t, weights=w, minlength=31)
        s1 = np.zeros((31,p))
        np.add.at(s1,t,w[:,None]*x)
        r0 = np.cumsum(s0[::-1])[::-1]
        r1 = np.cumsum(s1[::-1],axis=0)[::-1]
        if hessian:
            s2 = np.zeros((31,p,p))
            for time in np.unique(t):
                mask=t==time
                s2[time]=x[mask].T@(w[mask,None]*x[mask])
            r2=np.cumsum(s2[::-1],axis=0)[::-1]
        for time in np.unique(t[e]):
            event = e & (t==time)
            count = event.sum()
            mean = r1[time]/r0[time]
            loss += count*(np.log(r0[time])+shift)-eta[event].sum()
            grad += count*mean-x[event].sum(0)
            if hessian:
                hess += count*(r2[time]/r0[time]-np.outer(mean,mean))
    return loss,grad,hess


def fit(data, ridge=0.01):
    p=data[0]['x'].shape[1]
    beta=np.zeros(p)
    converged=False
    for it in range(60):
        loss,g,h=objective(beta,data,ridge)
        step=np.linalg.solve(h,g)
        rate=1.
        while rate>1e-8:
            candidate=beta-rate*step
            newloss=objective(candidate,data,ridge,False)[0]
            if newloss<=loss-1e-4*rate*(g@step):break
            rate*=.5
        if rate<=1e-8:
            if np.max(np.abs(g))<1e-4:converged=True
            break
        beta=candidate
        print('iteration',it,'loss',round(newloss,4),'step',round(float(np.max(np.abs(rate*step))),7),flush=True)
        if np.max(np.abs(rate*step))<1e-7:
            converged=True
            break
    loss,g,_=objective(beta,data,ridge)
    if not converged:raise RuntimeError(f'Cox fit failed convergence: gradient={np.max(np.abs(g))}')
    baselines={}
    for d in data:
        x,t,e=d['x'],d['t'],d['e']
        weights=np.exp(x@beta)
        risk=np.cumsum(np.bincount(t,weights=weights,minlength=31)[::-1])[::-1]
        deaths=np.bincount(t[e],minlength=31)
        hazard=np.divide(deaths,risk,out=np.zeros(31),where=risk>0)
        hazard[:d['landmark']+1]=0
        baselines[d['landmark']]=hazard
    return beta,baselines,dict(iterations=it+1,gradient_max=float(np.max(np.abs(g))),ridge=ridge,converged=converged)


def predict(fits, x, landmark, times):
    hazards=np.stack([np.exp(x@b)[:,None]*h[landmark][times][None,:] for b,h,_ in fits],axis=1)
    total=hazards.sum(1)
    surv=np.exp(-np.cumsum(total,axis=1))
    if len(fits)==1:return surv[:,None,:]
    before=np.column_stack([np.ones(len(x)),surv[:,:-1]])
    shares=np.divide(hazards,total[:,None,:],out=np.zeros_like(hazards),where=total[:,None,:]>0)
    cif=np.cumsum(before[:,None,:]*(-np.expm1(-total))[:,None,:]*shares,axis=-1)
    return np.concatenate([surv[:,None,:],cif],axis=1)


def check_curves(p):
    assert np.isfinite(p).all() and p.min()>=-1e-12 and p.max()<=1+1e-12
    assert (np.diff(p[:,0],axis=-1)<=1e-12).all()
    if p.shape[1]>1:
        assert (np.diff(p[:,1:],axis=-1)>=-1e-12).all()
        np.testing.assert_allclose(p.sum(1),1,atol=1e-12)


def self_test():
    # Breslow intercept-free KM analogue: two deaths at time 1 in risk set 4.
    d=[dict(x=np.zeros((4,1)),t=np.array([1,1,2,2]),e=np.array([1,1,0,1],bool),landmark=0)]
    b,h,info=fit(d)
    np.testing.assert_allclose(h[0][[1,2]],[.5,.5])
    p=predict([(b,h,info)],np.zeros((1,1)),0,np.array([1,2]))
    np.testing.assert_allclose(p[0,0],np.exp(-np.array([.5,1.])))
    # Compare analytic derivatives with finite differences on tied observations.
    d[0]['x']=np.array([[-1.],[.2],[.8],[1.3]])
    beta=np.array([.15]);eps=1e-5
    _,g,h=objective(beta,d,.01)
    numerical=(objective(beta+eps,d,.01)[0]-objective(beta-eps,d,.01)[0])/(2*eps)
    np.testing.assert_allclose(g,[numerical],atol=1e-7)
    numerical_h=(objective(beta+eps,d,.01)[1]-objective(beta-eps,d,.01)[1])/(2*eps)
    np.testing.assert_allclose(h[:,0],numerical_h,atol=1e-7)
    np.testing.assert_array_equal(run_length(np.array([[0,1,1,1],[0,0,1,2]]),4),[3,1])
    print('Self tests passed',flush=True)


def main():
    self_test()
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[];audit={}
    for name,(folder,landmarks,n_live,expected) in SPECS.items():
        source=BASE/folder;path=source/'checkpoints/cohort.npz'
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual==expected
        with np.load(path,allow_pickle=False) as f:c={k:f[k] for k in f.files}
        train,val,test=[set(c[k].tolist()) for k in ['train_index','val_index','test_index']]
        assert not (train&val or train&test or val&test)
        assert len(train|val|test)==50000
        features=Features(c,n_live)
        fits=[]
        for cause in ([0] if name=='Hard24' else [1,2]):
            print('Fitting',name,'cause',cause,flush=True)
            data=strata(c,c['train_index'],landmarks,features,cause)
            fits.append(fit(data))
        audit[name]=dict(source=str(source),cohort_sha256=actual,fit=[f[2] for f in fits],
                         n_train=len(train),n_validation=len(val),n_test=len(test),landmarks=landmarks,
                         test_not_used_for_training=True,oracle_not_used_for_training=True)
        model_arrays={'mu':features.mu,'sd':features.sd}
        for k,(beta,baselines,info) in enumerate(fits):
            model_arrays[f'beta_{k}']=beta
            for L,h in baselines.items():model_arrays[f'baseline_{k}_L{L}']=h
        np.savez_compressed(OUT/f'{name}_cox_model.npz',**model_arrays)
        for L in landmarks:
            refpath=source/f'evaluation/L{L}/reference.npz'
            with np.load(refpath) as r:ids=r['patient_ids'];times=r['times'];truth=r['truth']
            assert set(ids.tolist())<=test
            assert L==0 or (c['states'][ids,L-1]<n_live).all()
            if truth.ndim==2:truth=truth[:,None,:]
            x=features.raw(c,ids,L)
            p=predict(fits,x,L,times)
            check_curves(p)
            assert p.shape==truth.shape
            patient_mae=np.abs(p-truth).mean(-1)
            np.savez_compressed(OUT/f'{name}_L{L}.npz',patient_ids=ids,times=times,prediction=p,truth=truth,patient_mae=patient_mae)
            for k,target in enumerate(['Survival'] if name=='Hard24' else ['Survival','CIF_Death_A','CIF_Death_B']):
                rows.append(dict(experiment=name,model='Landmark-CPH-supermodel',target=target,landmark=L,n_patients=len(ids),MAE=float(patient_mae[:,k].mean())))
            print(name,L,patient_mae.mean(0),flush=True)
    with (OUT/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    audit['specification']=dict(ties='Breslow',time_scale='absolute calendar step, separate baseline by landmark',
        features='train-standardized baseline covariates and squares; current-state indicators; log1p(current run length); state by log1p(run length); all interacted with L/30',
        penalty='fixed 0.01 L2 on coefficients for numerical stability; no outcome-based tuning',
        competing_risks='cause-specific Cox; simultaneous hazard increments converted to coherent survival/CIF via exponential total hazard and proportional allocation',
        status='exploratory pilot; model specification needs review before manuscript incorporation',
        limitations='PH approximation; discrete tied event times; no validation tuning yet; no new-dataset uncertainty; not repeated neural training replicates')
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2))


if __name__=='__main__':main()
