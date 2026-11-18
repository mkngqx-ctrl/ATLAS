"""Refit the current-state-only multinomial Markov comparator on Hard24.

Same modeling approach as CR3; no run length or calendar time features. This
deliberately retains the Markov assumption and documents its misspecification.
"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import investigate_cox as c
import investigate_markov as m

OUT=Path(__file__).resolve().parent/'hard24_markov_results'


def design(x,degree):
    return np.column_stack([np.ones(len(x)),x] + ([x*x] if degree==2 else []))


def expanded(x,model):
    labels,beta=model
    p=np.zeros((len(x),5));p[:,labels]=m.probabilities(x,beta)
    return p


def counts(states):
    n=len(states);out=np.zeros((n,4,5))
    out[np.arange(n),3,states[:,0]]=1
    for j in range(1,30):
        ids=np.flatnonzero(states[:,j-1]<3)
        np.add.at(out,(ids,states[ids,j-1],states[ids,j]),1)
    return out


def propagate(x,models,state,L):
    q=np.stack([expanded(x,models[s]) for s in range(3)],axis=1)
    mass=np.zeros((len(x),3))
    if L:mass[np.arange(len(x)),state]=1
    survival=np.ones(len(x));curves=[]
    for time in range(L+1,31):
        nxt=expanded(x,models[3]) if time==1 else np.einsum('ns,nsd->nd',mass,q)
        risk=nxt.sum(1)
        hazard=np.divide(nxt[:,3],risk,out=np.zeros(len(x)),where=risk>0)
        survival*=1-hazard
        mass=nxt[:,:3]
        curves.append(survival.copy())
    return np.stack(curves,axis=-1)


def main():
    OUT.mkdir(exist_ok=True)
    folder,landmarks,nlive,expected=c.SPECS['Hard24'];source=c.BASE/folder
    path=source/'checkpoints/cohort.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
    with np.load(path) as f:cohort={k:f[k] for k in f.files}
    tr,va,te=[cohort[k] for k in ['train_index','val_index','test_index']]
    assert not (set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te))
    x=cohort['covariate'][:,None];mu=x[tr].mean(0);sd=x[tr].std(0);x=(x-mu)/sd
    observed=counts(cohort['states']);candidates=[];selection=[]
    for degree in [1,2]:
        z=design(x,degree)
        models=[m.fit(z[tr],observed[tr,s]) for s in range(4)]
        loss=sum(float(-(observed[va,s]*np.log(np.maximum(expanded(z[va],models[s]),1e-300))).sum()) for s in range(4))
        selection.append(dict(degree=degree,validation_logloss=loss/observed[va].sum()))
        candidates.append(models)
    chosen=int(np.argmin([r['validation_logloss'] for r in selection]))
    degree=selection[chosen]['degree'];models=candidates[chosen];z=design(x,degree)
    arrays=dict(mu=mu,sd=sd,degree=np.array(degree))
    for s,(labels,beta) in enumerate(models):arrays[f'labels_{s}']=labels;arrays[f'beta_{s}']=beta
    np.savez_compressed(OUT/'model.npz',**arrays)
    scores={};metric_rows=[]
    for L in landmarks:
        with np.load(source/f'evaluation/L{L}/reference.npz') as r:ids=r['patient_ids'];times=r['times'];truth=r['truth']
        assert set(ids)<=set(te)
        state=cohort['states'][ids,L-1] if L else None
        assert not L or (state<3).all()
        p=propagate(z[ids],models,state,L)
        c.check_curves(p[:,None,:]);assert p.shape==truth.shape
        error=np.abs(p-truth).mean(1);scores[L]=float(error.mean())
        np.savez_compressed(OUT/f'L{L}.npz',patient_ids=ids,times=times,truth=truth,prediction=p,patient_mae=error)
        metric_rows.append(dict(landmark=L,n_patients=len(ids),MAE=scores[L]))
    with (OUT/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=metric_rows[0]);w.writeheader();w.writerows(metric_rows)
    reference=Path(__file__).resolve().parents[1]/'supplementary/hard24_nine_training_runs/hard24_verified_summary.csv'
    with reference.open() as f:atlas=[r for r in csv.DictReader(f) if r['model'].startswith('ATLAS-') and r['target']=='Survival']
    rows=[]
    for r in atlas:
        L=int(r['landmark']);a=float(r['mean']);markov=scores[L]
        rows.append(dict(model=r['model'],sampler=r['method'],landmark=L,ATLAS_mean_MAE=a,ATLAS_SD=float(r['SD']),n_training=int(r['n_training']),Markov_MAE=markov,ATLAS_relative_reduction_pct=100*(markov-a)/markov,ATLAS_lower_mean=a<markov,ATLAS_worst_run_MAE=float(r['maximum']),ATLAS_every_run_lower=float(r['maximum'])<markov))
    assert len(rows)==84 and len({(r['model'],r['sampler']) for r in rows})==21
    with (OUT/'all_atlas_comparisons.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    audit=dict(cohort=str(path),cohort_sha256=expected,n_train=len(tr),n_val=len(va),n_test=len(te),
        specification='current-state-specific multinomial logistic; standardized covariate linear or quadratic; separate first-state regression; explicit censor outcome; exact event-first KM propagation',
        excludes='run length, calendar time, regime indicators, true transition probabilities',
        selection=selection,selected_degree=degree,penalty=.01,
        training_transition_counts_by_source=observed[tr].sum((0,2)).tolist(),
        fitting='two candidate specifications, four regressions each; deterministic full-batch Newton optimization',
        evaluation='original saved reference patients and exact curves; mean patient full-curve MAE',
        caveat='time-homogeneous Markov model is structurally misspecified for this duration-dependent, time-changing experiment',
        all_84_mean_comparisons_lower=all(r['ATLAS_lower_mean'] for r in rows),
        all_84_every_run_comparisons_lower=all(r['ATLAS_every_run_lower'] for r in rows))
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))
    print('METRICS',metric_rows)
    for L in landmarks:
        subset=[r for r in rows if r['landmark']==L]
        print('L',L,'ATLAS range',min(r['ATLAS_mean_MAE'] for r in subset),max(r['ATLAS_mean_MAE'] for r in subset),'all runs lower',all(r['ATLAS_every_run_lower'] for r in subset))


if __name__=='__main__':main()
