"""Fitted multinomial Markov comparator for CR3, with exact propagation.

Uses observed training transitions, including censoring as an explicit outcome.
No true transition coefficients, covariate transforms or oracle labels are used.
"""
import csv
import json
import numpy as np
from investigate_cox import BASE, OUT, SPECS, check_curves


def design(x, degree):
    z=np.column_stack([np.ones(len(x)),x])
    if degree==2:
        z=np.column_stack([z,x*x,x[:,0]*x[:,1],x[:,0]*x[:,2],x[:,1]*x[:,2]])
    return z


def probabilities(x,beta):
    eta=np.column_stack([x@beta,np.zeros(len(x))])
    eta-=eta.max(1,keepdims=True)
    p=np.exp(eta)
    return p/p.sum(1,keepdims=True)


def objective(beta,x,counts,ridge,hessian=True):
    p=probabilities(x,beta)
    totals=counts.sum(1)
    loss=-(counts*np.log(np.maximum(p,1e-300))).sum()+.5*ridge*np.sum(beta*beta)
    g=x.T@(totals[:,None]*p[:,:-1]-counts[:,:-1])+ridge*beta
    if not hessian:return loss,g,None
    k=beta.shape[1];d=x.shape[1]
    h=np.zeros((d,k,d,k))
    for a in range(k):
        for b in range(k):
            w=totals*p[:,a]*((a==b)-p[:,b])
            h[:,a,:,b]=x.T@(w[:,None]*x)
    h=h.reshape(beta.size,beta.size)+ridge*np.eye(beta.size)
    return loss,g,h


def fit(x,counts,ridge=.01):
    active=counts.sum(1)>0
    x=x[active];counts=counts[active]
    labels=np.flatnonzero(counts.sum(0)>0)
    counts=counts[:,labels]
    beta=np.zeros((x.shape[1],len(labels)-1))
    converged=False
    for it in range(60):
        loss,g,h=objective(beta,x,counts,ridge)
        step=np.linalg.solve(h,g.ravel()).reshape(beta.shape)
        rate=1.
        while rate>1e-8:
            candidate=beta-rate*step
            newloss=objective(candidate,x,counts,ridge,False)[0]
            if newloss<=loss-1e-4*rate*np.sum(g*step):break
            rate*=.5
        if rate<=1e-8:
            converged=np.max(np.abs(g))<1e-4
            break
        beta=candidate
        if np.max(np.abs(rate*step))<1e-7:
            converged=True
            break
    if not converged:raise RuntimeError('Multinomial fit did not converge')
    print('multinomial fit',len(x),'patients',len(labels),'labels',it+1,'iterations',flush=True)
    return labels,beta


def expanded(x,model):
    labels,beta=model
    out=np.zeros((len(x),10))
    out[:,labels]=probabilities(x,beta)
    return out


def transition_counts(states):
    # Aggregate repeated transitions within patient/source state without changing
    # the multinomial likelihood. Absorbing repeats are never training outcomes.
    n=len(states)
    counts=np.zeros((n,8,10))
    counts[np.arange(n),7,states[:,0]]=1
    for j in range(1,states.shape[1]):
        ids=np.flatnonzero(states[:,j-1]<7)
        np.add.at(counts,(ids,states[ids,j-1],states[ids,j]),1)
    return counts


def propagate(x,models,state,L):
    q=np.stack([expanded(x,models[s]) for s in range(7)],axis=1)
    mass=np.zeros((len(x),7))
    if L:mass[np.arange(len(x)),state]=1
    surv=np.ones(len(x));cif=np.zeros((len(x),2));curves=[]
    for time in range(L+1,31):
        nxt=expanded(x,models[7]) if time==1 else np.einsum('ns,nsd->nd',mass,q)
        risk=nxt.sum(1)
        death=np.divide(nxt[:,7:9],risk[:,None],out=np.zeros((len(x),2)),where=risk[:,None]>0)
        cif+=surv[:,None]*death
        surv*=1-death.sum(1)
        # Event-first KM/AJ target: deaths counted in same-step risk set before
        # removing censoring; propagate the remaining observed live-state mass.
        mass=nxt[:,:7]
        curves.append(np.column_stack([surv,cif]))
    return np.stack(curves,axis=-1)


def main():
    # Derivative checks are independent of the fitted benchmark data.
    rng=np.random.default_rng(123)
    tx=np.column_stack([np.ones(20),rng.normal(size=(20,2))])
    tc=rng.integers(0,5,(20,3));tb=rng.normal(size=(3,2))*.1
    _,g,h=objective(tb,tx,tc,.01);eps=1e-5
    for j in range(tb.size):
        delta=np.zeros_like(tb);delta.flat[j]=eps
        lp,gp,_=objective(tb+delta,tx,tc,.01)
        lm,gm,_=objective(tb-delta,tx,tc,.01)
        np.testing.assert_allclose(g.flat[j],(lp-lm)/(2*eps),atol=1e-7)
        np.testing.assert_allclose(h[:,j],((gp-gm)/(2*eps)).ravel(),atol=1e-7)
    source=BASE/SPECS['CR3'][0]
    with np.load(source/'checkpoints/cohort.npz') as f:c={k:f[k] for k in f.files}
    mu=c['covariates'][c['train_index']].mean(0);sd=c['covariates'][c['train_index']].std(0)
    x=(c['covariates']-mu)/sd
    counts=transition_counts(c['states'])
    candidates=[];selection=[]
    for degree in [1,2]:
        z=design(x,degree)
        models=[fit(z[c['train_index']],counts[c['train_index'],s]) for s in range(8)]
        val=c['val_index'];loss=0.;n=0
        for s in range(8):
            p=expanded(z[val],models[s]);observed=counts[val,s]
            loss-=float((observed*np.log(np.maximum(p,1e-300))).sum());n+=observed.sum()
        selection.append(dict(degree=degree,validation_logloss_per_transition=loss/n))
        candidates.append(models)
        print('degree',degree,'validation logloss',loss/n,flush=True)
    best=int(np.argmin([d['validation_logloss_per_transition'] for d in selection]))
    degree=[1,2][best];models=candidates[best];z=design(x,degree)
    arrays=dict(mu=mu,sd=sd,degree=np.array(degree))
    for s,(labels,beta) in enumerate(models):arrays[f'labels_{s}']=labels;arrays[f'beta_{s}']=beta
    np.savez_compressed(OUT/'CR3_multinomial_model.npz',**arrays)
    rows=[]
    for L in [0,10,20]:
        with np.load(source/f'evaluation/L{L}/reference.npz') as r:ids=r['patient_ids'];times=r['times'];truth=r['truth']
        p=propagate(z[ids],models,c['states'][ids,L-1] if L else None,L)
        check_curves(p)
        patient_mae=np.abs(p-truth).mean(-1)
        np.savez_compressed(OUT/f'CR3_markov_L{L}.npz',patient_ids=ids,times=times,prediction=p,truth=truth,patient_mae=patient_mae)
        for k,target in enumerate(['Survival','CIF_Death_A','CIF_Death_B']):
            rows.append(dict(experiment='CR3',model='Fitted-multinomial-Markov',target=target,landmark=L,n_patients=len(ids),MAE=float(patient_mae[:,k].mean())))
        print('CR3 Markov',L,patient_mae.mean(0),flush=True)
    with (OUT/'markov_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    (OUT/'markov_audit.json').write_text(json.dumps(dict(selection=selection,selected_degree=degree,
        training='train split only; per-current-state multinomial logistic observed transition likelihood',
        features='standardized baseline covariates; degree 2 adds squares and pairwise interactions',
        support='classes observed in training for each source state; no oracle transition graph supplied',
        initialization='separate fitted multinomial for first observed state at baseline',
        censoring='explicit censor outcome; event-first observed-process KM/AJ exact propagation',
        selection_rule='lowest held-out validation transition logloss; no oracle or test scoring used in selection',
        status='exploratory pilot, not yet incorporated in manuscript'),indent=2))


if __name__=='__main__':main()
