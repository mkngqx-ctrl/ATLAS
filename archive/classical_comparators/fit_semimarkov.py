"""Calendar-time-dependent semi-Markov multinomial model for Hard24.

Selection uses observed validation transition log loss only. Test/oracle curves
are loaded only after the model has been selected and saved.
"""
import sys
from workspace_paths import DEPENDENCIES
sys.path.insert(0,str(DEPENDENCIES))
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import logsumexp
import investigate_cox as c

OUT=Path(__file__).resolve().parent/'semimarkov_results'


def design(x,u,t,rich):
    x,u,t=np.broadcast_arrays(x,u,t)
    x=x.ravel();u=u.ravel().astype(int);t=t.ravel().astype(int)
    n=len(x);i=np.arange(n);v=np.log1p(u)/np.log(31)
    base=np.column_stack([np.ones(n),x,x*x,x**3,x**4])
    # Calendar t is the source time; transition arrives at t+1. Reference
    # categories are t=1 and u=1. Every time step is available as a feature;
    # no generator-specific breakpoint is supplied.
    ut=sparse.csr_matrix((np.ones(n),(i,np.maximum(u-2,0))),shape=(n,29))
    tt=sparse.csr_matrix((np.ones(n),(i,np.maximum(t-2,0))),shape=(n,28))
    ut=sparse.diags((u>=2).astype(float))@ut
    tt=sparse.diags((t>=2).astype(float))@tt
    blocks=[sparse.csr_matrix(base),ut,tt]
    if rich:
        blocks += [sparse.csr_matrix(np.column_stack([x*v,x*x*v,x*v*v,x*x*v*v])),sparse.diags(v)@tt]
    return sparse.hstack(blocks,format='csr')


def transition_rows(cohort,ids):
    states=cohort['states'][ids]
    us=np.ones(len(ids),int);parts=[]
    for j in range(29):
        if j:us=np.where(states[:,j]==states[:,j-1],us+1,1)
        active=states[:,j]<3
        parts.append((ids[active],states[active,j],us[active].copy(),np.full(active.sum(),j+1),states[active,j+1]))
    return [np.concatenate([r[k] for r in parts]) for k in range(5)]


def fit(X,y,ridge=1.):
    labels=np.unique(y);encoded=np.searchsorted(labels,y);k=len(labels)-1;p=X.shape[1];n=len(y)
    # Reparameterize only for optimizer conditioning; penalty remains on the
    # original coefficients, so the statistical objective is unchanged.
    scale=1/np.sqrt(np.maximum(np.asarray(X.multiply(X).mean(axis=0)).ravel(),1e-4))
    scaled=X@sparse.diags(scale)
    def obj(flat):
        theta=flat.reshape(p,k);b=scale[:,None]*theta
        eta=np.column_stack([scaled@theta,np.zeros(n)])
        logp=eta-logsumexp(eta,axis=1,keepdims=True)
        loss=(-logp[np.arange(n),encoded].sum()+.5*ridge*np.sum(b*b))/n
        residual=np.exp(logp);residual[np.arange(n),encoded]-=1
        grad=(scaled.T@residual[:,:k]+ridge*scale[:,None]*b)/n
        return loss,np.asarray(grad).ravel()
    started=time.time()
    counter=[0]
    def progress(theta):
        counter[0]+=1
        if counter[0]%100==0:print('optimizer iteration',counter[0],'observations',n,flush=True)
    res=minimize(obj,np.zeros(p*k),jac=True,callback=progress,method='L-BFGS-B',options=dict(maxiter=3000,ftol=1e-13,gtol=1e-7,maxls=40,maxcor=50))
    g=float(np.max(np.abs(res.jac)))
    if not res.success or g>2e-6:raise RuntimeError(f'Fit failed: {res.message}; gradient {g}')
    info=dict(n_observations=n,n_features=p,n_classes=len(labels),iterations=int(res.nit),gradient_max_average=g,converged=bool(res.success),seconds=time.time()-started,ridge=ridge)
    print('FIT',info,flush=True)
    return labels,scale[:,None]*res.x.reshape(p,k),info


def predict(X,model):
    labels,beta,_=model
    eta=np.column_stack([X@beta,np.zeros(X.shape[0])]);logp=eta-logsumexp(eta,axis=1,keepdims=True)
    out=np.zeros((len(eta),5));out[:,labels]=np.exp(logp)
    return out


def propagate(x,initial,models,rich,states,us,L):
    n=len(x);mass=np.zeros((n,3,30));surv=np.ones(n);curves=[]
    if L:mass[np.arange(n),states,us-1]=1
    for dest_time in range(L+1,31):
        if dest_time==1:
            first=predict(sparse.csr_matrix(np.column_stack([np.ones(n),x,x*x,x**3,x**4])),initial)
            nxt=np.zeros_like(mass);nxt[:,:,0]=first[:,:3];death=first[:,3];drop=first[:,4]
            risk=first.sum(1)
        else:
            nxt=np.zeros_like(mass);death=np.zeros(n);drop=np.zeros(n);risk=mass.sum((1,2))
            # Only possible source durations are evaluated.
            for s in range(3):
                for duration in range(1,dest_time):
                    w=mass[:,s,duration-1]
                    if not np.any(w):continue
                    q=predict(design(x,np.full(n,duration),np.full(n,dest_time-1),rich),models[s])
                    death+=w*q[:,3];drop+=w*q[:,4]
                    for d in range(3):
                        new_duration=duration+1 if d==s else 1
                        nxt[:,d,new_duration-1]+=w*q[:,d]
            np.testing.assert_allclose(nxt.sum((1,2))+death+drop,risk,atol=1e-12,rtol=0)
        hazard=np.divide(death,risk,out=np.zeros(n),where=risk>0)
        surv*=1-hazard;mass=nxt;curves.append(surv.copy())
    return np.stack(curves,axis=-1)


def save_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    plan=dict(candidates=['additive: quartic covariate + categorical duration + categorical source time','interaction: additive + covariate-duration terms + time-by-log-duration'],
        penalty=1.,selection='minimum observed validation transition log loss over all transitions and first states',
        no_generator_breakpoints=True,no_oracle_selection=True,no_test_selection=True)
    (OUT/'analysis_plan.json').write_text(json.dumps(plan,indent=2))
    folder,landmarks,_,expected=c.SPECS['Hard24'];source=c.BASE/folder;path=source/'checkpoints/cohort.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
    with np.load(path) as z:cohort={k:z[k] for k in z.files}
    tr,va,te=[cohort[k] for k in ['train_index','val_index','test_index']]
    assert not (set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te))
    mu=cohort['covariate'][tr].mean();sd=cohort['covariate'][tr].std();x=(cohort['covariate']-mu)/sd
    train=transition_rows(cohort,tr);val=transition_rows(cohort,va)
    initial=fit(sparse.csr_matrix(np.column_stack([np.ones(len(tr)),x[tr],x[tr]**2,x[tr]**3,x[tr]**4])),cohort['states'][tr,0])
    initial_val=predict(sparse.csr_matrix(np.column_stack([np.ones(len(va)),x[va],x[va]**2,x[va]**3,x[va]**4])),initial)
    initial_loss=-np.log(np.maximum(initial_val[np.arange(len(va)),cohort['states'][va,0]],1e-300)).sum()
    candidates=[];selection=[]
    for rich in [False,True]:
        models=[];loss=initial_loss
        for s in range(3):
            ids,ss,u,t,y=train;mask=ss==s
            model=fit(design(x[ids[mask]],u[mask],t[mask],rich),y[mask]);models.append(model)
            ids,ss,u,t,y=val;mask=ss==s
            p=predict(design(x[ids[mask]],u[mask],t[mask],rich),model)
            loss-=np.log(np.maximum(p[np.arange(mask.sum()),y[mask]],1e-300)).sum()
        score=float(loss/(len(val[0])+len(va)))
        selection.append(dict(interactions=rich,validation_logloss=score,fit=[q[2] for q in models]));candidates.append(models)
        print('VALIDATION',rich,score,flush=True)
    chosen=int(np.argmin([q['validation_logloss'] for q in selection]));rich=bool(chosen);models=candidates[chosen]
    arrays=dict(mu=np.array(mu),sd=np.array(sd),interactions=np.array(rich))
    for s,model in enumerate(models+[initial]):arrays[f'labels_{s}']=model[0];arrays[f'beta_{s}']=model[1]
    np.savez_compressed(OUT/'selected_model.npz',**arrays)
    (OUT/'selection.json').write_text(json.dumps(dict(plan=plan,selection=selection,chosen=chosen,initial_fit=initial[2],cohort=str(path),cohort_sha256=expected),indent=2))
    # First test/oracle access is below the saved, frozen selection.
    scores={};metrics=[]
    for L in landmarks:
        with np.load(source/f'evaluation/L{L}/reference.npz') as r:ids=r['patient_ids'];truth=r['truth'];times=r['times']
        assert set(ids)<=set(te)
        states=cohort['states'][ids,L-1] if L else None
        us=c.run_length(cohort['states'][ids],L).astype(int) if L else None
        p=propagate(x[ids],initial,models,rich,states,us,L)
        c.check_curves(p[:,None,:]);assert p.shape==truth.shape
        errors=np.abs(p-truth).mean(1);score=float(errors.mean());scores[L]=score
        np.savez_compressed(OUT/f'L{L}.npz',patient_ids=ids,times=times,truth=truth,prediction=p,patient_mae=errors)
        metrics.append(dict(landmark=L,n_patients=len(ids),MAE=score));print('TEST',metrics[-1],flush=True)
    save_csv(OUT/'metrics.csv',metrics)
    summary=Path(__file__).resolve().parents[1]/'supplementary/hard24_nine_training_runs/hard24_verified_summary.csv'
    rows=[]
    for r in csv.DictReader(summary.open()):
        if r['target']!='Survival':continue
        L=int(r['landmark']);score=float(r['mean'])
        rows.append(dict(model=r['model'],method=r['method'],landmark=L,mean_MAE=score,SD=float(r['SD']),n_training=int(r['n_training']),semiMarkov_MAE=scores[L],semiMarkov_lower_mean=scores[L]<score,semiMarkov_lower_every_run=scores[L]<float(r['minimum'])))
    controls=Path(__file__).resolve().parents[1]/'results_20260919/full_hard24_add_history_v4_summary.csv'
    for r in csv.DictReader(controls.open()):
        if r['model']!='CPH' and not ('oracle' in r['model'].lower()):continue
        L=int(r['landmark']);score=float(r['MAE'])
        rows.append(dict(model=r['model'],method=r['method'],landmark=L,mean_MAE=score,SD=r['seed_SD'],n_training=1 if r['model']=='CPH' else 0,semiMarkov_MAE=scores[L],semiMarkov_lower_mean=scores[L]<score,semiMarkov_lower_every_run=''))
    # Also include classical pilots already fitted in this investigation.
    for fname,label in [('hard24_markov_results/metrics.csv','Fitted-Markov'),('pilot_results/metrics.csv','Landmark-CPH-supermodel')]:
        for r in csv.DictReader((OUT.parent/fname).open()):
            if r.get('experiment','Hard24')!='Hard24':continue
            L=int(r['landmark']);score=float(r['MAE'])
            rows.append(dict(model=label,method='exact',landmark=L,mean_MAE=score,SD='',n_training=1,semiMarkov_MAE=scores[L],semiMarkov_lower_mean=scores[L]<score,semiMarkov_lower_every_run=scores[L]<score))
    save_csv(OUT/'all_model_comparisons.csv',rows)
    print('Finished',len(rows),'comparison rows',flush=True)


if __name__=='__main__':main()
