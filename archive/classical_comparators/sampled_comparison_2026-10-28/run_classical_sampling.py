"""Matched ATLAS IID/Sobol sampling of fitted Markov/semi-Markov models.

Runs under local Python 3.14. All persistent outputs go to ATLAS_work.
"""
from pathlib import Path
import sys,os,csv,json,time,hashlib,argparse
ROOT=Path(__file__).resolve().parents[3]
CODE=ROOT/'archive/classical_comparators'
sys.path.insert(0,str(CODE))
from workspace_paths import DEPENDENCIES,BENCHMARKS
sys.path.insert(0,str(DEPENDENCIES))
import numpy as np
from scipy import sparse
from scipy.special import logsumexp
import torch
import fit_semimarkov as sm
import investigate_cox as co

OUT=CODE/'sampled_comparison_2026-10-28'
SEEDS=[41001,52001,63001,74001,85001]
M=10000
torch.set_num_threads(1)


def savejson(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def csvwrite(p,rows):
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def base_design(x):
    return sparse.csr_matrix(np.column_stack([np.ones(len(x)),x,x*x,x**3,x**4,x[:,0]*x[:,1],x[:,0]*x[:,2],x[:,1]*x[:,2]]))


def cr_design(x,u,t,rich):
    n=len(x);i=np.arange(n);u=np.asarray(u,dtype=int);t=np.asarray(t,dtype=int)
    v=np.log1p(u)/np.log(31)
    du=sparse.csr_matrix(((u>=2).astype(float),(i,np.maximum(u-2,0))),shape=(n,29))
    dt=sparse.csr_matrix(((t>=2).astype(float),(i,np.maximum(t-2,0))),shape=(n,28))
    blocks=[base_design(x),du,dt]
    if rich:
        blocks += [sparse.csr_matrix(np.column_stack([x*v[:,None],x*x*v[:,None],x*(v*v)[:,None],x*x*(v*v)[:,None]])),sparse.diags(v)@dt]
    return sparse.hstack(blocks,format='csr')


def probs(X,model,nstates):
    labels,beta=model[:2]
    eta=np.column_stack([X@beta,np.zeros(X.shape[0])])
    p=np.exp(eta-logsumexp(eta,axis=1,keepdims=True))
    out=np.zeros((len(p),nstates));out[:,labels]=p
    return out


def rows(c,ids,nlive):
    states=c['states'][ids];u=np.ones(len(ids),int);parts=[]
    for j in range(29):
        if j:u=np.where(states[:,j]==states[:,j-1],u+1,1)
        a=states[:,j]<nlive
        parts.append((ids[a],states[a,j],u[a].copy(),np.full(a.sum(),j+1),states[a,j+1]))
    return [np.concatenate([r[k] for r in parts]) for k in range(5)]


def fit_cr3_semi():
    dest=OUT/'CR3_semiMarkov_fit';dest.mkdir(exist_ok=True)
    if (dest/'selected_model.npz').exists():return
    savejson(dest/'analysis_plan.json',dict(candidates=['quartic baseline covariates plus pairwise terms, categorical duration and calendar time','add covariate-duration and calendar-time-duration interactions'],penalty=1.,selection='minimum observed validation transition log loss including initial observations; train split only for fitting; no test or oracle selection'))
    source=BENCHMARKS/'CR3/42';path=source/'checkpoints/cohort.npz'
    assert sha(path)==co.SPECS['CR3'][3]
    with np.load(path) as z:c={k:z[k] for k in z.files}
    tr,va=c['train_index'],c['val_index'];mu=c['covariates'][tr].mean(0);sd=c['covariates'][tr].std(0);x=(c['covariates']-mu)/sd
    train,val=rows(c,tr,7),rows(c,va,7)
    def cached_fit(X,y,name):
        p=dest/(name+'.npz');meta=dest/(name+'.json')
        if p.exists():
            with np.load(p) as z:return z['labels'],z['beta'],json.loads(meta.read_text())
        model=sm.fit(X,y,ridge=1.)
        np.savez_compressed(p,labels=model[0],beta=model[1]);savejson(meta,model[2]);return model
    initial=cached_fit(base_design(x[tr]),c['states'][tr,0],'initial')
    ip=probs(base_design(x[va]),initial,10)
    iloss=-np.log(np.maximum(ip[np.arange(len(va)),c['states'][va,0]],1e-300)).sum()
    candidates=[];selection=[]
    for rich in [False,True]:
        fits=[];loss=iloss
        for s in range(7):
            ids,ss,u,t,y=train;a=ss==s
            model=cached_fit(cr_design(x[ids[a]],u[a],t[a],rich),y[a],f'candidate_{int(rich)}_state_{s}');fits.append(model)
            ids,ss,u,t,y=val;a=ss==s
            p=probs(cr_design(x[ids[a]],u[a],t[a],rich),model,10)
            loss-=np.log(np.maximum(p[np.arange(a.sum()),y[a]],1e-300)).sum()
        score=float(loss/(len(va)+len(val[0])))
        selection.append(dict(interactions=rich,validation_logloss=score,fit=[q[2] for q in fits]));candidates.append(fits)
        print('CR3 semi-Markov VALIDATION',rich,score,flush=True)
    chosen=int(np.argmin([r['validation_logloss'] for r in selection]));models=candidates[chosen]
    arrays=dict(mu=mu,sd=sd,interactions=np.array(bool(chosen)))
    for s,model in enumerate(models+[initial]):arrays[f'labels_{s}']=model[0];arrays[f'beta_{s}']=model[1]
    np.savez_compressed(dest/'selected_model.npz',**arrays)
    savejson(dest/'selection.json',dict(chosen=chosen,selection=selection,initial_fit=initial[2],cohort_sha256=sha(path),n_train=len(tr),n_val=len(va),status='new three-covariate counterpart of earlier Hard24 semi-Markov model; validation selected before test scoring'))


class Model:
    def __init__(self,experiment,kind):
        self.exp=experiment;self.kind=kind;self.nlive=3 if experiment=='Hard24' else 7;self.nstates=5 if experiment=='Hard24' else 10
        if kind=='Markov':p=CODE/('hard24_markov_results/model.npz' if experiment=='Hard24' else 'pilot_results/CR3_multinomial_model.npz')
        else:p=CODE/'semimarkov_results/selected_model.npz' if experiment=='Hard24' else OUT/'CR3_semiMarkov_fit/selected_model.npz'
        self.path=p;self.hash=sha(p)
        with np.load(p) as z:self.data={k:z[k] for k in z.files}
        self.mu=self.data['mu'];self.sd=self.data['sd']
        self.models=[(self.data[f'labels_{s}'],self.data[f'beta_{s}']) for s in range(self.nlive+1)]
    def features(self,x,u=None,t=None):
        if self.kind=='Markov':
            degree=int(self.data['degree'])
            if self.exp=='Hard24':return np.column_stack([np.ones(len(x)),x]+([x*x] if degree==2 else []))
            import investigate_markov as markov
            return markov.design(x,degree)
        if u is None:
            return sparse.csr_matrix(np.column_stack([np.ones(len(x)),x,x*x,x**3,x**4])) if self.exp=='Hard24' else base_design(x)
        rich=bool(self.data['interactions'])
        return sm.design(x,u,t,rich) if self.exp=='Hard24' else cr_design(x,u,t,rich)
    def prepare(self,rawx):
        x=(rawx-self.mu)/self.sd;self.x=x;n=len(x)
        self.initial=probs(self.features(x),self.models[-1],self.nstates)
        if self.kind=='Markov':
            self.q=np.stack([probs(self.features(x),m,self.nstates) for m in self.models[:-1]],axis=1)
        else:
            self.q=np.zeros((n,30,self.nlive,30,self.nstates))
            for t in range(1,30):
                xx=np.repeat(x,t,axis=0);u=np.tile(np.arange(1,t+1),n);tt=np.full(n*t,t)
                X=self.features(xx,u,tt)
                for s in range(self.nlive):self.q[:,t,s,:t,:]=probs(X,self.models[s],self.nstates).reshape(n,t,self.nstates)
            print('Prepared transition table',self.exp,self.kind,flush=True)
    def exact(self,indices,L,state,duration):
        n=len(indices);mass=np.zeros((n,self.nlive,30));surv=np.ones(n);cif=np.zeros((n,1 if self.nlive==3 else 2));curves=[]
        if L:mass[np.arange(n),state,duration-1]=1
        for dest in range(L+1,31):
            if dest==1:
                p=self.initial[indices];nxt=np.zeros_like(mass);nxt[:,:,0]=p[:,:self.nlive];death=p[:,self.nlive:self.nlive+cif.shape[1]];drop=p[:,-1];risk=p.sum(1)
            else:
                t=dest-1;q=self.q[indices,t] if self.kind=='SemiMarkov' else self.q[indices,:,None,:]
                weights=mass[:,:,:,None]*q
                death=weights[:,:,:,self.nlive:self.nlive+cif.shape[1]].sum((1,2));drop=weights[:,:,:,-1].sum((1,2));risk=mass.sum((1,2));nxt=np.zeros_like(mass)
                for s in range(self.nlive):
                    for d in range(self.nlive):
                        if s==d:nxt[:,d,1:]+=weights[:,s,:-1,d]
                        else:nxt[:,d,0]+=weights[:,s,:,d].sum(1)
                np.testing.assert_allclose(nxt.sum((1,2))+death.sum(1)+drop,risk,atol=1e-12,rtol=0)
            h=np.divide(death,risk[:,None],out=np.zeros_like(death),where=risk[:,None]>0)
            cif+=surv[:,None]*h;surv*=1-h.sum(1);mass=nxt
            curves.append(surv[:,None].copy() if self.nlive==3 else np.column_stack([surv,cif]))
        result=np.stack(curves,axis=-1);co.check_curves(result);return result
    def sample(self,i,L,state,duration,uniforms):
        n=len(uniforms);live=np.ones(n,bool);states=np.full(n,state if L else 0,dtype=int);us=np.full(n,duration if L else 0,dtype=int)
        surv=1.;cif=np.zeros(1 if self.nlive==3 else 2);curves=[];risks=[];deaths=[];censors=[]
        for j,dest in enumerate(range(L+1,31)):
            ids=np.flatnonzero(live);risk=len(ids)
            if risk:
                if dest==1:cdf=np.cumsum(self.initial[i])[None,:]
                else:
                    p=self.q[i,dest-1,states[ids],us[ids]-1] if self.kind=='SemiMarkov' else self.q[i,states[ids]]
                    p=p/p.sum(1,keepdims=True);cdf=np.cumsum(p,axis=1)
                cdf[:,-1]=1.
                u=uniforms[ids,j,None]
                nxt=(u>cdf).sum(1) if self.exp=='Hard24' else (u>=cdf).sum(1)
                # Archived CR3 code treats any terminal non-death label as
                # censoring, including an out-of-support label when a float32
                # Sobol draw rounds to exactly 1.0 and inverse CDF uses >=.
                d=np.array([(nxt==self.nlive+k).sum() for k in range(len(cif))]);c=int((nxt>=self.nstates-1).sum())
                h=d/risk;cif+=surv*h;surv*=1-h.sum()
                us[ids]=np.where(nxt==states[ids],us[ids]+1,1);states[ids]=nxt;live[ids[nxt>=self.nlive]]=False
            else:d=np.zeros(len(cif),int);c=0
            curves.append([surv] if self.nlive==3 else [surv,*cif]);risks.append(risk);deaths.append(d);censors.append(c)
        return np.asarray(curves).T,np.array(risks),np.array(deaths).T,np.array(censors)


def child_seed(exp,seed,patient_id,L,j):
    parts=[seed,j] if exp=='Hard24' else [seed,int(patient_id),L]
    return int(np.random.SeedSequence(parts).generate_state(1)[0]%(2**31))


def uniforms(method,T,seed):
    if method=='iid':return np.random.default_rng(seed).random((M,T))
    return torch.quasirandom.SobolEngine(T,scramble=True,seed=int(seed)).draw(M).double().numpy()


def evaluate(experiment):
    folder,landmarks,_,expected=co.SPECS[experiment];source=BENCHMARKS/folder
    assert sha(source/'checkpoints/cohort.npz')==expected
    with np.load(source/'checkpoints/cohort.npz') as z:c={k:z[k] for k in z.files}
    with np.load(source/'evaluation/L0/reference.npz') as z:base_ids=z['patient_ids']
    rawx=c['covariate'][base_ids] if experiment=='Hard24' else c['covariates'][base_ids]
    models={kind:Model(experiment,kind) for kind in ['Markov','SemiMarkov']}
    for model in models.values():model.prepare(rawx)
    index={int(pid):j for j,pid in enumerate(base_ids)};metrics=[]
    for L in landmarks:
        dest=OUT/experiment/f'L{L}';dest.mkdir(parents=True,exist_ok=True)
        with np.load(source/f'evaluation/L{L}/reference.npz') as z:ids=z['patient_ids'];times=z['times'];truth=z['truth']
        if truth.ndim==2:truth=truth[:,None,:]
        assert set(ids)<=set(c['test_index'])
        indices=np.array([index[int(i)] for i in ids]);states=c['states'][ids,L-1] if L else np.zeros(len(ids),int);us=co.run_length(c['states'][ids],L).astype(int) if L else np.zeros(len(ids),int)
        exact={kind:model.exact(indices,L,states,us) for kind,model in models.items()}
        for kind,p in exact.items():np.savez_compressed(dest/f'{kind}_exact.npz',patient_ids=ids,times=times,truth=truth,prediction=p,patient_mae=np.abs(p-truth).mean(-1))
        for method in ['iid','sobol']:
            for seed in SEEDS:
                started=time.time();buffers={kind:[] for kind in models};riskbuf={kind:[] for kind in models};deathbuf={kind:[] for kind in models};censorbuf={kind:[] for kind in models};children=[]
                paths={kind:dest/f'{kind}_{method}_{seed}.npz' for kind in models}
                if not all(p.exists() for p in paths.values()):
                    for j,pid in enumerate(ids):
                        child=child_seed(experiment,seed,pid,L,j);children.append(child);u=uniforms(method,30-L,child)
                        for kind,model in models.items():
                            p,r,d,cc=model.sample(indices[j],L,states[j],us[j],u);buffers[kind].append(p);riskbuf[kind].append(r);deathbuf[kind].append(d);censorbuf[kind].append(cc)
                    for kind,path in paths.items():
                        p=np.array(buffers[kind]);co.check_curves(p)
                        np.savez_compressed(path,patient_ids=ids,times=times,truth=truth,prediction=p,patient_mae=np.abs(p-truth).mean(-1),patient_sampling_mae=np.abs(p-exact[kind]).mean(-1),risk_counts=np.array(riskbuf[kind]),death_counts=np.array(deathbuf[kind]),censor_counts=np.array(censorbuf[kind]),child_seeds=np.array(children),M=M,seed=seed,model_sha256=np.array(models[kind].hash))
                for kind,path in paths.items():
                    with np.load(path) as z:
                        np.testing.assert_array_equal(z['patient_ids'],ids);assert int(z['M'])==M;assert str(z['model_sha256'])==models[kind].hash
                        p=z['prediction'];err=p-truth;sample_err=np.abs(p-exact[kind]);co.check_curves(p)
                        for k,target in enumerate(['Survival'] if experiment=='Hard24' else ['Survival','CIF_Death_A','CIF_Death_B']):
                            metrics.append(dict(experiment=experiment,model=kind,method=method,seed=seed,landmark=L,target=target,n_patients=len(ids),M=M,MAE=float(np.abs(err[:,k]).mean()),RMSE=float(np.sqrt((err[:,k]**2).mean())),sampling_MAE_vs_exact=float(sample_err[:,k].mean()),exact_MAE=float(np.abs(exact[kind][:,k]-truth[:,k]).mean())))
                print('SAMPLED',experiment,'L',L,method,seed,'patients',len(ids),'seconds',round(time.time()-started,1),flush=True)
    csvwrite(OUT/f'{experiment}_replicate_metrics.csv',metrics)


def summarize():
    raw=[]
    for exp in ['Hard24','CR3']:raw+=list(csv.DictReader((OUT/f'{exp}_replicate_metrics.csv').open(encoding='utf-8-sig')))
    summaries=[]
    for key in sorted({tuple(r[k] for k in ['experiment','model','method','landmark','target']) for r in raw}):
        exp,model,method,L,target=key;rr=[r for r in raw if tuple(r[k] for k in ['experiment','model','method','landmark','target'])==key];a=np.array([float(r['MAE']) for r in rr]);assert len(rr)==5
        summaries.append(dict(experiment=exp,model=model,method=method,landmark=int(L),target=target,n_patients=int(rr[0]['n_patients']),M=M,n_sampling_seeds=5,mean_MAE=float(a.mean()),sampling_seed_SD=float(a.std(ddof=1)),mean_sampling_MAE_vs_exact=float(np.mean([float(r['sampling_MAE_vs_exact']) for r in rr])),exact_MAE=float(rr[0]['exact_MAE'])))
    csvwrite(OUT/'sampled_summary.csv',summaries)
    historical=[];data=CODE.parent
    for exp,path in [('Hard24',data/'supplementary/hard24_nine_training_runs/hard24_verified_summary.csv'),('CR3',data/'supplementary/cr3_five_training_runs/verified_summary.csv')]:
        for r in csv.DictReader(path.open()):
            if r['model'] in ['CPH','Oracle sampler']:continue
            historical.append(dict(experiment=exp,model=r['model'],method=r['method'],landmark=int(r['landmark']),target=r['target'],mean_MAE=float(r['mean'] if exp=='Hard24' else r['mean_MAE']),SD=float(r['SD'] if exp=='Hard24' else r['between_SD']),replicate_type='training runs',n_replicates=int(r['n_training'] if exp=='Hard24' else r['n_runs'])))
        record={'Hard24':'full_hard24_add_history_v4','CR3':'full_cr3cov_v2'}[exp];controls=data/f'results_20260919/{record}_summary.csv'
        for r in csv.DictReader(controls.open()):
            if r['model'] not in ['CPH','Oracle sampler']:continue
            historical.append(dict(experiment=exp,model=r['model'],method=r['method'],landmark=int(r['landmark']),target=r.get('target','Survival'),mean_MAE=float(r['MAE']),SD=r['seed_SD'],replicate_type='deterministic fit' if r['model']=='CPH' else 'oracle sampling seeds',n_replicates=int(r['repeats'])))
    for r in csv.DictReader((CODE/'pilot_results/metrics.csv').open()):historical.append(dict(experiment=r['experiment'],model=r['model'],method='exact',landmark=int(r['landmark']),target=r['target'],mean_MAE=float(r['MAE']),SD='',replicate_type='deterministic fit',n_replicates=1))
    for r in summaries:
        if r['method']=='iid':historical.append(dict(experiment=r['experiment'],model=r['model'],method='exact',landmark=r['landmark'],target=r['target'],mean_MAE=r['exact_MAE'],SD='',replicate_type='deterministic fit',n_replicates=1))
        historical.append(dict(experiment=r['experiment'],model=r['model'],method=r['method'],landmark=r['landmark'],target=r['target'],mean_MAE=r['mean_MAE'],SD=r['sampling_seed_SD'],replicate_type='sampling seeds on one fit',n_replicates=5))
    csvwrite(OUT/'all_model_scores.csv',historical)
    comparisons=[]
    for r in summaries:
        for h in historical:
            if (r['experiment'],r['landmark'],r['target'])!=(h['experiment'],h['landmark'],h['target']):continue
            if r['model']==h['model'] and r['method']==h['method']:continue
            comparisons.append(dict(experiment=r['experiment'],sampled_model=r['model'],sampled_method=r['method'],landmark=r['landmark'],target=r['target'],sampled_mean_MAE=r['mean_MAE'],comparator=h['model'],comparator_method=h['method'],comparator_mean_MAE=h['mean_MAE'],difference_sampled_minus_comparator=r['mean_MAE']-h['mean_MAE'],sampled_lower_mean=r['mean_MAE']<h['mean_MAE']))
    csvwrite(OUT/'all_pairwise_comparisons.csv',comparisons)
    print('SUMMARY rows',len(summaries),'all scores',len(historical),'comparisons',len(comparisons),flush=True)


def verify(report_path=None):
    checked=0
    for exp in ['Hard24','CR3']:
        for L in co.SPECS[exp][1]:
            folder=OUT/exp/f'L{L}'
            for p in folder.glob('*_*.npz'):
                with np.load(p) as z:
                    if 'risk_counts' not in z.files:continue
                    pred=z['prediction'];risk=z['risk_counts'];deaths=z['death_counts'];censor=z['censor_counts'];n=len(pred);surv=np.ones(n);cif=np.zeros((n,deaths.shape[1]));curves=[]
                    assert (risk[:,0]==M).all()
                    for j in range(risk.shape[1]):
                        hazard=np.divide(deaths[:,:,j],risk[:,j,None],out=np.zeros((n,deaths.shape[1])),where=risk[:,j,None]>0)
                        cif+=surv[:,None]*hazard;surv*=1-hazard.sum(1)
                        curves.append(surv[:,None].copy() if exp=='Hard24' else np.column_stack([surv,cif]))
                        if j<risk.shape[1]-1:np.testing.assert_array_equal(risk[:,j+1],risk[:,j]-deaths[:,:,j].sum(1)-censor[:,j])
                    np.testing.assert_allclose(pred,np.stack(curves,axis=-1),atol=1e-14,rtol=0)
                    np.testing.assert_allclose(z['patient_mae'],np.abs(pred-z['truth']).mean(-1),atol=1e-15,rtol=0);co.check_curves(pred)
                    kind=p.name.split('_')[0]
                    with np.load(folder/f'{kind}_exact.npz') as exact:
                        np.testing.assert_array_equal(z['patient_ids'],exact['patient_ids'])
                        np.testing.assert_allclose(z['patient_sampling_mae'],np.abs(pred-exact['prediction']).mean(-1),atol=1e-15,rtol=0)
                    expected_children=np.array([child_seed(exp,int(z['seed']),pid,L,j) for j,pid in enumerate(z['patient_ids'])])
                    np.testing.assert_array_equal(z['child_seeds'],expected_children)
                    checked+=1
    assert checked==140,checked
    savejson(report_path or OUT/'verification.json',dict(sampled_files_checked=checked,risk_event_censor_accounting=True,KM_AJ_reconstruction=True,patient_MAE_recomputed=True,sampling_MAE_vs_exact_recomputed=True,archived_child_seed_mapping_checked=True,curve_constraints=True))
    print('VERIFIED',checked,'sampled files',flush=True)


def repair_diagnostics():
    """Repair exact Hard24 views and recompute diagnostics without resampling.

    Sampling predictions and truth-based errors are unchanged. Prior exact
    curves come from the independently verified, original propagation routines.
    """
    rawpath=OUT/'Hard24_replicate_metrics.csv'
    raw=list(csv.DictReader(rawpath.open(encoding='utf-8-sig')))
    for L in [0,20,23,24]:
        folder=OUT/'Hard24'/f'L{L}'
        for kind,original in [('Markov',CODE/f'hard24_markov_results/L{L}.npz'),('SemiMarkov',CODE/f'semimarkov_results/L{L}.npz')]:
            with np.load(original) as z:oldids=z['patient_ids'];exact=z['prediction'][:,None,:]
            exactpath=folder/f'{kind}_exact.npz'
            with np.load(exactpath) as z:arrays={k:z[k] for k in z.files}
            np.testing.assert_array_equal(arrays['patient_ids'],oldids)
            arrays['prediction']=exact;arrays['patient_mae']=np.abs(exact-arrays['truth']).mean(-1)
            np.savez_compressed(exactpath,**arrays)
            for method in ['iid','sobol']:
                for seed in SEEDS:
                    path=folder/f'{kind}_{method}_{seed}.npz'
                    with np.load(path) as z:arrays={k:z[k] for k in z.files}
                    arrays['patient_sampling_mae']=np.abs(arrays['prediction']-exact).mean(-1)
                    np.savez_compressed(path,**arrays)
                    for r in raw:
                        if (r['model'],r['method'],int(r['seed']),int(r['landmark']))==(kind,method,seed,L):
                            r['sampling_MAE_vs_exact']=float(arrays['patient_sampling_mae'].mean());r['exact_MAE']=float(np.abs(exact-arrays['truth']).mean())
    csvwrite(rawpath,raw)
    savejson(OUT/'diagnostic_correction.json',dict(issue='Hard24 exact-survival arrays were stored as views and aliased the final survival value',correction='copy each time step; cross-check with prior independent exact outputs; recompute all exact-reference diagnostics',sampled_predictions_unchanged=True,benchmark_MAEs_unchanged=True))


def repair_sobol_boundary():
    """Replay only CR3 patient draws containing the rare rounded value 1.0."""
    source=BENCHMARKS/co.SPECS['CR3'][0]
    with np.load(source/'checkpoints/cohort.npz') as z:c={k:z[k] for k in z.files}
    with np.load(source/'evaluation/L0/reference.npz') as z:baseids=z['patient_ids']
    lookup={int(pid):i for i,pid in enumerate(baseids)}
    models={kind:Model('CR3',kind) for kind in ['Markov','SemiMarkov']}
    for model in models.values():model.prepare(c['covariates'][baseids])
    touched=[];boundary_patients=0;boundary_values=0
    for L in [0,10,20]:
        folder=OUT/'CR3'/f'L{L}'
        for seed in SEEDS:
            arrays={}
            for kind in models:
                with np.load(folder/f'{kind}_sobol_{seed}.npz') as z:arrays[kind]={k:z[k] for k in z.files}
            ids=arrays['Markov']['patient_ids'];changed=False
            for j,pid in enumerate(ids):
                child=child_seed('CR3',seed,pid,L,j);u=uniforms('sobol',30-L,child)
                n_boundary=int((u==1.).sum())
                if not n_boundary:continue
                boundary_patients+=1;boundary_values+=n_boundary
                state=int(c['states'][pid,L-1]) if L else 0
                duration=int(co.run_length(c['states'][pid:pid+1],L)[0]) if L else 0
                for kind,model in models.items():
                    prediction,risk,deaths,censor=model.sample(lookup[int(pid)],L,state,duration,u)
                    old=arrays[kind]
                    np.testing.assert_array_equal(prediction,old['prediction'][j])
                    np.testing.assert_array_equal(risk,old['risk_counts'][j])
                    np.testing.assert_array_equal(deaths,old['death_counts'][j])
                    previous=old['censor_counts'][j].copy();old['censor_counts'][j]=censor
                    if not np.array_equal(previous,censor):touched.append(dict(landmark=L,seed=seed,patient_id=int(pid),model=kind,extra_censor_count=int((censor-previous).sum())))
                changed=True
            if changed:
                for kind in models:np.savez_compressed(folder/f'{kind}_sobol_{seed}.npz',**arrays[kind])
            print('Sobol boundary scan',L,seed,flush=True)
    savejson(OUT/'sobol_boundary_audit.json',dict(patient_seed_draws_containing_rounded_one=boundary_patients,total_rounded_one_values=boundary_values,changed_records=touched,handling='match archived CR3: terminal non-death inverse-CDF outcomes are censored',sampled_predictions_and_MAEs_unchanged=True))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',default='all',choices=['all','fit','sample','summary','verify','repair','boundaryrepair']);args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.stage=='all':
        plan=dict(M=M,seeds=SEEDS,methods=['iid','scrambled Sobol'],uniform_implementation='NumPy default_rng; torch.quasirandom.SobolEngine(scramble=True).draw(10000).double()',child_seed_Hard24='SeedSequence([sampling_seed, eligible_patient_position]).generate_state(1)[0] mod 2^31',child_seed_CR3='SeedSequence([sampling_seed, patient_id, landmark]).generate_state(1)[0] mod 2^31',non_power_of_two='10000 matches archived ATLAS; power-of-two Sobol balance guarantee does not apply',output_semantics='average patient/time MAE within each seed then average five sampling seeds; no averaging curves before scoring',training_replicates='classical one fit per experiment; ATLAS nine Hard24/five CR3 fits',scope='both fitted Markov and semi-Markov models in both experiments; Hard24 survival only, CR3 survival and two CIFs')
        savejson(OUT/'sampling_plan.json',plan)
    if args.stage in ['all','fit']:fit_cr3_semi()
    if args.stage in ['all','sample']:
        for exp in ['Hard24','CR3']:evaluate(exp)
    if args.stage in ['all','summary']:summarize()
    if args.stage=='all':verify()
    if args.stage=='verify':
        folder=ROOT/'outputs'
        folder.mkdir(exist_ok=True)
        verify(folder/'s9_verification.json')
    if args.stage=='repair':repair_diagnostics();summarize();verify()
    if args.stage=='boundaryrepair':repair_sobol_boundary();verify()


if __name__=='__main__':main()
