"""Train and compare models on the new Weibull semi-Markov cohort."""
from pathlib import Path
import json,time,hashlib,shutil,pickle
import numpy as np
import pandas as pd
import torch
from extension_workflow import Experiment,SPECS,HISTORY_SPECS,ALL_SM_LABELS
import sfsa_semi_markov as sm
import sfsa_semi_markov_extras as extras
import reference_recursion as ref
from autoregressive_samplers import METHODS,sample_curve

ROOT=Path(__file__).resolve().parent
RULE='Post-transition, t>=24: previous Sicker and proposed Sicker -> Dead; retain t=20 ramp change.'
GLAZIERS=['Glazier-GRU','Glazier-RNN','Glazier-Transformer']
DIRECT=[m for m in ALL_SM_LABELS if m not in GLAZIERS]

def hashes():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in list((ROOT/'source').glob('*.py'))+list(ROOT.glob('*.py'))}

def guard(path,config):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and json.loads(path.read_text())!=config:
        raise ValueError('Settings/code/checkpoints changed. Use a fresh output directory.')
    path.write_text(json.dumps(config,indent=2))

def _reuse_previous(checkpoints,folder,n,device):
    """Copy compatible hard24 cohort/checkpoints; never alter the archived run."""
    checkpoints=Path(checkpoints)
    if (checkpoints/'checkpoints'/'run_config.json').exists():checkpoints=checkpoints/'checkpoints'
    cfg=json.loads((checkpoints/'run_config.json').read_text())
    if cfg.get('hard_death_rule')!=RULE or cfg['n']!=n or cfg['cohort_seed']!=24001 or cfg['kind']!='sm' or cfg['scenario']!='Weibull':
        raise ValueError('Reuse requires the same hard24 Weibull cohort, size and seed')
    for name in ['sfsa_data_generation.py','sfsa_semi_markov.py']:
        actual=hashlib.sha256((ROOT/'source'/name).read_bytes()).hexdigest()
        if cfg['source_sha256'].get(name)!=actual:raise ValueError('Generator source differs: '+name)
    previous=Experiment.restore(checkpoints,device=device)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if folder.resolve()==checkpoints.resolve():raise ValueError('Reuse destination must be a fresh folder')
    shutil.copy2(checkpoints/'cohort.npz',folder/'cohort.npz')
    previous.out=folder
    # Only the unchanged comparators are reused; new history options are fitted afresh.
    previous.models={k:v for k,v in previous.models.items() if k in SPECS}
    previous.config['training']={k:v for k,v in previous.config['training'].items() if k in SPECS}
    previous.config['history_comparator_version']=2
    previous.config['reused_from']=str(checkpoints.resolve())
    previous.config['reused_cohort_sha256']=hashlib.sha256((folder/'cohort.npz').read_bytes()).hexdigest()
    previous.config['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'source').glob('*.py')}
    for name,(model,meta,_) in previous.models.items():
        previous.config['training'][name]['reused_checkpoint']=True
        if name=='CPH':
            with (folder/(name+'.pkl')).open('wb') as f:pickle.dump(model,f)
        else:
            torch.save(dict(state_dict=model.state_dict(),meta=meta,label=name,kind='sm',
                            class_name=type(model).__name__),folder/(name+'.pt'))
    previous._save_config()
    return previous


def train(out,n=50000,epochs=200,patience=15,device='cuda',history_options=None,
          labels=None,reuse_from=None,training_seed=42):
    if reuse_from is not None and training_seed != 42:
        raise ValueError('Do not reuse trained models for a second training seed')
    labels=list(ALL_SM_LABELS if labels is None else labels)
    if len(set(labels))!=len(labels) or any(label not in ALL_SM_LABELS for label in labels):
        raise ValueError('Unknown or duplicate model label')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    guard(out/'training_settings.json',dict(n=n,epochs=epochs,patience=patience,device=device,
         rule=RULE,source_hashes=hashes(),cohort_seed=24001,training_seed=training_seed,
         history_options=history_options or {},labels=labels,history_comparator_version=2,
         reuse_from=None if reuse_from is None else str(Path(reuse_from).resolve())))
    folder=out/'checkpoints'
    if (folder/'run_config.json').exists():e=Experiment.restore(folder,device=device)
    elif reuse_from is not None:e=_reuse_previous(reuse_from,folder,n,device)
    else:e=Experiment('sm','Weibull',n=n,seed=24001,out_dir=folder)
    e.config['hard_death_rule']=RULE;e._save_config()
    for name in labels:
        if name not in e.models:e.train([name],epochs=epochs,patience=patience,seed=training_seed,
                                        device=device,history_options=history_options)
    return e

def references(e,ids,L):
    curves=[]
    for i in ids:
        states,initial,k=ref.kernels(float(e.cohort['covariate'][i]),'Weibull','semi')
        if L==0:curves.append(ref.forward(initial,k)[1][1:])
        else:
            prefix=e.cohort['states'][i:i+1,:L]
            u=int(sm.current_state_sojourn_time(prefix)[0])
            z=states.index((int(prefix[0,-1]),u))
            curves.append(ref.conditional_curves(k,L,[z])[0])
    return np.asarray(curves)

def evaluate(e,out,M=10000,n_eval=300,seeds=(41001,52001,63001,74001,85001),landmarks=(0,20,23,24),batch_size=512,reuse_from=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    ckpt={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in e.out.iterdir() if p.suffix in ('.pt','.pkl','.npz')}
    settings=dict(M=M,n_eval=n_eval,seeds=list(seeds),landmarks=list(landmarks),
         batch_size=batch_size,rule=RULE,checkpoint_hashes=ckpt,source_hashes=hashes(),importance_death_tilt=2.)
    reuse_plan=None
    if reuse_from is not None:
        from reuse_results import plan_reuse,apply_reuse
        reuse_plan=plan_reuse(e,reuse_from,out,settings,references)
        settings['prediction_reuse']={k:v for k,v in reuse_plan.items() if k!='files'}
    guard(out/'evaluation_settings.json',settings)
    if reuse_plan is not None:apply_reuse(reuse_plan,out)
    reused_files=set() if reuse_plan is None else {str(Path(item['destination']).resolve()) for item in reuse_plan['files']}
    if len(seeds)<2:raise ValueError('Use at least two sampling seeds')
    c=e.cohort;ids0=np.random.default_rng(e.seed+2).permutation(c['test_index'])[:n_eval]
    rows=[]
    for L in landmarks:
        ids=ids0 if L==0 else ids0[c['states'][ids0,L-1]<3]
        if len(ids)==0:
            print('No eligible patients at L=',L,flush=True);continue
        folder=out/f'L{L}';folder.mkdir(exist_ok=True)
        grid=np.arange(L+1,31)
        if (folder/'reference.npz').exists():
            with np.load(folder/'reference.npz') as z:
                np.testing.assert_array_equal(z['patient_ids'],ids)
                np.testing.assert_array_equal(z['times'],grid)
                truth=z['truth']
        else:
            truth=references(e,ids,L)
            np.savez_compressed(folder/'reference.npz',patient_ids=ids,times=grid,truth=truth)
        x=c['covariate'][ids];prefix=None if L==0 else c['states'][ids,:L]
        tasks=[(name,'direct',-1) for name in DIRECT if name in e.models]
        tasks +=[(name,method,seed) for name in GLAZIERS if name in e.models for method in METHODS for seed in seeds]
        tasks += [('Oracle sampler','iid',seed) for seed in seeds]
        for name,method,seed in tasks:
            file=folder/f'{name}_{method}_{seed}.npz';start=time.perf_counter()
            if file.exists():
                with np.load(file) as saved:p=saved['prediction'];seconds=float(saved['seconds']);ess=float(saved['min_ESS'])
            else:
                ess=float(M)
                if method=='direct':p=e.predict(name,x,grid,prefix=prefix,L=None if L==0 else L)
                elif name=='Oracle sampler':
                    if L==0:p=extras.ground_truth_km_sm_for_validation_set(e.scenario,x,base_seed=seed,n_matched=M)[:,1:]
                    else:p=sm.ground_truth_km_landmark_sm_for_validation_set(e.scenario,x,prefix,L,base_seed=seed,n_matched=M)[:,L+1:]
                else:
                    model,meta,_=e.models[name];curves=[]
                    for j,cov in enumerate(x):
                        child=int(np.random.SeedSequence([seed,j]).generate_state(1)[0]%(2**31))
                        curve,diag=sample_curve(model,meta,float(cov),M=M,method=method,seed=child,
                             prefix=() if L==0 else prefix[j],batch_size=batch_size)
                        curves.append(curve);ess=min(ess,diag.get('trajectory_ESS',diag.get('min_particle_ESS',M)))
                    p=np.asarray(curves)
                seconds=time.perf_counter()-start
                assert p.shape==truth.shape and np.isfinite(p).all()
                assert p.min()>=-1e-6 and p.max()<=1+1e-6 and (np.diff(p,axis=1)<=1e-6).all()
                temp=file.with_suffix('.tmp.npz');np.savez_compressed(temp,prediction=p,seconds=seconds,min_ESS=ess);temp.replace(file)
            error=abs(p-truth);after=grid>=24
            rows.append(dict(model=name,method=method,seed=seed,landmark=L,n=len(ids),M=0 if method=='direct' else M,
                reused_prediction=str(file.resolve()) in reused_files,
                prediction_origin=reuse_plan['source_run'] if str(file.resolve()) in reused_files else str(out.resolve()),
                information='observed prefix' if L and (name in HISTORY_SPECS or name.startswith('Glazier') or name=='Oracle sampler') else ('baseline covariate' if L==0 else 'covariate + survival-only ratio'),
                MAE=error.mean(),post24_MAE=error[:,after].mean(),RMSE=np.sqrt(((p-truth)**2).mean()),
                seconds=seconds,min_ESS=ess if method!='direct' else np.nan))
            pd.DataFrame(rows).to_csv(out/'replicate_metrics.csv',index=False)
            origin='saved prediction; original runtime' if str(file.resolve()) in reused_files else 'runtime'
            print(f'L={L} {name}/{method} seed={seed}: MAE={error.mean():.5f}, {origin} {seconds:.1f}s',flush=True)
    df=pd.DataFrame(rows)
    summary=df.groupby(['landmark','model','method'],sort=False).agg(n=('n','first'),repeats=('seed','size'),
        MAE=('MAE','mean'),seed_SD=('MAE','std'),post24_MAE=('post24_MAE','mean'),RMSE=('RMSE','mean'),
        mean_seconds=('seconds','mean'),min_ESS=('min_ESS','min'),
        reused_prediction=('reused_prediction','all')).reset_index()
    summary.to_csv(out/'summary.csv',index=False)
    (out/'COMPLETE.json').write_text(json.dumps(dict(complete=True,rows=len(df))))
    return df,summary

def plot(out):
    import matplotlib.pyplot as plt
    out=Path(out);d=pd.read_csv(out/'summary.csv');dest=out/'plots';dest.mkdir(exist_ok=True)
    for L,group in d.groupby('landmark'):
        g=group.sort_values('MAE');labels=g.model+' / '+g.method
        fig,axes=plt.subplots(1,2,figsize=(16,10))
        colors=['#6750a4' if x.startswith('Glazier') else '#168b83' for x in g.model]
        axes[0].barh(labels,g.MAE,xerr=g.seed_SD.fillna(0),color=colors);axes[0].invert_yaxis()
        axes[0].set_xlabel('MAE; error bars are ±1 sampling-seed SD')
        axes[1].barh(labels,g.post24_MAE,color=colors);axes[1].invert_yaxis();axes[1].set_xlabel('MAE at times 24–30')
        fig.suptitle(f'Weibull semi-Markov, hard rule from t=24: L={L}')
        fig.tight_layout();fig.savefig(dest/f'comparison_L{L}.png',dpi=140,bbox_inches='tight');plt.show();plt.close(fig)
        # Average-patient curves show the regime change; error metrics above
        # still compare individual curves before averaging.
        folder=out/f'L{L}'
        with np.load(folder/'reference.npz') as z:t=z['times'];truth=z['truth']
        fig,ax=plt.subplots(figsize=(11,5));ax.plot(t,truth.mean(0),'k-',lw=3,label='Deterministic SFSA reference')
        for name,method in [(name,'direct') for name in DIRECT if name in ['DynamicDeepHit','SurvTransformer','SurvTransformer-Discrete','DynamicDeepHIt-Lite','SurvTransformer-Lite']]+[('Glazier-Transformer',m) for m in METHODS]:
            arrays=[]
            for p in folder.glob(f'{name}_{method}_*.npz'):
                with np.load(p) as z:arrays.append(z['prediction'])
            if arrays:ax.plot(t,np.mean(arrays,axis=(0,1)),label=f'{name}/{method}',alpha=.8)
        if t[0]<=24:ax.axvline(24,color='grey',ls=':')
        ax.set(xlabel='Absolute time',ylabel='Mean survival probability',title=f'Eligible-patient mean curves, L={L}')
        ax.legend(fontsize=8,ncol=2);fig.tight_layout();fig.savefig(dest/f'curves_L{L}.png',dpi=140);plt.show();plt.close(fig)
    return d
