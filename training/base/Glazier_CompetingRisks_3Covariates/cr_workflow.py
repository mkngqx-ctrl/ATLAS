"""Reproducible new-cohort training, resumable comparison, and plotting."""
from pathlib import Path
import hashlib,json,pickle,time,sys
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
from sfsa_cr import Scenario,STATES,COVARIATES,TARGETS,generate,reference_curves,validate_curves
from model_training import LABELS,HISTORY,GLAZIER,CLASSES,fit,predict
from samplers_cr import METHODS,Oracle,sample_curves


def hashes():
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(list(ROOT.glob('*.py'))+list((ROOT/'source').glob('*.py')))}


def guard(path,settings):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and json.loads(path.read_text())!=settings:
        raise ValueError(f'Settings/code changed for {path.parent}. Choose a fresh RUN_TAG.')
    path.write_text(json.dumps(settings,indent=2),encoding='utf-8')


def save_npz(path,**arrays):
    path=Path(path);temp=path.with_suffix('.temporary.npz')
    np.savez_compressed(temp,**arrays);temp.replace(path)


def restore(folder,device='cpu'):
    folder=Path(folder);models={}
    for name in LABELS:
        file=folder/(name+('.pkl' if name=='CPH' else '.pt'))
        if not file.exists():continue
        if name=='CPH':
            with file.open('rb') as handle:model,meta=pickle.load(handle)
        else:
            data=torch.load(file,map_location=device,weights_only=True)
            if data.get('benchmark_version')!='cr_3cov_v1' or data['label']!=name:
                raise ValueError('Incompatible checkpoint: '+str(file))
            meta=data['meta'];model=CLASSES[data['class_name']](**meta['model_config']).to(device)
            model.load_state_dict(data['state_dict']);model.eval()
        meta['device']=device;models[name]=(model,meta)
    return models


def train(out,n=50000,epochs=200,patience=15,seed=42,cohort_seed=34001,
          device='cpu',scenario=None,options=None):
    scenario=scenario or Scenario();options=options or {};out=Path(out)
    if set(options)-set(LABELS):raise ValueError('Unknown model option labels')
    if min(epochs,patience)<1:raise ValueError('Positive training budgets required')
    settings=dict(benchmark_version='cr_3cov_v1',n=n,epochs=epochs,patience=patience,
                  seed=seed,cohort_seed=cohort_seed,scenario=scenario.to_dict(),options=options,
                  labels=LABELS,source_hashes=hashes(),device=device)
    guard(out/'training_settings.json',settings)
    folder=out/'checkpoints';folder.mkdir(exist_ok=True)
    if (folder/'cohort.npz').exists():
        with np.load(folder/'cohort.npz',allow_pickle=False) as z:cohort={k:z[k] for k in z.files}
        expected=json.loads((folder/'cohort_manifest.json').read_text())['sha256']
        if hashlib.sha256((folder/'cohort.npz').read_bytes()).hexdigest()!=expected:raise ValueError('Saved cohort changed')
    else:
        cohort=generate(n,cohort_seed,scenario);save_npz(folder/'cohort.npz',**cohort)
        (folder/'cohort_manifest.json').write_text(json.dumps(dict(sha256=hashlib.sha256((folder/'cohort.npz').read_bytes()).hexdigest(),
            n=n,split_seed=cohort_seed+1,states=STATES,covariates=COVARIATES,scenario=scenario.to_dict()),indent=2))
    models=restore(folder,device)
    records_path=out/'training_summary.csv'
    records=pd.read_csv(records_path).to_dict('records') if records_path.exists() else []
    for label in LABELS:
        if label in models:
            print('Restored:',label,flush=True);continue
        print('Training:',label,flush=True);started=time.perf_counter()
        model,meta=fit(label,cohort,scenario.horizon,epochs,patience,seed,device,options.get(label))
        elapsed=time.perf_counter()-started
        file=folder/(label+('.pkl' if label=='CPH' else '.pt'));temp=file.with_suffix(file.suffix+'.temporary')
        if label=='CPH':
            with temp.open('wb') as handle:pickle.dump((model,meta),handle)
        else:
            torch.save(dict(benchmark_version='cr_3cov_v1',label=label,class_name=type(model).__name__,
                            meta=meta,state_dict=model.state_dict()),temp)
        temp.replace(file);models[label]=(model,meta)
        if meta['training_log']:pd.DataFrame(meta['training_log']).to_csv(folder/(label+'_training.csv'),index=False)
        records.append(dict(model=label,seconds=elapsed,best_epoch=meta.get('best_epoch',0),
                            training_prefixes=meta.get('training_prefixes',len(cohort['train_index']))))
        pd.DataFrame(records).to_csv(records_path,index=False)
    (out/'environment.json').write_text(json.dumps(dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,
        device=device,gpu=torch.cuda.get_device_name(0) if device=='cuda' else None),indent=2))
    return dict(models=models,cohort=cohort,scenario=scenario,out=out,cohort_seed=cohort_seed)


def evaluate(experiment,out=None,M=10000,n_eval=300,seeds=(41001,52001,63001,74001,85001),
             landmarks=(0,10,20),batch_size=512):
    if M<2 or len(seeds)<2 or len(set(seeds))!=len(seeds):raise ValueError('Use M>=2 and at least two distinct sampling seeds')
    scenario=experiment['scenario'];H=scenario.horizon
    if len(set(landmarks))!=len(landmarks) or any(L<0 or L>=H for L in landmarks):raise ValueError('Invalid landmarks')
    if n_eval<1 or batch_size<1:raise ValueError('Positive evaluation budgets required')
    out=Path(out or experiment['out']/'evaluation');ckpt=experiment['out']/'checkpoints'
    settings=dict(M=M,n_eval=n_eval,seeds=list(seeds),landmarks=list(landmarks),batch_size=batch_size,
        labels=LABELS,methods=list(METHODS),source_hashes=hashes(),scenario=scenario.to_dict(),
        checkpoint_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ckpt.iterdir() if p.suffix in ('.pt','.pkl','.npz')})
    guard(out/'evaluation_settings.json',settings)
    c=experiment['cohort'];models=experiment['models']
    if set(models)!=set(LABELS):raise ValueError('All ten requested models must be present')
    selected=np.random.default_rng(experiment['cohort_seed']+2).permutation(c['test_index'])[:n_eval]
    rows=[];eligible=[]
    for L in landmarks:
        ids=selected if L==0 else selected[c['states'][selected,L-1]<7]
        eligible.append(dict(landmark=L,n=len(ids)))
        if not len(ids):print(f'No eligible patients at L={L}; increase n_eval.',flush=True);continue
        folder=out/f'L{L}';folder.mkdir(exist_ok=True)
        x=c['covariates'][ids];prefix=c['states'][ids,:L];grid=np.arange(L+1,H+1)
        if (folder/'reference.npz').exists():
            with np.load(folder/'reference.npz') as z:
                np.testing.assert_array_equal(z['patient_ids'],ids)
                np.testing.assert_array_equal(z['times'],grid)
                np.testing.assert_array_equal(z['covariates'],x)
                np.testing.assert_array_equal(z['prefix'],prefix)
                truth=z['truth']
        else:
            truth=reference_curves(x,prefix,L,scenario)
            save_npz(folder/'reference.npz',patient_ids=ids,covariates=x,prefix=prefix,times=grid,truth=truth)
        validate_curves(truth)
        tasks=[(label,'direct',-1) for label in LABELS if label not in GLAZIER]
        tasks += [(label,method,seed) for label in GLAZIER for method in METHODS for seed in seeds]
        tasks += [('Oracle sampler','iid',seed) for seed in seeds]
        for label,method,seed in tasks:
            file=folder/f'{label}_{method}_{seed}.npz'
            if file.exists():
                with np.load(file,allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['patient_ids'],ids);np.testing.assert_array_equal(z['times'],grid)
                    prediction=z['prediction'];seconds=float(z['seconds']);ess=float(z['min_ESS'])
                status='cached'
            else:
                start=time.perf_counter();ess=float(M)
                if method=='direct':
                    model,meta=models[label];prediction=predict(label,model,meta,x,prefix,L,H,batch_size);ess=float('nan')
                else:
                    if label=='Oracle sampler':model,meta=Oracle(scenario),dict(horizon=H,x_mean=[0,0,0],x_sd=[1,1,1])
                    else:model,meta=models[label]
                    curves=[]
                    for j,cov in enumerate(x):
                        child=int(np.random.SeedSequence([seed,int(ids[j]),L]).generate_state(1)[0]%(2**31))
                        curve,diag=sample_curves(model,meta,cov,M,method,child,prefix[j],batch_size)
                        curves.append(curve);ess=min(ess,diag['min_ESS'])
                    prediction=np.asarray(curves)
                seconds=time.perf_counter()-start;status='new'
                if prediction.shape!=truth.shape:raise ValueError('Prediction/reference shape mismatch')
                validate_curves(prediction)
                save_npz(file,prediction=prediction,patient_ids=ids,times=grid,seconds=seconds,min_ESS=ess)
            validate_curves(prediction)
            for k,target in enumerate(TARGETS):
                error=prediction[:,k]-truth[:,k]
                rows.append(dict(model=label,method=method,seed=seed,landmark=L,target=target,n=len(ids),
                    M=M if method!='direct' else 0,MAE=np.abs(error).mean(),RMSE=np.sqrt((error**2).mean()),
                    seconds=seconds,min_ESS=ess,information='full observed prefix + 3 covariates' if L and (label in HISTORY or label in GLAZIER or label=='Oracle sampler') else ('3 baseline covariates' if L==0 else '3 covariates + survival-only conditioning')))
            pd.DataFrame(rows).to_csv(out/'replicate_metrics.csv',index=False)
            print(f'L={L} {label}/{method}, seed={seed}: S MAE={abs(prediction[:,0]-truth[:,0]).mean():.5f} ({status}, {seconds:.1f}s)',flush=True)
    df=pd.DataFrame(rows)
    if not len(df):raise ValueError('No eligible evaluation patients')
    summary=df.groupby(['landmark','target','model','method'],sort=False).agg(n=('n','first'),repeats=('seed','size'),
        MAE=('MAE','mean'),seed_SD=('MAE','std'),RMSE=('RMSE','mean'),mean_seconds=('seconds','mean'),min_ESS=('min_ESS','min')).reset_index()
    summary.to_csv(out/'summary.csv',index=False);pd.DataFrame(eligible).to_csv(out/'eligible_patients.csv',index=False)
    (out/'COMPLETE.json').write_text(json.dumps(dict(complete=True,rows=len(df),prediction_files=len(df)//3),indent=2))
    return df,summary


def plot(out):
    import matplotlib.pyplot as plt
    out=Path(out);d=pd.read_csv(out/'summary.csv');dest=out/'plots';dest.mkdir(exist_ok=True)
    for L,group in d.groupby('landmark'):
        ordered=group[group.target=='Survival'].sort_values('MAE');pairs=list(zip(ordered.model,ordered.method))
        labels=[f'{name} / {method}' for name,method in pairs]
        fig,axes=plt.subplots(1,3,figsize=(22,10))
        for ax,target in zip(axes,TARGETS):
            g=group[group.target==target].set_index(['model','method']).loc[pairs]
            ax.barh(labels,g.MAE,xerr=g.seed_SD.fillna(0),color=['#6750a4' if m.startswith('Glazier') else '#168b83' for m,_ in pairs])
            ax.invert_yaxis();ax.set_title(target);ax.set_xlabel('Curve MAE; ±1 sampling-seed SD')
        fig.suptitle(f'Two-disease SFSA, three covariates: L={L}');fig.tight_layout()
        fig.savefig(dest/f'comparison_L{L}.png',dpi=140,bbox_inches='tight');plt.show();plt.close(fig)
        with np.load(out/f'L{L}/reference.npz') as z:grid=z['times'];truth=z['truth'];cov=z['covariates'];ids=z['patient_ids'];prefix=z['prefix']
        focus=[('Glazier-GRU','sobol'),('Glazier-Transformer','sobol')]+[(name,'direct') for name in HISTORY]
        fig,axes=plt.subplots(1,3,figsize=(18,5))
        for k,ax in enumerate(axes):
            ax.plot(grid,truth[:,k].mean(0),color='black',lw=3,label='Exact SFSA reference')
            for name,method in focus:
                arrays=[]
                for file in (out/f'L{L}').glob(f'{name}_{method}_*.npz'):
                    with np.load(file) as z:arrays.append(z['prediction'])
                if arrays:ax.plot(grid,np.mean(arrays,axis=(0,1))[k],label=f'{name}/{method}')
            ax.set(title=TARGETS[k],xlabel='Absolute time',ylabel='Mean probability',ylim=(0,1))
        axes[0].legend(fontsize=7);fig.suptitle(f'Eligible-patient mean curves, L={L}');fig.tight_layout()
        fig.savefig(dest/f'mean_curves_L{L}.png',dpi=140);plt.show();plt.close(fig)
        # Deterministic patient selection: calcium quantiles, independent of errors.
        picks=np.unique(np.argsort(cov[:,0])[np.linspace(0,len(cov)-1,min(3,len(cov))).astype(int)])
        for j in picks:
            fig,axes=plt.subplots(1,3,figsize=(17,4))
            for k,ax in enumerate(axes):
                ax.plot(grid,truth[j,k],color='black',lw=3,label='Exact SFSA reference')
                for name,method in focus:
                    arrays=[]
                    for file in (out/f'L{L}').glob(f'{name}_{method}_*.npz'):
                        with np.load(file) as z:arrays.append(z['prediction'][j,k])
                    if arrays:ax.plot(grid,np.mean(arrays,axis=0),label=f'{name}/{method}')
                ax.set(title=TARGETS[k],xlabel='Absolute time',ylim=(0,1))
            axes[0].legend(fontsize=6)
            state='START' if not L else STATES[prefix[j,-1]]
            fig.suptitle(f'Patient {ids[j]}, L={L}, {state}; calcium={cov[j,0]:.2f}, age={cov[j,1]:.1f}, drug={cov[j,2]:.2f}')
            fig.tight_layout();fig.savefig(dest/f'patient_{ids[j]}_L{L}.png',dpi=140);plt.close(fig)
    return d
