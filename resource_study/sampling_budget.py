"""Checkpoint-only semi-Markov sampling diagnostics; no training on import."""
from pathlib import Path
import json, hashlib, time
import numpy as np
import pandas as pd
from extension_workflow import Experiment
import reference_recursion as ref
import sfsa_semi_markov as sm
import sfsa_semi_markov_extras as extras

GLAZIERS=['Glazier-GRU','Glazier-RNN','Glazier-Transformer']
DIRECT=['DynamicDeepHit','SurvTransformer']

def exact_reference(e, ids, landmark):
    c=e.cohort; curves=[]
    for i in ids:
        states,initial,k=ref.kernels(float(c['covariate'][i]),e.scenario.name,'semi')
        if landmark==0:curves.append(ref.forward(initial,k)[1][1:])
        else:
            prefix=c['states'][i:i+1,:landmark]
            u=int(sm.current_state_sojourn_time(prefix)[0])
            index=states.index((int(prefix[0,-1]),u))
            curves.append(ref.conditional_curves(k,landmark,[index])[0])
    return np.asarray(curves)

def check_curve(a, shape):
    assert a.shape==shape,(a.shape,shape)
    assert np.isfinite(a).all()
    assert a.min()>=-1e-6 and a.max()<=1+1e-6
    assert (np.diff(a,axis=-1)<=1e-6).all()

def decompose(draws, truth):
    """Exact finite-replicate MSE identity; ensemble error is not pure bias."""
    mean=draws.mean(0)
    variance=draws.var(0,ddof=1)
    mse=((draws-truth)**2).mean()
    ensemble_mse=((mean-truth)**2).mean()
    sampling_component=(len(draws)-1)/len(draws)*variance.mean()
    assert np.isclose(mse,ensemble_mse+sampling_component,atol=1e-12)
    return dict(mean_draw_MSE=mse,ensemble_MSE=ensemble_mse,
                finite_replicate_sampling_component=sampling_component,
                sampling_RMS=np.sqrt(variance.mean()),ensemble_MAE=abs(mean-truth).mean())

def run(bundle, out, scenarios=('Weibull','ReLU'), budgets=(500,2000,10000),
        seeds=(41001,52001,63001,74001,85001), n_eval=300,
        landmarks=(0,10,15,20), device='cuda', models=GLAZIERS):
    bundle=Path(bundle);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    assert len(seeds)>=2 and len(set(seeds))==len(seeds),'Use at least two distinct seeds'
    assert len(set(budgets))==len(budgets) and min(budgets)>=2
    assert n_eval>0 and all(0<=L<30 for L in landmarks)
    assert set(models)<=set(GLAZIERS)
    config=dict(scenarios=list(scenarios),budgets=list(budgets),seeds=list(seeds),n_eval=n_eval,
                landmarks=list(landmarks),device=device,models=list(models),
                reference='deterministic population event-first KM of preserved default semi-Markov generator',
                source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (bundle/'source').glob('*.py')},
                diagnostic_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                input_hashes={str(p.relative_to(bundle)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for s in scenarios for p in (bundle/('sm_'+s)).iterdir() if p.suffix in ('.pt','.pkl','.npz','.json')})
    conf=out/'experiment_config.json'
    if conf.exists() and json.loads(conf.read_text())!=config:
        raise ValueError('Configuration or inputs changed. Choose a new output folder to avoid mixing experiments.')
    conf.write_text(json.dumps(config,indent=2))
    rows=[];summaries=[];paired=[]
    for scenario in scenarios:
        e=Experiment.restore(bundle/('sm_'+scenario),device=device)
        if e.kind!='sm':raise ValueError('Semi-Markov checkpoints required')
        c=e.cohort
        ids0=np.random.default_rng(e.seed+2).permutation(c['test_index'])[:n_eval]
        for L in landmarks:
            ids=ids0 if L==0 else ids0[c['states'][ids0,L-1]<3]
            if not len(ids):
                print(f'{scenario}, L={L}: no eligible patients; skipped',flush=True);continue
            folder=out/f'{scenario}_L{L}';folder.mkdir(exist_ok=True)
            grid=np.arange(L+1,31);x=c['covariate'][ids]
            prefix=None if L==0 else c['states'][ids,:L]
            truth=exact_reference(e,ids,L);check_curve(truth,(len(ids),len(grid)))
            np.savez_compressed(folder/'reference.npz',patient_ids=ids,times=grid,truth=truth)
            direct={}
            for label in DIRECT:
                p=e.predict(label,x,grid,prefix=prefix,L=None if L==0 else L)
                check_curve(p,truth.shape);direct[label]=p
                np.savez_compressed(folder/(label+'.npz'),prediction=p)
                rows.append(dict(scenario=scenario,landmark=L,model=label,budget=0,seed=-1,n=len(ids),
                                 MAE=abs(p-truth).mean(),MSE=((p-truth)**2).mean(),seconds=0.))
            for label in list(models)+['Oracle sampler']:
                for M in budgets:
                    draws=[];maes=[];mses=[]
                    for seed in seeds:
                        # Separate reproducible streams across model, budget, and replicate.
                        model_id=(GLAZIERS+['Oracle sampler']).index(label)
                        actual_seed=int(np.random.SeedSequence([int(seed),int(M),model_id,int(L)]).generate_state(1)[0] % (2**31))
                        path=folder/f'{label}_M{M}_seed{seed}.npz';start=time.perf_counter()
                        if path.exists():
                            with np.load(path) as z:p=z['prediction'];seconds=float(z['seconds'])
                        else:
                            if label=='Oracle sampler':
                                if L==0:
                                    p=extras.ground_truth_km_sm_for_validation_set(e.scenario,x,base_seed=actual_seed,n_matched=M)[:,1:]
                                else:
                                    p=sm.ground_truth_km_landmark_sm_for_validation_set(e.scenario,x,prefix,L,base_seed=actual_seed,n_matched=M)[:,L+1:]
                            else:p=e.predict(label,x,grid,samples=M,seed=actual_seed,prefix=prefix,L=None if L==0 else L)
                            seconds=time.perf_counter()-start;check_curve(p,truth.shape)
                            temporary=path.with_suffix('.tmp.npz')
                            np.savez_compressed(temporary,prediction=p,seconds=seconds,actual_seed=actual_seed)
                            temporary.replace(path)
                        check_curve(p,truth.shape);draws.append(p)
                        mae=abs(p-truth).mean();mse=((p-truth)**2).mean();maes.append(mae);mses.append(mse)
                        rows.append(dict(scenario=scenario,landmark=L,model=label,budget=M,seed=seed,n=len(ids),MAE=mae,MSE=mse,seconds=seconds))
                        print(f'{scenario} L={L} {label} M={M} seed={seed}: MAE={mae:.5f}',flush=True)
                        pd.DataFrame(rows).to_csv(out/'replicate_metrics.csv',index=False)
                    draws=np.stack(draws)
                    summaries.append(dict(scenario=scenario,landmark=L,model=label,budget=M,n=len(ids),
                        repeats=len(seeds),mean_MAE=np.mean(maes),seed_SD_MAE=np.std(maes,ddof=1),
                        **decompose(draws,truth)))
                    for comparator,p in direct.items():
                        differences=np.asarray(maes)-abs(p-truth).mean()
                        paired.append(dict(scenario=scenario,landmark=L,model=label,budget=M,comparator=comparator,
                            mean_MAE_difference=differences.mean(),seed_SD_difference=differences.std(ddof=1)))
                    pd.DataFrame(summaries).to_csv(out/'budget_summary.csv',index=False)
                    pd.DataFrame(paired).to_csv(out/'direct_comparisons.csv',index=False)
        del e
        import gc,torch
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    (out/'COMPLETE.json').write_text(json.dumps(dict(completed=True,replicate_rows=len(rows),summary_rows=len(summaries)),indent=2))
    return pd.DataFrame(rows),pd.DataFrame(summaries),pd.DataFrame(paired)

def plot_results(out):
    import matplotlib.pyplot as plt
    out=Path(out);summary=pd.read_csv(out/'budget_summary.csv');raw=pd.read_csv(out/'replicate_metrics.csv')
    colors=dict(zip(GLAZIERS+['Oracle sampler'],['#6750a4','#9862ac','#c479a5','#222222']))
    plots=out/'plots';plots.mkdir(exist_ok=True)
    for (scenario,L),group in summary.groupby(['scenario','landmark']):
        fig,axes=plt.subplots(1,2,figsize=(12,4.8))
        for model,d in group.groupby('model',sort=False):
            d=d.sort_values('budget')
            axes[0].errorbar(d.budget,d.mean_MAE,yerr=d.seed_SD_MAE,marker='o',capsize=3,label=model,color=colors[model])
            axes[1].plot(d.budget,d.sampling_RMS,'o-',label=model,color=colors[model])
        for model in DIRECT:
            v=raw[(raw.scenario==scenario)&(raw.landmark==L)&(raw.model==model)].MAE.iloc[0]
            axes[0].axhline(v,ls='--',label=model,color={'DynamicDeepHit':'#168b83','SurvTransformer':'#d59d26'}[model])
        for ax in axes:
            ax.set_xscale('log');ax.set_xlabel('Trajectories per patient');ax.grid(alpha=.2)
            ticks=sorted(group.budget.unique())
            ax.set_xticks(ticks,labels=[f'{v:,}' for v in ticks]);ax.minorticks_off()
        axes[0].set_ylabel('MAE against deterministic reference');axes[0].set_title('Error bars: ±1 sampling-seed SD')
        axes[1].set_ylabel('RMS of pointwise sampling SD');axes[1].set_title('Prediction variability across sampling seeds')
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=9)
        fig.suptitle(f'{scenario}: '+('baseline' if L==0 else f'landmark L={L} (different input information)'))
        fig.tight_layout(rect=[0,.14,1,.94]);fig.savefig(plots/f'{scenario}_L{L}.png',dpi=170,bbox_inches='tight')
        plt.show();plt.close(fig)
    return summary
