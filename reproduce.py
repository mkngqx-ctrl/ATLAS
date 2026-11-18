"""Reproduce the reported test-set MAEs from saved prediction curves."""
from pathlib import Path
from collections import defaultdict
import argparse,csv,io,json,subprocess,sys,zipfile
import numpy as np

ROOT=Path(__file__).resolve().parent
INTEGRATED=ROOT/'archive/supplementary/optimized_DRSA_integrated_2026-11-16'
OUT=ROOT/'outputs'
SEEDS={'Hard24':[42,43,44,45,50,51,52,62,63],'CR3':[42,43,45,46,47]}
LANDMARKS={'Hard24':[0,20,23,24],'CR3':[0,10,20]}
def unpack(exp,seed):
    dest=OUT/'runs'/exp/str(seed)
    with zipfile.ZipFile(ROOT/'runs'/exp/f'seed_{seed}.zip') as z:
        z.extractall(dest)
    return dest
def read_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write_rows(name,rows):
    OUT.mkdir(exist_ok=True)
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
def curve_metrics(pred,truth,times):
    err=pred-truth
    mae=np.abs(err).mean(axis=(0,2));rmse=np.sqrt(np.square(err).mean(axis=(0,2)))
    post=np.abs(err[:,:,times>=24]).mean(axis=(0,2))
    return np.column_stack([mae,rmse,post])
def score():
    expected=read_rows(INTEGRATED/'updated_all_models_per_training_run.csv')
    calculated={}
    for exp,seeds in SEEDS.items():
        for seed in seeds:
            with zipfile.ZipFile(ROOT/'runs'/exp/f'seed_{seed}.zip') as archive:
                load=lambda path:np.load(io.BytesIO(archive.read(path)),allow_pickle=False)
                with load('checkpoints/cohort.npz') as cohort:test=cohort['test_index']
                for L in LANDMARKS[exp]:
                    prefix=f'evaluation/L{L}/'
                    with load(prefix+'reference.npz') as ref:
                        truth,ids,times=ref['truth'],ref['patient_ids'],ref['times']
                    assert len(np.unique(ids)) == len(ids) and np.isin(ids,test).all() and (L or len(ids)==300)
                    if truth.ndim==2:truth=truth[:,None,:]
                    groups=defaultdict(list)
                    for path in archive.namelist():
                        if not path.startswith(prefix) or not path.endswith('.npz') or path.endswith('reference.npz'):continue
                        name=Path(path).stem
                        method=next((v for v in ('conditional_mc','weighted_branching','stratified','importance','antithetic','sobol','iid','direct') if ('_'+v+'_') in name),None)
                        if not method:continue
                        model=name.rsplit('_'+method+'_',1)[0].replace('Glazier-','ATLAS-').replace('DynamicDeepHIt-Lite','DynamicDeepHit-Lite')
                        if model in ('DRSA','DynamicDeepHit'):continue
                        with load(path) as z:
                            if 'patient_ids' in z:np.testing.assert_array_equal(z['patient_ids'],ids)
                            pred=z['prediction']
                        if pred.ndim==2:pred=pred[:,None,:]
                        assert pred.shape==truth.shape
                        groups[model,method].append(curve_metrics(pred,truth,times))
                    for model,p in [('DynamicDeepHit',ROOT/'optimized_ddh/evaluation'/exp/f'seed_{seed}_L{L}.npz'),('DRSA',INTEGRATED/'predictions'/exp/f'seed_{seed}'/f'L{L}.npz')]:
                        with np.load(p) as z:
                            np.testing.assert_array_equal(z['patient_ids'],ids);np.testing.assert_array_equal(z['times'],times)
                            assert z['prediction'].shape==truth.shape
                            groups[model,'direct'].append(curve_metrics(z['prediction'],truth,times))
                    for (model,method),rep in groups.items():
                        for target,metrics in zip(['Survival','CIF_Death_A','CIF_Death_B'],np.mean(rep,axis=0)):
                            calculated[exp,seed,L,target,model,method]=metrics
    results=[];aggregate=defaultdict(list)
    checked=set()
    for row in expected:
        key=(row['experiment'],int(row['training_seed']),int(row['landmark']),row['target'],row['model'],row['method'])
        if key in checked:
            raise ValueError(f'Duplicate reference row: {key}')
        checked.add(key)
        for i,metric in enumerate(('MAE','RMSE','post24_MAE')):
            if row[metric]:np.testing.assert_allclose(calculated[key][i],float(row[metric]),atol=1e-7,rtol=0,err_msg=str((key,metric)))
    # The duration-dependent oracle is shared across fits. Its benchmark
    # appears once per landmark in the overview rather than in the fit table.
    overview=read_rows(ROOT/'archive/supplementary/overview_plot_values.csv')
    for row in overview:
        if row['experiment']=='Hard24' and row['model']=='Oracle sampler':
            for seed in SEEDS['Hard24']:
                key=('Hard24',seed,int(row['landmark']),'Survival',row['model'],row['method'])
                np.testing.assert_allclose(calculated[key][0],float(row['mean_MAE']),atol=1e-7,rtol=0)
                checked.add(key)
    if set(calculated) != checked:
        raise ValueError('Computed model/seed/landmark/target combinations differ from the reference table')
    for (exp,seed,L,target,model,method),val in calculated.items():
        results.append(dict(experiment=exp,training_seed=seed,landmark=L,target=target,model=model,method=method,MAE=val[0],RMSE=val[1],post24_MAE=val[2] if exp=='Hard24' else ''));aggregate[exp,L,target,model,method].append(val)
    write_rows('per_run_MAE.csv',results)
    write_rows('mean_MAE.csv',[dict(experiment=k[0],landmark=k[1],target=k[2],model=k[3],method=k[4],mean_MAE=np.mean(v,axis=0)[0],mean_RMSE=np.mean(v,axis=0)[1],between_SD=np.std(np.array(v)[:,0],ddof=1) if len(v)>1 else 0,minimum=min(row[0] for row in v),maximum=max(row[0] for row in v)) for k,v in aggregate.items()])
    print(f'Checked {len(expected)} archived per-run MAEs against predictions. Results in outputs/.')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['scores','bootstrap','figures','all','prepare','train-base','train-ddh','train-drsa','s9','resource']);p.add_argument('--experiment',choices=list(SEEDS),default='Hard24');p.add_argument('--seed',type=int,default=43);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args()
    if a.action in ('all','scores'):score()
    for name in ['bootstrap','figures']:
        if a.action in ('all',name):subprocess.run([sys.executable,str(ROOT/'tools'/f'{name}.py')],check=True)
    if a.action in ['prepare','train-base','train-drsa','figures','all','s9']:
        for exp in SEEDS:unpack(exp,42)
    if a.action in ('all','figures'):
        for program in ('intuition','main_figures','supplement'):subprocess.run([sys.executable,str(ROOT/'tools'/f'{program}.py')],check=True)
    if a.action=='s9':
        c=ROOT/'archive/classical_comparators'
        subprocess.run([sys.executable,str(c/'sampled_comparison_2026-10-28/run_classical_sampling.py'),'--stage','verify'],check=True)
    if a.action=='resource':
        folder=ROOT/'resource_study';sys.path[:0]=[str(folder),str(folder/'source')]
        import sampling_budget
        sampling_budget.run(folder,OUT/'resource',device=a.device)
    if a.action=='train-base' and a.seed==42:
        subprocess.run([sys.executable,str(ROOT/'tools/seed42.py'),a.experiment,a.device],check=True);return
    if a.action=='train-base':
        subprocess.run([sys.executable,str(ROOT/'training'/('local_hard24' if a.experiment=='Hard24' and a.seed in (50,51,52) else 'base')/'second_seed.py'),'--experiment',a.experiment.lower(),'--previous',str(OUT/'runs'/a.experiment/'42'),'--out',str(OUT/'refits'/a.experiment/str(a.seed)),'--training-seed',str(a.seed),'--device',a.device],check=True)
    if a.action=='train-drsa':subprocess.run([sys.executable,str(ROOT/'training/drsa/run.py')],check=True)
    if a.action=='train-ddh':
        bundle=ROOT/'training/ddh';sys.path.insert(0,str(bundle));sys.path.insert(0,str(next(bundle.rglob('ddh_base.py')).parent))
        from colab_ddh_engine import Study
        study=Study(bundle,OUT/'ddh',mode='FAST',device=a.device);study.search();study.final_fits();study.evaluate_final()
if __name__=='__main__':main()
