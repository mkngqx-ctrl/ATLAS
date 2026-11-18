from pathlib import Path
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import csv,io,json,math,re,zipfile
import numpy as np

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'DDH_step3_numerical_update_2026-11-02';OUT.mkdir(exist_ok=False)
R=Path('C:/Users/moeni/Documents/ATLAS_work/current');D=R/'working/ATLAS_SiM_data_and_code';M=R/'working/ATLAS_SiM_manuscript'
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def loadnp(raw):
    with np.load(io.BytesIO(raw)) as z:return {k:z[k] for k in z.files}
inv=read(R/'review/S2_validation_loss_2026-10-30/checkpoint_selection_inventory.csv')
allrows=[];new=[]
for exp,path in [('Hard24',D/'supplementary/hard24_nine_training_runs/hard24_verified_per_checkpoint.csv'),('CR3',D/'supplementary/cr3_five_training_runs/verified_per_training_run.csv')]:
    for z in read(path):
        allrows.append(dict(experiment=exp,target=z['target'],landmark=int(z['landmark']),model=z['model'],method=z['method'],training_seed=int(z['training_seed']),MAE=float(z['MAE']),RMSE=float(z['RMSE']),post24_MAE=float(z['post24_MAE']) if z.get('post24_MAE') else '',rollout_SD=float(z['rollout_SD']) if z.get('rollout_SD') else '',repeats=int(z['repeats']),implementation='existing'))
with zipfile.ZipFile(R/'review/S2_DDH_returned_results_audit_2026-11-02/ATLAS_DDH_FAST_v1_results.zip') as z:
    for exp,seeds in [('Hard24',[42,43,44,45,50,51,52,62,63]),('CR3',[42,43,45,46,47])]:
        run=Path(next(q for q in inv if q['experiment']==exp)['root'])
        for L in ([0,20,23,24] if exp=='Hard24' else [0,10,20]):
            ref=loadnp((run/f'evaluation/L{L}/reference.npz').read_bytes());truth=ref['truth'];truth=truth[:,None,:] if truth.ndim==2 else truth
            if exp=='Hard24':
                control=loadnp((run/f'evaluation/L{L}/CPH_direct_-1.npz').read_bytes())['prediction'];control=control[:,None,:] if control.ndim==2 else control
                delta=control-truth
                for seed in seeds:allrows.append(dict(experiment=exp,target='Survival',landmark=L,model='CPH',method='direct',training_seed=seed,MAE=float(np.abs(delta).mean()),RMSE=float(np.sqrt(np.square(delta).mean())),post24_MAE=float(np.abs(delta[:,:,ref['times']>=24]).mean()),rollout_SD='',repeats=1,implementation='unchanged numerical control'))
            for seed in seeds:
                p=loadnp(z.read(f'evaluation/{exp}/seed_{seed}_L{L}.npz'));np.testing.assert_array_equal(p['patient_ids'],ref['patient_ids']);np.testing.assert_array_equal(p['times'],ref['times']);delta=p['prediction']-truth
                for k,target in enumerate(['Survival'] if exp=='Hard24' else ['Survival','CIF_Death_A','CIF_Death_B']):
                    row=dict(experiment=exp,target=target,landmark=L,model='DynamicDeepHit',method='direct',training_seed=seed,MAE=float(np.abs(delta[:,k]).mean()),RMSE=float(np.sqrt(np.square(delta[:,k]).mean())),post24_MAE=float(np.abs(delta[:,k,ref['times']>=24]).mean()) if exp=='Hard24' else '',rollout_SD='',repeats=1,implementation='optimized validation-selected DDH')
                    old=next(q for q in allrows if q['experiment']==exp and q['landmark']==L and q['target']==target and q['training_seed']==seed and q['model']=='DynamicDeepHit')
                    old.update(row);new.append(row)
assert len(new)==81
summary=[]
for exp,L,target,model,method in sorted(set((q['experiment'],q['landmark'],q['target'],q['model'],q['method']) for q in allrows)):
    group=[q for q in allrows if (q['experiment'],q['landmark'],q['target'],q['model'],q['method'])==(exp,L,target,model,method)];a=np.array([q['MAE'] for q in group]);sampling=[q['rollout_SD'] for q in group if q['rollout_SD']!='']
    summary.append(dict(experiment=exp,landmark=L,target=target,model=model,method=method,mean_MAE=float(a.mean()),between_SD=float(a.std(ddof=1)),minimum=float(a.min()),maximum=float(a.max()),median=float(np.median(a)),n_runs=len(group),mean_RMSE=float(np.mean([q['RMSE'] for q in group])),mean_post24_MAE=float(np.mean([q['post24_MAE'] for q in group])) if exp=='Hard24' else '',pooled_rollout_SD=float(np.sqrt(np.mean(np.square(sampling)))) if sampling else ''))
def select(exp,L,target,model,method='direct'):
    return next(q for q in summary if (q['experiment'],q['landmark'],q['target'],q['model'],q['method'])==(exp,L,target,model,method))
best_by_setting=[];best_by_seed=[];tests=[]
for exp,L,target in sorted(set((q['experiment'],q['landmark'],q['target']) for q in allrows)):
    direct=[q for q in summary if q['experiment']==exp and q['landmark']==L and q['target']==target and q['method']=='direct']
    assert len(direct)==8
    best=min(direct,key=lambda q:q['mean_MAE']);best_by_setting.append(dict(experiment=exp,landmark=L,target=target,model=best['model'],MAE=best['mean_MAE']))
    seeds=sorted(set(q['training_seed'] for q in allrows if q['experiment']==exp))
    for seed in seeds:
        choices=[q for q in allrows if q['experiment']==exp and q['landmark']==L and q['target']==target and q['training_seed']==seed and q['method']=='direct'];b=min(choices,key=lambda q:q['MAE'])
        best_by_seed.append(dict(experiment=exp,landmark=L,target=target,seed=seed,model=b['model'],MAE=b['MAE']))
    for model,method in sorted(set((q['model'],q['method']) for q in allrows if q['experiment']==exp and q['model'].startswith('ATLAS-'))):
        for rule in ['fixed strongest mean comparator','best direct within each training run']:
            differences=[];against_ddh=[]
            for seed in seeds:
                a=next(q for q in allrows if q['experiment']==exp and q['landmark']==L and q['target']==target and q['training_seed']==seed and q['model']==model and q['method']==method)['MAE']
                b=next(q for q in allrows if q['experiment']==exp and q['landmark']==L and q['target']==target and q['training_seed']==seed and q['model']==best['model'] and q['method']=='direct')['MAE'] if rule.startswith('fixed') else next(q['MAE'] for q in best_by_seed if q['experiment']==exp and q['landmark']==L and q['target']==target and q['seed']==seed)
                ddh=next(q['MAE'] for q in new if q['experiment']==exp and q['landmark']==L and q['target']==target and q['training_seed']==seed)
                differences.append(a-b);against_ddh.append(a-ddh)
            wins=sum(d<-1e-14 for d in differences);losses=sum(d>1e-14 for d in differences);n=wins+losses
            tests.append(dict(experiment=exp,landmark=L,target=target,ATLAS_model=model,method=method,comparison_rule=rule,comparator=best['model'] if rule.startswith('fixed') else 'setting-specific',wins=wins,losses=losses,ties=len(seeds)-n,one_sided_sign_p=sum(math.comb(n,j) for j in range(wins,n+1))/2**n if n else 1.,mean_difference=float(np.mean(differences)),wins_vs_optimized_DDH=sum(d<-1e-14 for d in against_ddh),training_runs=len(seeds)))
write('updated_all_models_per_training_run.csv',allrows);write('updated_all_models_summary.csv',summary);write('optimized_DDH_per_seed_metrics.csv',new);write('strongest_direct_by_setting.csv',best_by_setting);write('strongest_direct_by_seed.csv',best_by_seed);write('paired_training_run_sign_tests.csv',tests)
display=lambda name:'Dynamic-DeepHit (optimized)' if name=='DynamicDeepHit' else 'Weibull-MLP' if name=='Weibull' else name
edits={}
def edit(name,func):
    s=(M/'sections'/name).read_text(encoding='utf-8');newtext=func(s);assert s!=newtext;edits[name]=newtext
for name,exp,target in [('table_single_cov_main.tex','Hard24','Survival'),('table_cr_survival.tex','CR3','Survival'),('table_cr_cause_a.tex','CR3','CIF_Death_A'),('table_cr_cause_b.tex','CR3','CIF_Death_B')]:
    landmarks=[0,20,23,24] if exp=='Hard24' else [0,10,20]
    line='Dynamic-DeepHit (optimized) & '+' & '.join(f"{select(exp,L,target,'DynamicDeepHit')['mean_MAE']:.6f}" for L in landmarks)+r' \\'
    edit(name,lambda s,line=line:re.sub(r'^DynamicDeepHit &.*$',lambda _:line,s,flags=re.M))
for name,exp in [('table_hard24_training_dispersion.tex','Hard24'),('table_cr_training_dispersion.tex','CR3')]:
    s=(M/'sections'/name).read_text(encoding='utf-8');L=None;target='Survival';out=[];count=0
    for line in s.splitlines():
        if r'\textit{' in line and 'Landmark' in line or r'\textit{' in line and 'landmark' in line:
            match=re.search(r'L=(\d+)',line)
            if match:L=int(match[1])
            if exp=='CR3':target='CIF_Death_A' if 'Death A' in line else 'CIF_Death_B' if 'Death B' in line else 'Survival'
        if line.startswith('DynamicDeepHit &'):
            q=select(exp,L,target,'DynamicDeepHit');ending=r'\\*' if line.rstrip().endswith(r'\\*') else r'\\'
            line=f"Dynamic-DeepHit (optimized) & {q['between_SD']:.6f} & {q['minimum']:.6f} & {q['maximum']:.6f} & --- "+ending;count+=1
        out.append(line)
    assert count==(4 if exp=='Hard24' else 9),(name,count)
    edits[name]='\n'.join(out)+'\n'
for name,exp in [('table_hard24_checkpoint_scores.tex','Hard24'),('table_cr_checkpoint_scores.tex','CR3')]:
    s=(M/'sections'/name).read_text(encoding='utf-8');L=None;target='Survival';out=[];count=0
    for line in s.splitlines():
        if r'\textit{' in line:
            match=re.search(r'L=(\d+)',line)
            if match:L=int(match[1])
            if exp=='CR3':target='CIF_Death_A' if 'Death A' in line else 'CIF_Death_B' if 'Death B' in line else 'Survival'
        if re.match(r'^\d+ &',line):
            pieces=line.split(' & ');seed=int(pieces[0]);b=next(q for q in best_by_seed if q['experiment']==exp and q['landmark']==L and q['target']==target and q['seed']==seed)
            ending=r'\\*' if line.rstrip().endswith(r'\\*') else r'\\'
            line=' & '.join(pieces[:-2]+[display(b['model']),f"{b['MAE']:.6f} "+ending]);count+=1
        out.append(line)
    assert count==(36 if exp=='Hard24' else 45),(name,count)
    edits[name]='\n'.join(out)+'\n'
for name,text in edits.items():(OUT/name).write_text(text,encoding='utf-8')
statistics_report=dict(new_DDH_metric_rows=len(new),updated_table_files=list(edits),sign_test_rows=len(tests),headline_postbaseline_sign_tests=[q for q in tests if q['method']=='sobol' and q['comparison_rule'].startswith('best') and q['landmark']>0 and q['ATLAS_model']==('ATLAS-Transformer' if q['experiment']=='Hard24' else 'ATLAS-GRU')],vs_DDH_any_nonwin=[q for q in tests if q['comparison_rule'].startswith('fixed') and q['wins_vs_optimized_DDH']<q['training_runs']],strongest_direct=best_by_setting)
(OUT/'numerical_update_summary.json').write_text(json.dumps(statistics_report,indent=2))
print('Prepared 8 revised tables; all DDH accuracy metrics and paired-run statistics recalculated.')
print('Headline post-baseline signs:',[(q['experiment'],q['landmark'],q['target'],q['wins'],q['training_runs'],q['one_sided_sign_p']) for q in statistics_report['headline_postbaseline_sign_tests']])
print('ATLAS configurations with any per-seed nonwin against optimized DDH:',len(statistics_report['vs_DDH_any_nonwin']))
