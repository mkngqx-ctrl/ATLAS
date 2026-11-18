"""S3: paired patient bootstrap of original manuscript comparisons; no retraining."""
from pathlib import Path
import csv,json,hashlib,time,sys,html,shutil
import numpy as np

ROOT=Path(r'C:/Users/moeni/Documents/ATLAS_work')
DATA=ROOT/'current/working/ATLAS_SiM_data_and_code'
MANUSCRIPT=ROOT/'current/working/ATLAS_SiM_manuscript'
OUT=Path(__file__).resolve().parent/'S3_paired_patient_bootstrap_2026-10-30'
OUT.mkdir(exist_ok=True)
B=5000;SEED=20261030
SAMPLING_SEEDS=[41001,52001,63001,74001,85001]
DIRECT=['CPH','Weibull','DRSA','DynamicDeepHit-Lite','SurvTransformer-Lite','DynamicDeepHit','SurvTransformer','SurvTransformer-Discrete']
SPECS={'Hard24':dict(Ls=[0,20,23,24],nlive=3,targets=['Survival'],atlas=['ATLAS-GRU','ATLAS-RNN','ATLAS-Transformer'],headline='ATLAS-Transformer',n_runs=9,cohort_hash='676d66bf67e67a52dba3be50cfe7925d3b49f8f5e9ec9876913ed45146e03c54',summary='hard24_nine_training_runs/hard24_verified_summary.csv'),
 'CR3':dict(Ls=[0,10,20],nlive=7,targets=['Survival','CIF_Death_A','CIF_Death_B'],atlas=['ATLAS-GRU','ATLAS-Transformer'],headline='ATLAS-GRU',n_runs=5,cohort_hash='db08a45363fd873f7d62d5736e49fbb86df4f199109d56a5b7d683c38d8f2c15',summary='cr3_five_training_runs/verified_summary.csv')}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def writecsv(name,rows):
 with (OUT/name).open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def readcsv(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def relocate(text):
 text=text.replace('\\','/')
 for old,new in [('C:/Users/moeni/Documents/Codex/2026-09-13/thi',ROOT/'history/2026-09-13/thi'),('C:/Users/moeni/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an',ROOT/'history/2026-10-01/referenced-chatgpt-conversation-this-is-an')]:text=text.replace(old,str(new).replace('\\','/'))
 return Path(text)
def manuscript_hashes():return {str(p.relative_to(MANUSCRIPT)):sha(p) for p in MANUSCRIPT.rglob('*') if p.is_file()}
before=manuscript_hashes();start=time.perf_counter()
plan=dict(date='2026-10-30',resamples=B,random_seed=SEED,RNG='NumPy default_rng; separate SeedSequence children for Hard24 and CR3, in that order',
 population='Original 300 held-out test patients in each experiment; later landmarks restrict the resampled patient IDs by the archived live/observed eligibility mask',
 comparison_set='Every original ATLAS architecture with common Sobol, versus all eight original direct comparators; MMLR/HAMMLR and oracle controls excluded from this S3 analysis',
 patient_error='Mean absolute prediction-reference difference over scored times; average these patient errors over sampling seeds within fit, then equally over training fits; never average curves before scoring',
 bootstrap='5000 draws of 300 patient IDs with replacement, represented by multinomial multiplicity weights; reuse one weight matrix across all models, targets and landmarks within each experiment',
 intervals='Pointwise 95% percentile intervals; numpy.quantile method=linear; no multiple-comparison adjustment',
 contrast='ATLAS MAE minus direct comparator MAE; negative favors ATLAS',
 headline='Fixed headline ATLAS-Transformer for Hard24, ATLAS-GRU for CR3; strongest direct comparator fixed at each setting by original mean MAE, not reselected inside bootstrap',
 secondary='Every original pair and per-model MAE interval; strongest-direct contrast for every ATLAS architecture; optional reselected-best-direct sensitivity is not included',
 interpretation='Conditional on saved fits, selected model specifications, realized sampling outputs and the fixed synthetic cohort; does not include training-sample variation, new-cohort variation, model-selection uncertainty or clinical transport',
 analysis_status='Frozen before calculation in this execution; no manuscript edits')
(OUT/'analysis_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf8')
inventory=readcsv(DATA/'classical_comparators/pilot_results/source_run_inventory.csv')
comparison_rows=[];model_rows=[];headline=[];best_by_atlas=[];file_rows=[];cohort_rows=[];point_checks=[];runs_checked=[];counts_summary=[]
rngs=[np.random.default_rng(s) for s in np.random.SeedSequence(SEED).spawn(2)]
for ei,(exp,spec) in enumerate(SPECS.items()):
 source_rows=sorted([r for r in inventory if r['experiment']==exp],key=lambda r:int(r['training_seed']))
 assert len(source_rows)==spec['n_runs']
 roots=[relocate(r['root']) for r in source_rows]
 with np.load(roots[0]/'checkpoints/cohort.npz') as z:c={k:z[k] for k in z.files}
 train,val,test=[set(map(int,c[k])) for k in ['train_index','val_index','test_index']]
 assert len(train)==40000 and len(val)==len(test)==5000 and not(train&val or train&test or val&test)
 with np.load(roots[0]/'evaluation/L0/reference.npz') as z:baseline=z['patient_ids']
 assert len(baseline)==300 and len(set(baseline))==300 and set(baseline)<=test
 config=spec['atlas']+DIRECT;nmodels=len(config)
 reference={}
 for L in spec['Ls']:
  with np.load(roots[0]/f'evaluation/L{L}/reference.npz') as z:ref={k:z[k] for k in z.files}
  ids=ref['patient_ids'];truth=ref['truth'];truth=truth[:,None,:] if truth.ndim==2 else truth
  expected=baseline if L==0 else baseline[c['states'][baseline,L-1]<spec['nlive']]
  np.testing.assert_array_equal(ids,expected)
  reference[L]=(ids,ref['times'],truth)
 errors={L:np.zeros((len(reference[L][0]),len(spec['targets']),nmodels)) for L in spec['Ls']}
 reference_CPH={};per_run=[]
 for si,(root,sr) in enumerate(zip(roots,source_rows)):
  cohort_path=root/'checkpoints/cohort.npz';actual_hash=sha(cohort_path);assert actual_hash==spec['cohort_hash']==sr['cohort_sha256']
  cohort_rows.append(dict(experiment=exp,training_seed=int(sr['training_seed']),cohort_path=str(cohort_path),sha256=actual_hash))
  for L in spec['Ls']:
   ids,times,truth=reference[L]
   with np.load(root/f'evaluation/L{L}/reference.npz') as z:
    np.testing.assert_array_equal(z['patient_ids'],ids);np.testing.assert_array_equal(z['times'],times)
    tt=z['truth'];tt=tt[:,None,:] if tt.ndim==2 else tt
    np.testing.assert_allclose(tt,truth,atol=1e-15,rtol=0)
   for j,model in enumerate(config):
    method='sobol' if model.startswith('ATLAS-') else 'direct'
    labels=[model.replace('ATLAS-','Glazier-')] if model!='DynamicDeepHit-Lite' else ['DynamicDeepHIt-Lite','DynamicDeepHit-Lite']
    folder=root/f'evaluation/L{L}';files=[]
    for label in labels:
     files=sorted(folder.glob(f'{label}_{method}_*.npz'))
     if files:break
    assert len(files)==(5 if method=='sobol' else 1),(exp,sr['training_seed'],L,model,len(files))
    if method=='sobol':assert sorted(int(f.stem.rsplit('_',1)[1]) for f in files)==SAMPLING_SEEDS
    local=[]
    for f in files:
     with np.load(f) as z:
      p=z['prediction'];p=p[:,None,:] if p.ndim==2 else p
      if 'patient_ids' in z:np.testing.assert_array_equal(z['patient_ids'],ids)
      if 'times' in z:np.testing.assert_array_equal(z['times'],times)
     assert p.shape==truth.shape and np.isfinite(p).all()
     assert set(ids)<=test and not(set(ids)&train or set(ids)&val)
     local.append(np.abs(p-truth).mean(-1))
     file_rows.append(dict(experiment=exp,training_seed=int(sr['training_seed']),landmark=L,model=model,method=method,path=str(f),sha256=sha(f),n_patients=len(ids)))
     if model=='CPH':
      if L not in reference_CPH:reference_CPH[L]=p
      else:np.testing.assert_allclose(p,reference_CPH[L],atol=1e-12,rtol=0)
    run_error=np.mean(local,axis=0)
    errors[L][:,:,j]+=run_error/len(roots)
    for k,target in enumerate(spec['targets']):per_run.append(dict(training_seed=int(sr['training_seed']),landmark=L,model=model,method=method,target=target,MAE=float(run_error[:,k].mean())))
  print('Loaded',exp,'training seed',sr['training_seed'],flush=True)
 runs_checked+=per_run
 # Reproduce published per-run and aggregate metrics before calculating intervals.
 summaries=readcsv(DATA/'supplementary'/spec['summary'])
 fullkeys={(r['model'],r['method'],int(r['landmark']),r['target']):float(r['mean'] if exp=='Hard24' else r['mean_MAE']) for r in summaries}
 controls=readcsv(DATA/f'results_20260919/{"full_hard24_add_history_v4" if exp=="Hard24" else "full_cr3cov_v2"}_summary.csv')
 for r in controls:
  if r['model']=='CPH':fullkeys['CPH','direct',int(r['landmark']),r.get('target','Survival')]=float(r['MAE'])
 for L in spec['Ls']:
  for k,target in enumerate(spec['targets']):
   for j,model in enumerate(config):
    method='sobol' if model.startswith('ATLAS-') else 'direct';mean=float(errors[L][:,k,j].mean());published=fullkeys[model,method,L,target]
    assert abs(mean-published)<1e-12,(exp,L,target,model,mean,published)
    point_checks.append(dict(experiment=exp,landmark=L,target=target,model=model,method=method,recomputed_MAE=mean,published_MAE=published,absolute_discrepancy=abs(mean-published)))
 # One patient-ID resample matrix per experiment; reuse it for every comparison.
 weights=rngs[ei].multinomial(300,np.full(300,1/300),size=B).astype(np.int16)
 assert np.all(weights.sum(1)==300)
 np.savez_compressed(OUT/f'{exp}_patient_resamples.npz',baseline_patient_ids=baseline,multiplicities=weights,random_seed=SEED)
 packed=dict(baseline_patient_ids=baseline,model_names=np.array(config),target_names=np.array(spec['targets']))
 boot_arrays={};quant=lambda a:np.quantile(a,[.025,.975],method='linear')
 for L in spec['Ls']:
  ids,times,truth=reference[L];mask=np.isin(baseline,ids);np.testing.assert_array_equal(baseline[mask],ids)
  w=weights[:,mask].astype(float);den=w.sum(1)
  assert (den>0).all()
  nlow,nhigh=quant(den)
  counts_summary.append(dict(experiment=exp,landmark=L,original_n=len(ids),minimum_bootstrap_n=int(den.min()),percentile_2_5_n=float(nlow),percentile_97_5_n=float(nhigh),maximum_bootstrap_n=int(den.max())))
  e=errors[L];bm=(w@e.reshape(len(ids),-1)/den[:,None]).reshape(B,len(spec['targets']),nmodels)
  # A non-resampled unit multiplicity vector must return the original point estimates.
  np.testing.assert_allclose(e.mean(0),(np.ones(len(ids))@e.reshape(len(ids),-1)/len(ids)).reshape(len(spec['targets']),nmodels),atol=1e-15,rtol=0)
  packed[f'patient_ids_L{L}']=ids;packed[f'patient_MAE_L{L}']=e;packed[f'times_L{L}']=times
  boot_arrays[f'model_MAE_L{L}']=bm
  for k,target in enumerate(spec['targets']):
   direct_indices=[config.index(m) for m in DIRECT]
   strongest_index=min(direct_indices,key=lambda j:float(e[:,k,j].mean()))
   strongest_model=config[strongest_index]
   for j,model in enumerate(config):
    lo,hi=quant(bm[:,k,j])
    model_rows.append(dict(experiment=exp,landmark=L,target=target,model=model,method='sobol' if model.startswith('ATLAS-') else 'direct',n_patients=len(ids),MAE=float(e[:,k,j].mean()),CI_lower=float(lo),CI_upper=float(hi),training_fits=1 if model=='CPH' else spec['n_runs'],resamples=B))
   for atlas in spec['atlas']:
    a=config.index(atlas)
    for comparator in DIRECT:
     d=config.index(comparator);difference=e[:,k,a]-e[:,k,d]
     bd=bm[:,k,a]-bm[:,k,d];lo,hi=quant(bd)
     relative=100*(bm[:,k,d]-bm[:,k,a])/bm[:,k,d];rlo,rhi=quant(relative)
     row=dict(experiment=exp,landmark=L,target=target,ATLAS_model=atlas,ATLAS_method='sobol',comparator=comparator,n_patients=len(ids),ATLAS_MAE=float(e[:,k,a].mean()),comparator_MAE=float(e[:,k,d].mean()),difference_ATLAS_minus_comparator=float(difference.mean()),CI_lower=float(lo),CI_upper=float(hi),relative_reduction_pct=float(100*(e[:,k,d].mean()-e[:,k,a].mean())/e[:,k,d].mean()),relative_CI_lower=float(rlo),relative_CI_upper=float(rhi),patient_fraction_ATLAS_lower=float(np.mean(difference<0)),interval_position='below_zero' if hi<0 else ('above_zero' if lo>0 else 'includes_zero'),strongest_direct_original_mean=comparator==strongest_model,headline_architecture=atlas==spec['headline'],resamples=B)
     comparison_rows.append(row)
     if comparator==strongest_model:
      best_by_atlas.append(row)
      if atlas==spec['headline']:headline.append(row)
  print('Bootstrapped',exp,'landmark',L,'eligible n',len(ids),flush=True)
 np.savez_compressed(OUT/f'{exp}_patient_errors.npz',**packed)
 np.savez_compressed(OUT/f'{exp}_bootstrap_model_means.npz',model_names=np.array(config),target_names=np.array(spec['targets']),**boot_arrays)
writecsv('paired_comparisons.csv',comparison_rows);writecsv('headline_comparisons.csv',headline);writecsv('strongest_direct_by_ATLAS.csv',best_by_atlas);writecsv('model_MAE_intervals.csv',model_rows);writecsv('landmark_resample_sizes.csv',counts_summary)
writecsv('prediction_sources.csv',file_rows);writecsv('cohort_sources.csv',cohort_rows);writecsv('published_mean_reproduction.csv',point_checks)
assert len(comparison_rows)==240 and len(headline)==13 and len(model_rows)==134
assert manuscript_hashes()==before,'Manuscript changed during analysis'
verification=dict(prediction_files_read=len(file_rows),cohort_copies_checked=len(cohort_rows),published_model_means_reproduced=len(point_checks),max_mean_discrepancy=max(r['absolute_discrepancy'] for r in point_checks),all_evaluation_patients_in_test=True,any_training_or_validation_overlap=False,baseline_n=300,all_landmark_ID_orders_match=True,shared_resamples_across_models_targets_landmarks=True,zero_empty_landmark_resamples=True,manuscript_unchanged=True,numpy=np.__version__,python=sys.version,elapsed_seconds=time.perf_counter()-start)
(OUT/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf8')
(OUT/'manuscript_unchanged_hashes.json').write_text(json.dumps(before,indent=2),encoding='utf8')
summary={}
for exp,spec in SPECS.items():
 rr=[r for r in comparison_rows if r['experiment']==exp];best=[r for r in best_by_atlas if r['experiment']==exp];hh=[r for r in headline if r['experiment']==exp]
 summary[exp]=dict(all_pairs=len(rr),pairs_CI_below_zero=sum(r['interval_position']=='below_zero' for r in rr),pairs_CI_includes_zero=sum(r['interval_position']=='includes_zero' for r in rr),pairs_CI_above_zero=sum(r['interval_position']=='above_zero' for r in rr),post_baseline_pairs=sum(r['landmark']>0 for r in rr),post_baseline_pairs_CI_below_zero=sum(r['landmark']>0 and r['interval_position']=='below_zero' for r in rr),strongest_direct_pairs=len(best),strongest_direct_CI_below_zero=sum(r['interval_position']=='below_zero' for r in best),headline_settings=len(hh),headline_intervals_CI_below_zero=sum(r['interval_position']=='below_zero' for r in hh))
(OUT/'summary_counts.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
print('HEADLINE RESULTS')
for r in headline:print(r['experiment'],r['landmark'],r['target'],r['comparator'],r['difference_ATLAS_minus_comparator'],r['CI_lower'],r['CI_upper'],r['interval_position'])
print('SUMMARY',json.dumps(summary));print('VERIFICATION',json.dumps(verification))
