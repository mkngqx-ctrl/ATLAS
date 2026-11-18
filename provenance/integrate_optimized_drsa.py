import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
from pathlib import Path
import csv,json,re,shutil,hashlib,math
import numpy as np

R=Path(r'C:/Users/moeni/Documents/ATLAS_work/current');M=R/'working/ATLAS_SiM_manuscript';D=R/'working/ATLAS_SiM_data_and_code'
STUDY=R/'review/DRSA_optimization_2026-11-04';OLD=R/'review/S3_optimized_DDH_bootstrap_2026-11-02'
OUT=R/'review/DRSA_manuscript_update_2026-11-16';OUT.mkdir(exist_ok=False)
STAGE=OUT/'manuscript';shutil.copytree(M,STAGE)
DATA=OUT/'data';DATA.mkdir();BOOT=DATA/'bootstrap';BOOT.mkdir()
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def jsonout(p,obj):p.write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')
rows=[dict(r) for r in read(STUDY/'reports/all_models_per_training_run.csv') if r['model']!='DRSA-original']
for r in rows:
 if r['model']=='DRSA-optimized':r['model']='DRSA'
summary=[dict(r) for r in read(STUDY/'reports/all_models_summary.csv') if r['model']!='DRSA-original']
for r in summary:
 if r['model']=='DRSA-optimized':r['model']='DRSA'
write(DATA/'updated_all_models_per_training_run.csv',rows);write(DATA/'updated_all_models_summary.csv',summary)
def selected(exp,L,target):return next(r for r in summary if r['experiment']==exp and int(r['landmark'])==L and r['target']==target and r['model']=='DRSA')
for name,exp,target in [('table_single_cov_main.tex','Hard24','Survival'),('table_cr_survival.tex','CR3','Survival'),('table_cr_cause_a.tex','CR3','CIF_Death_A'),('table_cr_cause_b.tex','CR3','CIF_Death_B')]:
 p=STAGE/'sections'/name;s=p.read_text(encoding='utf-8');Ls=[0,20,23,24] if exp=='Hard24' else [0,10,20]
 line='DRSA (optimized) & '+' & '.join(f"{float(selected(exp,L,target)['mean_MAE']):.6f}" for L in Ls)+r' \\'
 s,n=re.subn(r'^DRSA &.*$',lambda _:line,s,flags=re.M);assert n==1;p.write_text(s,encoding='utf-8')
for name,exp in [('table_hard24_training_dispersion.tex','Hard24'),('table_cr_training_dispersion.tex','CR3')]:
 p=STAGE/'sections'/name;out=[];L=None;target='Survival';count=0
 for line in p.read_text(encoding='utf-8').splitlines():
  if r'\textit{' in line:
   match=re.search(r'L=(\d+)',line)
   if match:L=int(match[1])
   target='CIF_Death_A' if 'Death A' in line else 'CIF_Death_B' if 'Death B' in line else 'Survival'
  if line.startswith('DRSA &'):
   q=selected(exp,L,target);ending=r'\\*' if line.rstrip().endswith(r'\\*') else r'\\'
   line='DRSA (optimized) & '+' & '.join(f"{float(q[k]):.6f}" for k in ['between_SD','minimum','maximum'])+' & --- '+ending;count+=1
  out.append(line)
 assert count==(4 if exp=='Hard24' else 9);p.write_text('\n'.join(out)+'\n',encoding='utf-8')
# Check individual-run strongest-comparator tables; no DRSA result becomes strongest.
best=[]
for exp,L,target in sorted(set((r['experiment'],int(r['landmark']),r['target']) for r in rows)):
 for seed in sorted(set(r['training_seed'] for r in rows if r['experiment']==exp)):
  rr=[r for r in rows if r['experiment']==exp and int(r['landmark'])==L and r['target']==target and r['training_seed']==seed and r['method']=='direct']
  b=min(rr,key=lambda r:float(r['MAE']));assert b['model']!='DRSA';best.append(dict(experiment=exp,landmark=L,target=target,seed=seed,model=b['model'],MAE=b['MAE']))
write(DATA/'strongest_direct_by_seed.csv',best)
# Preserve current paired-bootstrap design; replace only DRSA patient errors.
paired=read(OLD/'paired_comparisons.csv');intervals=read(OLD/'model_MAE_intervals.csv');heads=read(OLD/'headline_comparisons.csv');updated_count=0
with np.load(STUDY/'reports/optimized_DRSA_patient_errors.npz') as z:newerrors={k:z[k] for k in z.files}
for exp in ['Hard24','CR3']:
 with np.load(OLD/(exp+'_patient_errors.npz')) as z:data={k:z[k] for k in z.files}
 with np.load(OLD/(exp+'_patient_resamples.npz')) as z:res={k:z[k] for k in z.files}
 j=data['model_names'].tolist().index('DRSA');index={int(pid):i for i,pid in enumerate(res['baseline_patient_ids'])}
 for L in ([0,20,23,24] if exp=='Hard24' else [0,10,20]):
  errors=newerrors[f'{exp}_patient_MAE_L{L}'];data[f'patient_MAE_L{L}'][:,:,j]=errors
  ids=data[f'patient_ids_L{L}'];w=res['multiplicities'][:,[index[int(pid)] for pid in ids]].astype(float);den=w.sum(1)
  for ti,target in enumerate(data['target_names']):
   b=errors[:,ti];draw_b=w@b/den;ci=np.quantile(draw_b,[.025,.975])
   record=next(q for q in intervals if q['experiment']==exp and int(q['landmark'])==L and q['target']==target and q['model']=='DRSA');record.update(MAE=float(b.mean()),CI_lower=float(ci[0]),CI_upper=float(ci[1]))
   for row in paired:
    if not(row['experiment']==exp and int(row['landmark'])==L and row['target']==target and row['comparator']=='DRSA'):continue
    ai=data['model_names'].tolist().index(row['ATLAS_model']);a=data[f'patient_MAE_L{L}'][:,ti,ai];delta=a-b;draw_delta=w@delta/den;ci=np.quantile(draw_delta,[.025,.975]);relative=100*(1-(w@a/den)/draw_b);rci=np.quantile(relative,[.025,.975])
    row.update(ATLAS_MAE=float(a.mean()),comparator_MAE=float(b.mean()),difference_ATLAS_minus_comparator=float(delta.mean()),CI_lower=float(ci[0]),CI_upper=float(ci[1]),relative_reduction_pct=float(100*(1-a.mean()/b.mean())),relative_CI_lower=float(rci[0]),relative_CI_upper=float(rci[1]),patient_fraction_ATLAS_lower=float(np.mean(a<b)),interval_position='below_zero' if ci[1]<0 else 'above_zero' if ci[0]>0 else 'includes_zero');updated_count+=1
 np.savez_compressed(BOOT/(exp+'_patient_errors.npz'),**data);shutil.copy2(OLD/(exp+'_patient_resamples.npz'),BOOT/(exp+'_patient_resamples.npz'))
assert updated_count==30
write(BOOT/'paired_comparisons.csv',paired);write(BOOT/'model_MAE_intervals.csv',intervals);write(BOOT/'headline_comparisons.csv',heads)
for h in heads:assert h['comparator']!='DRSA'
plan=json.loads((OLD/'analysis_plan.json').read_text());plan['optimized_DRSA_update']='Only DRSA patient errors replaced using verified optimized fits; original5000multiplicity weights reused;30contrasts updated; strongest comparators unchanged.';jsonout(BOOT/'analysis_plan.json',plan)
# S8 full numerical table: replace 30 DRSA contrasts; headline tables unchanged.
p=STAGE/'sections/supp_patient_bootstrap.tex';s=p.read_text(encoding='utf-8');out=[];exp=None;count=0
for line in s.splitlines():
 if r'\multicolumn{6}' in line and 'Duration-dependent experiment' in line:exp='Hard24'
 if r'\multicolumn{6}' in line and 'Competing-risks experiment' in line:exp='CR3'
 if ' & DRSA & ' in line:
  a,L,target,*_=line.split(' & ');target={'S':'Survival','A':'CIF_Death_A','B':'CIF_Death_B'}[target]
  q=next(r for r in paired if r['experiment']==exp and r['ATLAS_model']=='ATLAS-'+a and int(r['landmark'])==int(L) and r['target']==target and r['comparator']=='DRSA')
  line=f"{a} & {L} & "+{'Survival':'S','CIF_Death_A':'A','CIF_Death_B':'B'}[target]+f" & DRSA (optimized) & {float(q['difference_ATLAS_minus_comparator']):.6f} [{float(q['CI_lower']):.6f}, {float(q['CI_upper']):.6f}] & {float(q['relative_reduction_pct']):.1f} [{float(q['relative_CI_lower']):.1f}, {float(q['relative_CI_upper']):.1f}]"+r'\\';count+=1
 out.append(line)
assert count==30;s='\n'.join(out)+'\n';s=s.replace('including Dynamic-DeepHit (optimized).','including Dynamic-DeepHit (optimized) and DRSA (optimized).');p.write_text(s,encoding='utf-8')
# Reuse established bootstrap plotting layout; update DRSA legend only.
plot=(OLD/'plot_s3_advantage.py').read_text(encoding='utf-8')
plot=plot.replace("else 'Dynamic-DeepHit (optimized)' if m=='DynamicDeepHit' else m)","else 'Dynamic-DeepHit (optimized)' if m=='DynamicDeepHit' else 'DRSA (optimized)' if m=='DRSA' else m)")
(BOOT/'plot_s3_advantage.py').write_text(plot,encoding='utf-8')
# Methods, Results and Discussion.
p=STAGE/'main.tex';s=p.read_text(encoding='utf-8')
s=s.replace('deep recurrent survival analysis (DRSA)~','deep recurrent survival analysis (DRSA; optimized)~')
anchor='The two SurvTransformer variants select their checkpoints using their combined validation losses.'
addition="DRSA was also optimized separately for each benchmark using a staged validation-based search over recurrent width and depth, learned covariate projection, composite-loss weight, learning rate, regularization and batch size. The search used 24 screening candidates and 12 refinement entries per experiment, followed by three-seed confirmation of two finalists. Configurations and checkpoints were selected using observed-outcome conditional validation negative log-likelihood averaged equally across the evaluated landmarks, without using analytical reference curves or test-set MAE. The selected configurations were fitted using the same nine and five training seeds and are labelled DRSA (optimized). This follow-up was motivated by the initial comparison results; Supplementary Section~S1.2 gives the search and selected settings.\n\n"
assert s.count(anchor)==1;s=s.replace(anchor,addition+anchor)
s=s.replace('DRSA, which uses covariates only, has baseline MAE 0.146263.','DRSA (optimized), which uses covariates only, has baseline MAE 0.118561.')
s=s.replace('Dynamic-DeepHit (optimized) selects checkpoints using landmark-averaged validation negative log-likelihood','Dynamic-DeepHit (optimized) and DRSA (optimized) select checkpoints using landmark-averaged validation negative log-likelihood')
s=s.replace('Dynamic-DeepHit received the additional validation-based search described in Supplementary Section~S1, while the other model configurations were retained.','Dynamic-DeepHit and DRSA received the additional validation-based searches described in Supplementary Section~S1; the other model configurations were retained.')
s=s.replace("DRSA's comparatively high baseline curve MAEs (0.146 and 0.131;",'The comparatively high baseline curve MAEs for DRSA (optimized) (0.119 and 0.083;')
assert '0.146263' not in s and '0.146 and 0.131' not in s
p.write_text(s,encoding='utf-8')
# Detailed S1.2 specification and frozen search history; no historical-result tables.
p=STAGE/'sections/supp_model_settings.tex';s=p.read_text(encoding='utf-8');s=s.replace('CPH, Weibull-MLP, DRSA, DynamicDeepHit-Lite','CPH, Weibull-MLP, DRSA (optimized), DynamicDeepHit-Lite')
old=next(x for x in s.split('\n\n') if x.startswith('DRSA represents interval probabilities'))
new=r'''DRSA (optimized) represents interval probabilities using an LSTM over the 30-step prediction grid. A learned ReLU projection transforms the training-standardized baseline covariates before normalized prediction time is appended. The model does not receive the observed health-state sequence. The duration-dependent head predicts a sigmoid death hazard, while the competing-risks head uses a softmax over conditional survival and the two death causes. Survival and CIFs are obtained by probability propagation; survival beyond step 30 remains possible. These are adaptations of the published model families rather than exact ports of all released settings~\citep{ren2019,martinsson2017,lee2018deephit}.

The DRSA objective is $\alpha L_z+(1-\alpha)L_c$, where $L_z$ is the uncensored event-time negative log-likelihood and $L_c$ combines the event-status and censoring terms, averaged over the minibatch. The competing-risks event-time term uses the observed cause. The search retained this composite objective and considered $\alpha\in\{0.05,0.10,0.25,0.50,0.75,0.90\}$. Both selected configurations use $\alpha=0.50$, giving equal weights to the two objective components. Observed deaths use the recorded terminal time for both event-time and event-status terms; this differs from an observation scheme that follows individuals beyond their event time.

The search was frozen before optimization and used the original 40,000 training and 5,000 validation patients in each experiment. Screening crossed the six alpha values with four architectures: 32 units/one layer/no projection; 128 units/one layer/no projection; 128 units/one layer/ReLU-64; and 256 units/two layers/ReLU-128. These 24 candidates used seed 42 and cumulative budgets of 5, 15 and 30 epochs. Two candidates per alpha were retained after the first budget and one per alpha after the second, so all six alpha values remained represented through screening. The two leading alpha values then entered refinement. Each was crossed with five fixed new recipes and its screening winner, giving 12 refinement entries. The new recipes explored widths 64, 128 and 256, one or two layers, ReLU projections of width 64 or 128, learning rates from 0.0001 to 0.003, dropout 0 to 0.2, L2 coefficients 0 to 0.0001, and batch sizes 128 or 256. Refinement used cumulative budgets of 10, 30 and 60 epochs, retaining six then four entries. Two finalists were confirmed using training seeds 42, 43 and 45; the selected configuration minimized their mean best validation score.

Every checkpoint and promotion decision used conditional observed-outcome validation negative log-likelihood averaged equally over the evaluated landmarks, using all eligible validation patients. Eligibility required the recorded event or censoring time to exceed the landmark. The score did not include the training penalty. Analytical reference curves and test-set MAE were not used for selection. Confirmation and final training used Adam with $(\beta_1,\beta_2)=(0.9,0.99)$, gradient clipping at 5, a maximum of 200 epochs and patience 15, with a minimum validation improvement of $10^{-6}$. The L2 penalty was $\lambda\|\theta\|_2^2/(2\cdot128)$, with the fixed denominator preserving its strength across batch sizes. Dropout was applied after the covariate projection and recurrent output, and between recurrent layers when applicable. Final fits used all training patients, one record per patient; selected confirmation fits were reused unchanged for seeds 42, 43 and 45.

\begin{center}\begin{minipage}{\textwidth}\centering\small
\begin{tabular}{lrr}\toprule
Selected DRSA setting & Duration-dependent & Competing risks\\\midrule
LSTM width / layers & 256 / 1 & 256 / 2\\
ReLU projection width & 128 & 128\\
Alpha & 0.50 & 0.50\\
Learning rate & 0.003 & 0.0015\\
Dropout & 0.20 & 0.10\\
L2 coefficient & 0.0001 & 0.00001\\
Batch size & 128 & 256\\
Selected epoch across final fits & 12--66 & 11--15\\\bottomrule
\end{tabular}
\captionof{table*}{Selected DRSA configurations. The nine duration-dependent and five competing-risk fits use the original training seeds and evaluation patients. Selected epochs refer to retained checkpoints, not stopping epochs. Per-fit settings and logs are supplied in the accompanying data/code package.}
\end{minipage}\end{center}'''
# Use an unnumbered caption so existing supplement table numbers remain stable.
new=new.replace(r'\captionof{table*}',r'\captionof*{table}')
assert s.count(old)==1;s=s.replace(old,new);p.write_text(s,encoding='utf-8')
p=STAGE/'sections/appendix_metrics.tex';s=p.read_text(encoding='utf-8');anchor='The recorded timings therefore do not constitute a controlled speed comparison.'
s=s.replace(anchor,'DRSA (optimized) was fitted locally on an NVIDIA GeForce RTX 3060 using PyTorch 2.11.0 with CUDA 13.0; its per-fit elapsed times and hardware metadata are supplied with the optimized results. '+anchor);p.write_text(s,encoding='utf-8')
# Captions identify both optimized comparators.
for name in ['figures_hard24_nine_seed.tex','figures_cr3_five_seed.tex']:
 p=STAGE/'sections'/name;s=p.read_text(encoding='utf-8');s=s.replace('Dynamic-DeepHit (optimized)','Dynamic-DeepHit (optimized) and DRSA (optimized)') if 'Dynamic-DeepHit (optimized)' in s else s
 p.write_text(s,encoding='utf-8')
# Write code/data exports for the current paper only; history stays in research archive.
epochrows=[]
for exp in ['Hard24','CR3']:
 for r in json.loads((STUDY/exp/'final_fit_inventory.json').read_text()):epochrows.append(dict(experiment=exp,seed=r['seed'],selected_epoch=r['best_epoch'],stopping_epoch=r['epochs_completed'],validation_NLL=r['best_validation_NLL']))
write(DATA/'optimized_DRSA_selected_epochs.csv',epochrows)
jsonout(OUT/'preparation_validation.json',dict(main_DRSA_settings=13,updated_bootstrap_contrasts=updated_count,unchanged_bootstrap_contrasts=210,headline_comparators_unchanged=True,historical_DRSA_results_in_paper=False))
shutil.copy2(__file__,OUT/'integrate_optimized_drsa.py')
print('Prepared manuscript stage, numerical tables, bootstrap data and search descriptions:',OUT)
