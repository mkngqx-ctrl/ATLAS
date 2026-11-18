from pathlib import Path
import csv,json,numpy as np
root=Path(__file__).resolve().parents[1]
integrated=root/'archive/supplementary/optimized_DRSA_integrated_2026-11-16'
def readcsv(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def check(label,passes,error):
 if not passes:raise ValueError((label,error))
 print(label,'maximum discrepancy',error)
def js(name,obj):
 folder=root/'outputs';folder.mkdir(exist_ok=True);(folder/name).write_text(json.dumps(obj,indent=2))
bootstrap=integrated/'bootstrap';pairs=readcsv(bootstrap/'paired_comparisons.csv');boot_checks=[]
for exp in ['Hard24','CR3']:
 with np.load(bootstrap/(exp+'_patient_errors.npz')) as z:err={k:z[k] for k in z.files}
 with np.load(bootstrap/(exp+'_patient_resamples.npz')) as z:res={k:z[k] for k in z.files}
 mapping={int(x):i for i,x in enumerate(res['baseline_patient_ids'])};names=err['model_names'].tolist();targets=err['target_names'].tolist()
 for L in ([0,20,23,24] if exp=='Hard24' else [0,10,20]):
  ids=err[f'patient_ids_L{L}'];e=err[f'patient_MAE_L{L}'];w=res['multiplicities'][:,[mapping[int(x)] for x in ids]].astype(float);den=w.sum(1);means=(w@e.reshape(len(ids),-1)).reshape(5000,len(targets),len(names))/den[:,None,None]
  for row in pairs:
   if row['experiment']!=exp or int(row['landmark'])!=L:continue
   k=targets.index(row['target']);a=names.index(row['ATLAS_model']);b=names.index(row['comparator']);draw=means[:,k,a]-means[:,k,b];ci=np.quantile(draw,[.025,.975]);relative=100*(1-means[:,k,a]/means[:,k,b]);rci=np.quantile(relative,[.025,.975])
   actual=np.array([e[:,k,a].mean(),e[:,k,b].mean(),(e[:,k,a]-e[:,k,b]).mean(),*ci,100*(1-e[:,k,a].mean()/e[:,k,b].mean()),*rci])
   recorded=np.array([float(row[key]) for key in ['ATLAS_MAE','comparator_MAE','difference_ATLAS_minus_comparator','CI_lower','CI_upper','relative_reduction_pct','relative_CI_lower','relative_CI_upper']])
   difference=float(np.max(np.abs(actual-recorded)));boot_checks.append(dict(experiment=exp,L=L,target=row['target'],ATLAS=row['ATLAS_model'],comparator=row['comparator'],absolute_difference=actual[2],CI_lower=ci[0],CI_upper=ci[1],relative_reduction_pct=actual[5],relative_CI_lower=rci[0],relative_CI_upper=rci[1],max_abs_difference=difference))
check('All240bootstrap contrasts independently reproduce',len(boot_checks)==240 and max(r['max_abs_difference'] for r in boot_checks)<1e-10,max(r['max_abs_difference'] for r in boot_checks));js('bootstrap_verification.json',boot_checks)
