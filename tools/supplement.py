from pathlib import Path
import csv, io, zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
out=root/'outputs';out.mkdir(exist_ok=True)
def read(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
checks=[]
for exp,filename in [('Hard24','duration_dependent_original_per_seed.csv'),('CR3','competing_risks_original_per_seed.csv')]:
 for row in read(root/'archive/supplementary/original_DDH_results_2026-11-02'/filename):
  seed=row['training_seed'];L=row['landmark'];target=row['target'];axis=['Survival','CIF_Death_A','CIF_Death_B'].index(target)
  with zipfile.ZipFile(root/'runs'/exp/f'seed_{seed}.zip') as archive:
   def array(name,key):
    with np.load(io.BytesIO(archive.read(f'evaluation/L{L}/'+name))) as z:return z[key]
   truth=array('reference.npz','truth');pred=array('DynamicDeepHit_direct_-1.npz','prediction')
  err=pred-truth
  if err.ndim==3:err=err[:,axis,:]
  mae=float(np.abs(err).mean());np.testing.assert_allclose(mae,float(row['MAE']),atol=1e-7,rtol=0)
  checks.append(dict(experiment=exp,seed=seed,L=L,target=target,MAE=mae))
with (out/'original_DDH_MAE.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=checks[0].keys());w.writeheader();w.writerows(checks)
print('Section S10:',len(checks),'original DDH MAEs reproduced')
summary=read(root/'archive/resource_results/budget_summary.csv')
fig,axes=plt.subplots(1,2,figsize=(8,3.6))
for ax,scenario in zip(axes,['Weibull','ReLU']):
 for model in sorted({r['model'] for r in summary}):
  rows=sorted([r for r in summary if r['model']==model and r['scenario']==scenario and int(r['landmark'])==0],key=lambda r:int(r['budget']))
  if rows:ax.errorbar([int(r['budget']) for r in rows],[float(r['mean_MAE']) for r in rows],yerr=[float(r['seed_SD_MAE']) for r in rows],marker='o',label=model.replace('Glazier','ATLAS'))
 for row in read(root/'archive/resource_results/budget_replicates.csv'):
  if row['scenario']==scenario and row['landmark']=='0' and row['budget']=='0':
   ax.axhline(float(row['MAE']),ls='--',label=row['model']+'-Lite')
 ax.set(title=scenario,xscale='log',xlabel='Trajectories per patient',ylabel='Baseline MAE');ax.legend(fontsize=7)
fig.tight_layout();(out/'figures').mkdir(exist_ok=True);fig.savefig(out/'figures/resource_budget_baseline.pdf')
