import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
from pathlib import Path
import subprocess,sys,shutil,json
R=Path(r'C:/Users/moeni/Documents/ATLAS_work/current');BASE=R/'review/DRSA_manuscript_update_2026-11-16';STAGE=BASE/'manuscript';ASSETS=BASE/'data/figure_assets';ASSETS.mkdir()
template=(R/'working/ATLAS_SiM_data_and_code/supplementary/optimized_DDH_integrated_2026-11-02/figure_assets/regenerate_figures.py').read_text(encoding='utf-8')
body=template[template.index('# Full comparison panels'):template.index("(OUT/'validation.json')")]
body=body.replace("if key[0]!='DynamicDeepHit':","if key[0] not in ['DynamicDeepHit','DRSA']:")
body=body.replace("if model=='DynamicDeepHit':return 'Dynamic-DeepHit (optimized)'","if model=='DynamicDeepHit':return 'Dynamic-DeepHit (optimized)'\n    if model=='DRSA':return 'DRSA (optimized)'")
header=r'''from pathlib import Path
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('C:/Users/moeni/Documents/ATLAS_work/current');D=R/'working/ATLAS_SiM_data_and_code';BASE=R/'review/DRSA_manuscript_update_2026-11-16';OUT=BASE/'data/figure_assets'
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(name,rows):
 with (OUT/name).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def save(fig,name,dpi=180,tight=True):
 kw={'bbox_inches':'tight'} if tight else {}
 fig.savefig(OUT/(name+'.pdf'),**kw);fig.savefig(OUT/(name+'.png'),dpi=dpi,**kw);plt.close(fig)
rows=read(BASE/'data/updated_all_models_per_training_run.csv')
oldhard=read(D/'supplementary/hard24_nine_training_runs/hard24_verified_per_checkpoint.csv');oldcr=read(D/'supplementary/cr3_five_training_runs/verified_per_training_run.csv')
inv=read(R/'review/S2_validation_loss_2026-10-30/checkpoint_selection_inventory.csv');runs={exp:Path(next(z for z in inv if z['experiment']==exp)['root']) for exp in ['Hard24','CR3']}
series_data=[]
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
'''
(ASSETS/'regenerate_full_comparisons.py').write_text(header+'\n'+body,encoding='utf-8')
subprocess.run([sys.executable,str(ASSETS/'regenerate_full_comparisons.py')],check=True)
bootstrap=BASE/'data/bootstrap/plot_s3_advantage.py';plot=bootstrap.read_text(encoding='utf-8').replace('2 November 2026; optimized DDH bootstrap update.','16 November 2026; optimized DDH and DRSA.').replace('Scientific figures updated for optimized DDH, 2 November 2026.','Scientific figures updated for optimized DDH and DRSA, 16 November 2026.')
bootstrap.write_text(plot,encoding='utf-8');subprocess.run([sys.executable,str(bootstrap)],check=True)
for exp,prefix in [('Hard24','hard24_nine_seed_comparison'),('CR3','cr3_five_seed_comparison')]:
 for L in [0,20]:
  for ext in ['pdf','png']:shutil.copy2(ASSETS/f'{prefix}_L{L}.{ext}',STAGE/'figures'/f'{prefix}_L{L}.{ext}')
for name in ['figure_1_headline_bootstrap','figure_2_duration_comparator_matrix','figure_3_competing_risk_comparator_matrix']:
 shutil.copy2(BASE/'data/bootstrap/figures'/f'{name}.pdf',STAGE/'figures'/f'{name}.pdf')
for name in ['figures_hard24_nine_seed.tex','figures_cr3_five_seed.tex']:
 p=STAGE/'sections'/name;s=p.read_text(encoding='utf-8').replace('and DRSA (optimized) uses','and DRSA (optimized) use');p.write_text(s,encoding='utf-8')
p=STAGE/'main.tex';s=p.read_text(encoding='utf-8').replace('The comparatively high baseline curve MAEs for DRSA (optimized) (0.119 and 0.083;','The baseline curve MAEs for DRSA (optimized) (0.119 and 0.083;');p.write_text(s,encoding='utf-8')
shutil.copy2(__file__,BASE/'generate_drsa_update_figures.py')
print('Regenerated4full comparisons and3bootstrap figure PDFs.')
