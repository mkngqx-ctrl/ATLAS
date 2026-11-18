from pathlib import Path
import csv,json,hashlib,shutil,textwrap,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle,Patch

ROOT=Path(__file__).resolve().parent
OUT=Path(__file__).resolve().parent/'figures'
OUT.mkdir(exist_ok=True)
def read(name):return list(csv.DictReader((ROOT/name).open(encoding='utf8')))
rows=read('paired_comparisons.csv');heads=read('headline_comparisons.csv')
DIRECT=['CPH','Weibull','DRSA','DynamicDeepHit-Lite','SurvTransformer-Lite','DynamicDeepHit','SurvTransformer','SurvTransformer-Discrete']
names={m:('Weibull-MLP' if m=='Weibull' else 'Dynamic-DeepHit (optimized)' if m=='DynamicDeepHit' else 'DRSA (optimized)' if m=='DRSA' else m) for m in DIRECT}
blue='#156a9c';amber='#b97516';ink='#203640';gray='#9daeb6'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelcolor':ink,'text.color':ink,'xtick.color':ink,'ytick.color':ink,'axes.edgecolor':'#aab8bf','axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
captions=[]
def save(fig,name,caption):
 for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=300,facecolor='white')
 captions.append((name,caption));plt.close(fig)

# Figure 1: all prespecified headline settings, including the uncertain baseline.
fig,axes=plt.subplots(2,2,figsize=(12.8,8.5),sharex=True)
panels=[('Hard24','Survival','a  Duration-dependent: survival','ATLAS-Transformer'),('CR3','Survival','b  Competing risks: survival','ATLAS-GRU'),('CR3','CIF_Death_A','c  Competing risks: Death_A CIF','ATLAS-GRU'),('CR3','CIF_Death_B','d  Competing risks: Death_B CIF','ATLAS-GRU')]
for ax,(exp,target,title,atlas) in zip(axes.flat,panels):
 rr=sorted([r for r in heads if r['experiment']==exp and r['target']==target],key=lambda r:int(r['landmark']))
 y=np.arange(len(rr));ax.axvline(0,color='#617780',lw=1.1,ls='--',zorder=1)
 for k,r in enumerate(rr):
  mid=float(r['relative_reduction_pct']);lo=float(r['relative_CI_lower']);hi=float(r['relative_CI_upper']);color=blue if r['interval_position']=='below_zero' else amber
  ax.errorbar(mid,k,xerr=[[mid-lo],[hi-mid]],fmt='o',ms=6.5,color=color,ecolor=color,capsize=4,elinewidth=1.8,zorder=3)
  ax.annotate(f'{mid:.1f}%',(mid,k),xytext=(0,11),textcoords='offset points',ha='center',fontsize=9,fontweight='bold',color=color)
 ax.set_yticks(y,[f"L = {r['landmark']}   (n = {r['n_patients']})" for r in rr]);ax.set_ylim(len(rr)-.55,-.65)
 ax.set_title(title+'\n'+atlas,loc='left',fontsize=11.5,pad=14,fontweight='bold')
 ax.set_xlim(-15,105);ax.set_xticks([0,20,40,60,80,100]);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
 ax.tick_params(axis='y',length=0)
for ax in axes[1]:ax.set_xlabel('MAE reduction relative to strongest direct comparator (%)',fontsize=9.5)
fig.suptitle('ATLAS advantage with paired patient-bootstrap uncertainty',x=.07,ha='left',fontsize=17,fontweight='bold',y=.985)
fig.text(.07,.936,'Points: unresampled mean error reductions. Bars: pointwise 95% percentile intervals from 5,000 paired resamples.',fontsize=10)
fig.text(.07,.082,'Positive values favor ATLAS; zero indicates equal MAE. Amber indicates an interval that includes zero.',fontsize=9)
fig.text(.07,.056,'Duration-dependent: DynamicDeepHit-Lite at baseline; SurvTransformer-Discrete at L20/L23; Dynamic-DeepHit (optimized) at L24.',fontsize=8.2)
fig.text(.07,.034,'Competing risks: Weibull-MLP for baseline survival/Death B; Dynamic-DeepHit (optimized) for baseline Death A and all later targets.',fontsize=8.2)
fig.text(.07,.013,'Common Sobol sampler for ATLAS. Fits held fixed; intervals are not simultaneous and do not include model-selection or new-training-cohort uncertainty.',fontsize=8.6)
fig.subplots_adjust(left=.14,right=.97,bottom=.145,top=.85,hspace=.46,wspace=.37)
cap='Headline ATLAS configurations versus the strongest fitted direct comparator at each landmark and target. Relative MAE reduction is 100 times (comparator MAE minus ATLAS MAE) divided by comparator MAE; positive favors ATLAS. Horizontal intervals are pointwise 95% percentile bootstrap intervals from 5,000 paired resamples of the original 300 test patient IDs, with original landmark eligibility applied. Per-patient errors are averaged across saved sampling seeds and training runs before resampling. The direct comparator is fixed by the unresampled updated mean errors, not reselected during resampling. ATLAS uses common Sobol sampling; Transformer is shown for the duration-dependent experiment and GRU for competing risks. The baseline duration-dependent interval includes zero. These intervals condition on saved fits and realized sampling outputs, with no multiplicity or model-selection adjustment; reductions concern prediction error, not mortality risk.'
save(fig,'figure_1_headline_bootstrap',cap)

norm=TwoSlopeNorm(vmin=-100,vcenter=0,vmax=100);cmap=plt.get_cmap('RdBu')
lookup={(r['experiment'],r['target'],int(r['landmark']),r['ATLAS_model'],r['comparator']):r for r in rows}
def heat(ax,exp,target,architectures,Ls,show_y=True):
 columns=[(L,a) for L in Ls for a in architectures]
 values=np.zeros((len(DIRECT),len(columns)))
 for i,d in enumerate(DIRECT):
  for j,(L,a) in enumerate(columns):values[i,j]=float(lookup[exp,target,L,a,d]['relative_reduction_pct'])
 im=ax.imshow(values,cmap=cmap,norm=norm,aspect='auto',interpolation='nearest')
 for i,d in enumerate(DIRECT):
  for j,(L,a) in enumerate(columns):
   r=lookup[exp,target,L,a,d];v=values[i,j]
   ax.text(j,i,f'{v:.0f}',ha='center',va='center',fontsize=8.7,color='white' if abs(v)>58 else ink)
   if r['interval_position']=='includes_zero':ax.add_patch(Rectangle((j-.5,i-.5),1,1,fill=False,hatch='////',edgecolor='#444',lw=.8))
 ax.set_yticks(np.arange(8),[names[d] for d in DIRECT] if show_y else [])
 ax.set_xticks(np.arange(len(columns)),[a.replace('ATLAS-','') for _,a in columns],rotation=48,ha='right',fontsize=8.5)
 ax.set_xticks(np.arange(-.5,len(columns),1),minor=True);ax.set_yticks(np.arange(-.5,8,1),minor=True);ax.grid(which='minor',color='white',lw=.6);ax.tick_params(which='minor',bottom=False,left=False);ax.tick_params(axis='y',length=0)
 nmap={'Hard24':{0:300,20:102,23:78,24:57},'CR3':{0:300,10:251,20:179}}
 for g,L in enumerate(Ls):
  pos=g*len(architectures)+(len(architectures)-1)/2
  ax.text(pos,-.98,f'L = {L}\nn = {nmap[exp][L]}',ha='center',va='bottom',fontsize=9,clip_on=False)
  if g:ax.axvline(g*len(architectures)-.5,color='#425e6c',lw=1.4)
 return im

fig,ax=plt.subplots(figsize=(13,6.4))
im=heat(ax,'Hard24','Survival',['ATLAS-GRU','ATLAS-RNN','ATLAS-Transformer'],[0,20,23,24])
fig.suptitle('Duration-dependent experiment: every ATLAS architecture versus every direct comparator',fontsize=14,fontweight='bold',x=.02,ha='left',y=.98)
fig.text(.02,.924,'Cell values: percentage reduction in mean survival-curve MAE. Blue favors ATLAS; red favors the comparator.',fontsize=10)
fig.subplots_adjust(left=.22,right=.89,bottom=.22,top=.77)
cax=fig.add_axes([.916,.25,.017,.49]);cb=fig.colorbar(im,cax=cax,ticks=[-100,-50,0,50,100]);cb.set_label('MAE reduction (%)',fontsize=10)
fig.text(.22,.065,'Hatching: paired pointwise 95% interval includes zero. All other displayed intervals favor ATLAS.',fontsize=9)
fig.text(.22,.025,'Common Sobol sampler; nine saved training fits. Intervals condition on fitted models; no multiple-comparison adjustment.',fontsize=9)
cap='Duration-dependent survival comparison for all three ATLAS architectures and all eight fitted direct comparators at four landmarks. Each cell is the percentage reduction in mean MAE relative to the row comparator. The colors use a common -100% to +100% scale centered at zero; blue denotes lower ATLAS error. Hatched cells indicate that the corresponding pointwise 95% paired patient-bootstrap interval for the MAE difference includes zero; lack of hatching does not constitute a simultaneous significance statement. The baseline Transformer/DynamicDeepHit-Lite cell is the only such exception. ATLAS means use common Sobol sampling and nine saved fits. Landmark samples and prediction windows differ, so changes across landmarks do not isolate the effect of additional history. Numerical values are rounded to the nearest whole percentage; full precision and intervals remain in paired_comparisons.csv.'
save(fig,'figure_2_duration_comparator_matrix',cap)

fig,axes=plt.subplots(1,3,figsize=(17.5,6.7))
for ax,target,title in zip(axes,['Survival','CIF_Death_A','CIF_Death_B'],['Survival','Death_A CIF','Death_B CIF']):
 im=heat(ax,'CR3',target,['ATLAS-GRU','ATLAS-Transformer'],[0,10,20],show_y=ax is axes[0])
 ax.set_title(title,fontsize=12,fontweight='bold',y=1.25)
fig.suptitle('Competing-risk experiment: ATLAS versus all eight direct comparators',x=.02,ha='left',fontsize=16,fontweight='bold',y=.98)
fig.text(.02,.917,'Cell values: percentage reduction in mean curve MAE. One shared color scale for survival and both competing-event CIFs.',fontsize=10)
fig.subplots_adjust(left=.165,right=.91,bottom=.25,top=.72,wspace=.1)
cax=fig.add_axes([.929,.29,.011,.4]);cb=fig.colorbar(im,cax=cax,ticks=[-100,-50,0,50,100]);cb.set_label('MAE reduction (%)',fontsize=9,labelpad=5)
fig.text(.165,.085,'Blue favors ATLAS. Hatching: paired pointwise 95% interval includes zero; this occurs only for baseline Transformer versus Weibull-MLP, Death_B CIF.',fontsize=9)
fig.text(.165,.045,'Common Sobol sampler; five saved training fits. 5,000 paired patient resamples. Fits held fixed; intervals are not simultaneous.',fontsize=9)
cap='Competing-risk comparison for both ATLAS architectures and all eight fitted direct comparators, separately for survival, Death_A CIF and Death_B CIF at landmarks 0, 10 and 20. Cells show relative reductions in mean MAE; the common scale and hatching have the same meanings as in Figure 2. All point estimates favor ATLAS, but the baseline Transformer versus Weibull-MLP Death_B CIF interval includes zero. ATLAS uses the common Sobol sampler and five saved training fits; predictions and errors are paired by patient. These are conditional pointwise uncertainty summaries, without adjustment for the displayed family of comparisons. Percentage error reductions are not reductions in clinical event risk.'
save(fig,'figure_3_competing_risk_comparator_matrix',cap)

lines=['S3 FIGURE CAPTIONS','16 November 2026; optimized DDH and DRSA.','']
for name,caption in captions:lines.extend([name,caption,''])
(OUT/'FIGURE_CAPTIONS.txt').write_text('\n'.join(lines),encoding='utf8')
for name in ['headline_comparisons.csv','paired_comparisons.csv','analysis_plan.json']:shutil.copy2(ROOT/name,OUT/name)
shutil.copy2(Path(__file__).resolve(),OUT/'plot_s3_advantage.py')
index=['<!doctype html><html><head><meta charset="utf-8"><title>S3 ATLAS advantage figures</title><style>body{font:16px/1.5 Arial;max-width:1500px;margin:35px auto;color:#203640}img{width:100%;height:auto}section{margin:50px 0}h1,h2{color:#164c61}</style></head><body><h1>ATLAS advantage: S3 paired patient bootstrap</h1><p>Scientific figures updated for optimized DDH and DRSA, 16 November 2026. Positive MAE reductions favor ATLAS. All 95% intervals are pointwise, conditional on saved fitted models; no multiple-comparison adjustment. Baseline exceptions are shown.</p>']
for name,caption in captions:index.append(f'<section><h2>{html.escape(name.replace("_"," "))}</h2><img src="{name}.png" alt="{html.escape(caption)}"><p>{html.escape(caption)}</p><p><a href="{name}.pdf">Vector PDF</a> | <a href="{name}.svg">Editable SVG</a> | <a href="{name}.png">PNG</a></p></section>')
index.append('</body></html>');(OUT/'index.html').write_text('\n'.join(index),encoding='utf8')
manifest={'files':[{'path':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='manifest.json']}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
print('Created 3 figures in PNG, PDF and SVG;',len(manifest['files']),'files at',OUT)
