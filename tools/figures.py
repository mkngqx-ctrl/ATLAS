from pathlib import Path
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import csv,json,shutil,re,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

R=Path(__file__).resolve().parents[1]
D=R/'archive/supplementary/optimized_DRSA_integrated_2026-11-16'
O=R/'outputs';F=O/'figures';F.mkdir(parents=True,exist_ok=True)
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
inputs=[D/'figure_assets/full_comparison_plot_values.csv',D/'bootstrap/paired_comparisons.csv',D/'bootstrap/headline_comparisons.csv']
sources={str(p):sha(p) for p in inputs};full,pairs,heads=map(read,inputs)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':9,'ytick.labelsize':9,'savefig.facecolor':'white'})
plotted=[]
def save(fig,name):
 fig.savefig(F/(name+'.pdf'));fig.savefig(F/(name+'.png'),dpi=200);plt.close(fig)
methodnames={'iid':'IID','sobol':'Sobol','stratified':'Stratified','antithetic':'Antithetic','importance':'Importance','weighted_branching':'Weighted branching','conditional_mc':'Conditional MC'}
def label(model,method=None):
 model={'DynamicDeepHit':'Dynamic-DeepHit (optimized)','DRSA':'DRSA (optimized)','Weibull':'Weibull-MLP'}.get(model,model)
 return model+' / '+methodnames[method] if model.startswith('ATLAS-') else model+(' / IID' if model=='Oracle sampler' else '')
for exp in ['Hard24','CR3']:
 for L in [0,20]:
  rr=[r for r in full if r['experiment']==exp and int(r['landmark'])==L];base=[r for r in rr if r['target']=='Survival' and r['metric']=='MAE'];base.sort(key=lambda r:float(r['mean']));order=[(r['model'],r['method']) for r in base];N=len(order);ys=np.arange(N)
  panels=[('Survival','MAE','Full-curve MAE'),('Survival','post24_MAE','MAE at steps 24–30'),('Survival','RMSE','Full-curve RMSE')] if exp=='Hard24' else [('Survival','MAE','Survival MAE'),('CIF_Death_A','MAE','Death A CIF MAE'),('CIF_Death_B','MAE','Death B CIF MAE')]
  seeds=[42,43,44,45,50,51,52,62,63] if exp=='Hard24' else [42,43,45,46,47]
  fig,axes=plt.subplots(1,3,figsize=(9.5,5.9),sharey=True)
  for ax,(target,metric,title) in zip(axes,panels):
   lookup={(r['model'],r['method']):r for r in rr if r['target']==target and r['metric']==metric};mean=np.array([float(lookup[k]['mean']) for k in order]);sd=np.array([0 if k[0] in ['CPH','Oracle sampler'] else float(lookup[k]['between_SD']) for k in order]);colors=['#8b8b8b' if k[0] in ['CPH','Oracle sampler'] else '#8064a2' if k[0].startswith('ATLAS-') else '#168b82' for k in order]
   ax.barh(ys,mean,xerr=sd,color=colors,alpha=.62,height=.74,error_kw={'elinewidth':.7,'capsize':1.7,'ecolor':'#333'})
   for j,seed in enumerate(seeds):
    xx=[];yy=[]
    for i,k in enumerate(order):
     if k[0] in ['CPH','Oracle sampler']:continue
     vals=json.loads(lookup[k]['seeds']);xx.append(float(vals[str(seed)]));yy.append(i+(j-(len(seeds)-1)/2)*(.055 if exp=='Hard24' else .09))
    ax.scatter(xx,yy,s=10,marker=['o','s','^','D','v'][j%5],color=plt.get_cmap('tab10')(j),zorder=3,label=str(seed))
   ax.set_yticks(ys);ax.tick_params(axis='y',length=0);ax.set_title(title,pad=7);ax.set_xlabel('Error (lower is better)');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);ax.set_ylim(N-.6,-.6)
   maximum=max([float(v) for k in order for v in json.loads(lookup[k]['seeds']).values()]+list(mean+sd));ax.set_xlim(0,maximum*1.06);ax.locator_params(axis='x',nbins=4)
   for k in order:plotted.append(dict(experiment=exp,landmark=L,target=target,metric=metric,model=k[0],method=k[1],mean=float(lookup[k]['mean']),between_SD=float(lookup[k]['between_SD']),seeds=json.loads(lookup[k]['seeds'])))
  axes[0].set_yticklabels([label(*k) for k in order],fontsize=8.5)
  fig.suptitle(('Duration-dependent' if exp=='Hard24' else 'Competing risks')+f': all models and samplers, L = {L}',fontsize=11,fontweight='bold',y=.99)
  handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,title='Training seed (individual markers)',title_fontsize=8.5,fontsize=8.5,ncol=5 if exp=='Hard24' else 5,loc='upper center',bbox_to_anchor=(.63,.957),frameon=False,columnspacing=1.15,handletextpad=.35)
  fig.subplots_adjust(left=.30,right=.99,bottom=.10,top=.82 if exp=='Hard24' else .84,wspace=.16)
  save(fig,('hard24_nine_seed' if exp=='Hard24' else 'cr3_five_seed')+f'_comparison_L{L}')

blue='#156a9c';amber='#b97516';ink='#203640'
fig,axes=plt.subplots(2,2,figsize=(9.5,5.5),sharex=True)
for ax,(exp,target,title,atlas) in zip(axes.flat,[('Hard24','Survival','a  Duration-dependent survival','ATLAS-Transformer'),('CR3','Survival','b  Competing-risks survival','ATLAS-GRU'),('CR3','CIF_Death_A','c  Death A CIF','ATLAS-GRU'),('CR3','CIF_Death_B','d  Death B CIF','ATLAS-GRU')]):
 rr=sorted([r for r in heads if r['experiment']==exp and r['target']==target],key=lambda r:int(r['landmark']));ax.axvline(0,color='#617780',ls='--',lw=.8)
 for k,r in enumerate(rr):
  mid,lo,hi=map(lambda n:float(r[n]),['relative_reduction_pct','relative_CI_lower','relative_CI_upper']);color=blue if r['interval_position']=='below_zero' else amber;ax.errorbar(mid,k,xerr=[[mid-lo],[hi-mid]],fmt='o',ms=5,color=color,capsize=3,elinewidth=1.3);ax.annotate(f'{mid:.1f}%',(mid,k),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9,fontweight='bold',color=color)
 ax.set_yticks(range(len(rr)),[f"L = {r['landmark']} (n = {r['n_patients']})" for r in rr]);ax.set_ylim(len(rr)-.45,-.8);ax.set_title(title+'\n'+atlas,loc='left',fontsize=10,pad=8);ax.set_xlim(-15,105);ax.set_xticks([0,20,40,60,80,100]);ax.grid(axis='x',alpha=.2);ax.tick_params(axis='y',length=0)
for ax in axes[1]:ax.set_xlabel('MAE reduction relative to strongest direct comparator (%)',fontsize=9)
fig.suptitle('ATLAS advantage: paired patient-bootstrap intervals',fontsize=12,fontweight='bold',y=.99)
fig.text(.02,.018,'Positive values favor ATLAS. Bars: pointwise 95% intervals; amber: interval includes zero.',fontsize=9)
fig.subplots_adjust(left=.14,right=.985,bottom=.13,top=.84,hspace=.75,wspace=.48);save(fig,'figure_1_headline_bootstrap')

DIRECT=['CPH','Weibull','DRSA','DynamicDeepHit-Lite','SurvTransformer-Lite','DynamicDeepHit','SurvTransformer','SurvTransformer-Discrete'];lookup={(r['experiment'],r['target'],int(r['landmark']),r['ATLAS_model'],r['comparator']):r for r in pairs};norm=TwoSlopeNorm(vmin=-100,vcenter=0,vmax=100)
def heat(ax,exp,target,arch,Ls,show_y):
 columns=[(L,a) for L in Ls for a in arch];values=np.array([[float(lookup[exp,target,L,a,d]['relative_reduction_pct']) for L,a in columns] for d in DIRECT]);im=ax.imshow(values,cmap='RdBu',norm=norm,aspect='auto',interpolation='nearest')
 for i,d in enumerate(DIRECT):
  for j,(L,a) in enumerate(columns):
   v=values[i,j];ax.text(j,i,f'{v:.0f}',ha='center',va='center',fontsize=9,color='white' if abs(v)>58 else ink)
   if lookup[exp,target,L,a,d]['interval_position']=='includes_zero':ax.add_patch(Rectangle((j-.5,i-.5),1,1,fill=False,hatch='////',edgecolor='#444',lw=.8))
 ax.set_yticks(range(8),[label(d) for d in DIRECT] if show_y else []);ax.set_xticks(range(len(columns)),[a.replace('ATLAS-','') for _,a in columns],rotation=55,ha='right',fontsize=9);ax.set_xticks(np.arange(-.5,len(columns),1),minor=True);ax.set_yticks(np.arange(-.5,8,1),minor=True);ax.grid(which='minor',color='white',lw=.7);ax.tick_params(which='minor',bottom=False,left=False);ax.tick_params(axis='y',length=0,labelsize=9)
 nmap={'Hard24':{0:300,20:102,23:78,24:57},'CR3':{0:300,10:251,20:179}}
 for g,L in enumerate(Ls):
  ax.text(g*len(arch)+(len(arch)-1)/2,-.83,f'L = {L}\nn = {nmap[exp][L]}',ha='center',va='bottom',fontsize=9,clip_on=False)
  if g:ax.axvline(g*len(arch)-.5,color='#425e6c',lw=1.3)
 return im
fig,ax=plt.subplots(figsize=(9.5,4.7));im=heat(ax,'Hard24','Survival',['ATLAS-GRU','ATLAS-RNN','ATLAS-Transformer'],[0,20,23,24],True);fig.suptitle('Duration-dependent survival: all ATLAS–comparator contrasts',fontsize=12,fontweight='bold',y=.99);fig.subplots_adjust(left=.27,right=.86,bottom=.24,top=.79);cax=fig.add_axes([.89,.25,.018,.52]);cb=fig.colorbar(im,cax=cax,ticks=[-100,-50,0,50,100]);cb.set_label('MAE reduction (%)',fontsize=9);fig.text(.27,.018,'Blue favors ATLAS; red favors the comparator. Hatching: 95% interval includes zero.',fontsize=9);save(fig,'figure_2_duration_comparator_matrix')
fig,axes=plt.subplots(1,3,figsize=(9.5,5.2))
for i,(ax,target,title) in enumerate(zip(axes,['Survival','CIF_Death_A','CIF_Death_B'],['Survival','Death A CIF','Death B CIF'])):
 im=heat(ax,'CR3',target,['ATLAS-GRU','ATLAS-Transformer'],[0,10,20],i==0);ax.set_title(title,fontsize=10,fontweight='bold',y=1.21)
fig.suptitle('Competing risks: all ATLAS–comparator contrasts',fontsize=12,fontweight='bold',y=.99);fig.subplots_adjust(left=.27,right=.88,bottom=.25,top=.77,wspace=.13);cax=fig.add_axes([.9,.27,.013,.48]);cb=fig.colorbar(im,cax=cax,ticks=[-100,-50,0,50,100]);cb.ax.tick_params(labelsize=8);cb.set_label('MAE reduction (%)',fontsize=9);fig.text(.27,.02,'Blue favors ATLAS. Hatching: pointwise 95% interval includes zero.',fontsize=9);save(fig,'figure_3_competing_risk_comparator_matrix')

# Verify plotted statistics are exactly the saved figure inputs.
assert len(plotted)==len(full)
original={(r['experiment'],int(r['landmark']),r['target'],r['metric'],r['model'],r['method']):r for r in full}
for row in plotted:
 k=tuple(row[n] for n in ['experiment','landmark','target','metric','model','method']);r=original[k];assert row['mean']==float(r['mean']) and row['between_SD']==float(r['between_SD']) and row['seeds']==json.loads(r['seeds'])
assert all(sha(Path(p))==h for p,h in sources.items())
(O/'plot_verification.json').write_text(json.dumps(dict(full_comparison_series=len(plotted),bootstrap_contrasts=len(pairs),headline_contrasts=len(heads),source_input_hashes=sources,source_inputs_unchanged=True,plotted_full_comparison_statistics_identical=True),indent=2))

