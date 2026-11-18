from pathlib import Path
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

# Full comparison panels, preserving the original figure format and axis ranges.
methods={'iid':'IID','sobol':'Sobol','stratified':'Stratified','antithetic':'Antithetic','importance':'Importance','weighted_branching':'WeightedBranching','conditional_mc':'ConditionalMC'}
def label(model,method,exp):
    if model=='DynamicDeepHit':return 'Dynamic-DeepHit (optimized)'
    if model=='DRSA':return 'DRSA (optimized)'
    if model.startswith('ATLAS-'):return model+'-'+methods[method]
    if model=='DynamicDeepHit-Lite':model='DynamicDeepHIt-Lite' # Preserve the existing Lite legend spelling.
    return model+(' / '+method if exp=='CR3' or method!='direct' else '')
def metric_rows(exp,L,old=False):
    if old:
        source=oldhard if exp=='Hard24' else oldcr
        selected=[dict(z,experiment=exp) for z in source if int(z['landmark'])==L]
        if exp=='Hard24':selected.extend(dict(z) for z in rows if z['experiment']==exp and int(z['landmark'])==L and z['model']=='CPH')
    else:selected=[dict(z) for z in rows if z['experiment']==exp and int(z['landmark'])==L]
    if exp=='Hard24':
        control=next(z for z in read(runs[exp]/'evaluation/summary.csv') if int(z['landmark'])==L and z['model']=='Oracle sampler' and z['method']=='iid')
        selected.append(dict(control,target='Survival',training_seed=42))
    return selected
def grouped(source,target,metric):
    ans={}
    for model,method in sorted(set((z['model'],z['method']) for z in source if z['target']==target)):
        rr=[z for z in source if z['target']==target and z['model']==model and z['method']==method];vals=np.array([float(z[metric]) for z in rr]);ans[(model,method)]=dict(mean=float(vals.mean()),sd=float(vals.std(ddof=1)) if len(vals)>1 else 0.,values={int(z['training_seed']):float(z[metric]) for z in rr})
    return ans
def old_limits(source,target,metric,exp):
    group=grouped(source,target,metric);fig,ax=plt.subplots()
    for j,((model,method),g) in enumerate(group.items()):
        ax.barh(j,g['mean'],xerr=0 if model in ['CPH','Oracle sampler'] else g['sd'])
        if model not in ['CPH','Oracle sampler']:ax.scatter(list(g['values'].values()),[j]*len(g['values']))
    lim=ax.get_xlim();plt.close(fig);return lim
for exp in ['Hard24','CR3']:
    for L in [0,20]:
        selected=metric_rows(exp,L);prior=metric_rows(exp,L,True);order=sorted(grouped(selected,'Survival','MAE'),key=lambda key:grouped(selected,'Survival','MAE')[key]['mean']);N=len(order);y=np.arange(N)
        if exp=='Hard24':
            plt.rcParams.update({'font.size':10,'axes.titlesize':12,'axes.labelsize':10,'axes.spines.top':True,'axes.spines.right':True})
            fig,axes=plt.subplots(1,3,figsize=(17,max(5,.32*N+2)),sharey=True,layout='constrained');seeds=[42,43,44,45,50,51,52,62,63];panels=[('Survival','MAE','Overall MAE'),('Survival','post24_MAE','MAE at steps 24-30'),('Survival','RMSE','RMSE')]
        else:
            plt.rcParams.update({'font.size':10,'axes.titlesize':12,'axes.labelsize':10,'figure.titlesize':14,'axes.spines.top':False,'axes.spines.right':False})
            fig,axes=plt.subplots(1,3,figsize=(21,10));seeds=[42,43,45,46,47];panels=[('Survival','MAE','Survival'),('CIF_Death_A','MAE','CIF_Death_A'),('CIF_Death_B','MAE','CIF_Death_B')]
        for ax,(target,metric,title) in zip(axes,panels):
            g=grouped(selected,target,metric);old=grouped(prior,target,metric);limits=old_limits(prior,target,metric,exp)
            for key in order:
                if key[0] not in ['DynamicDeepHit','DRSA']:
                    assert abs(g[key]['mean']-old[key]['mean'])<1e-13,(exp,L,target,metric,key)
            values=[g[key]['mean'] for key in order];errors=[0 if key[0] in ['CPH','Oracle sampler'] else g[key]['sd'] for key in order]
            if exp=='Hard24':
                colors=['#888888' if key[0] in ['CPH','Oracle sampler'] else '#8064A2' if key[0].startswith('ATLAS-') else '#168B82' for key in order]
                ax.barh(y,values,xerr=errors,color=colors,alpha=.55,capsize=2)
                for j,seed in enumerate(seeds):
                    for i,key in enumerate(order):
                        if key[0] in ['CPH','Oracle sampler']:continue
                        ax.scatter([g[key]['values'][seed]],[i],marker=['o','s','^','D','v'][j%5],s=18,color=f'C{j}',zorder=3,label=f'Training seed {seed}' if i==next(k for k,key2 in enumerate(order) if key2[0] not in ['CPH','Oracle sampler']) else None)
                ax.set_xlabel('Error; lower is better');ax.grid(axis='x',alpha=.2)
            else:
                colors=['#6650a4' if key[0].startswith('ATLAS-') else '#158a83' for key in order]
                ax.barh(y,values,color=colors,height=.72,alpha=.9,xerr=errors,error_kw={'elinewidth':1.,'capsize':2,'ecolor':'#252525'})
                for j,seed in enumerate(seeds):ax.scatter([g[key]['values'][seed] for key in order],y+(j-2)*.11,s=12,marker=['o','s','^','D','v'][j],color=['#111111','#444444','#777777','#aaaaaa','#cccccc'][j],zorder=4,label=f'Seed {seed}')
                ax.set_yticks(y,[label(*key,exp) for key in order],fontsize=8.3);ax.invert_yaxis();ax.set_xlabel('Curve MAE; mean ± 1 training-seed SD');ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True)
            ax.set_title(title);ax.set_xlim(limits)
            extrema=[v for key in order for v in g[key]['values'].values()]+[g[key]['mean']+g[key]['sd'] for key in order]
            if min(extrema)<limits[0] or max(extrema)>limits[1]:raise RuntimeError(('New values exceed preserved axis range',exp,L,target,metric,min(extrema),max(extrema),limits))
            for key in order:
                item=g[key];series_data.append(dict(experiment=exp,landmark=L,target=target,metric=metric,model=key[0],method=key[1],mean=item['mean'],between_SD=item['sd'],seeds=json.dumps(item['values'])))
        if exp=='Hard24':
            axes[0].set_yticks(y,[label(*key,exp) for key in order]);axes[0].invert_yaxis();axes[-1].legend(fontsize=8)
            fig.suptitle(f'Hard24: Survival, L={L}; training seeds '+', '.join(map(str,seeds))+'\nBars: mean; error bars: SD of checkpoint means; markers: individual checkpoints.\nSoftware/hardware may differ across runs; see runtime_provenance.csv.\nCPH/oracle shown from seed-42 source only, without training-variation error bars.',fontsize=10)
            save(fig,f'hard24_nine_seed_comparison_L{L}',180,tight=False)
        else:
            fig.suptitle(f'Two-disease SFSA, three covariates: L={L}\nTraining seeds 42, 43, 45, 46, 47',y=1.01);fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,-.045),ncol=5,fontsize=10,frameon=False);fig.tight_layout(w_pad=3);save(fig,f'cr3_five_seed_comparison_L{L}',300)
write('full_comparison_plot_values.csv',series_data)
