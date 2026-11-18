from pathlib import Path
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import csv,io,json,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'DDH_step4_figures_2026-11-02';OUT.mkdir(exist_ok=False)
R=Path('C:/Users/moeni/Documents/ATLAS_work/current');D=R/'working/ATLAS_SiM_data_and_code';M=R/'working/ATLAS_SiM_manuscript'
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def npz(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
def write(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
rows=read(D/'supplementary/optimized_DDH_integrated_2026-11-02/updated_all_models_per_training_run.csv')
oldhard=read(D/'supplementary/hard24_nine_training_runs/hard24_verified_per_checkpoint.csv');oldcr=read(D/'supplementary/cr3_five_training_runs/verified_per_training_run.csv')
inv=read(R/'review/S2_validation_loss_2026-10-30/checkpoint_selection_inventory.csv');runs={exp:Path(next(z for z in inv if z['experiment']==exp)['root']) for exp in ['Hard24','CR3']}
series_data=[];axis_notes=[]
def save(fig,name,dpi=180,tight=True):
    kw={'bbox_inches':'tight'} if tight else {}
    fig.savefig(OUT/(name+'.pdf'),**kw)
    fig.savefig(OUT/(name+'.png'),dpi=dpi,**kw)
    plt.close(fig)
def mean(exp,L,model,method='direct',target='Survival'):
    a=[float(z['MAE']) for z in rows if z['experiment']==exp and int(z['landmark'])==L and z['model']==model and z['method']==method and z['target']==target];assert len(a)>0
    return float(np.mean(a))

# Figure 5: preserve original axes and every other plotted value.
original_overview=read(D/'supplementary/overview_plot_values.csv');updated=[]
for z in original_overview:
    z=dict(z)
    if z['model']=='DynamicDeepHit':z['mean_MAE']=mean(z['experiment'],int(z['landmark']),'DynamicDeepHit')
    updated.append(z)
styles=[('ATLAS-GRU','sobol','ATLAS-GRU / Sobol','#6b3ba1','o'),('ATLAS-Transformer','sobol','ATLAS-Transformer / Sobol','#b080df','s'),('SurvTransformer-Discrete','direct','SurvTransformer-Discrete','#1f61ad','^'),('SurvTransformer','direct','SurvTransformer','#41a6c5','v'),('DynamicDeepHit','direct','Dynamic-DeepHit\n(optimized)','#e86100','D'),('Oracle sampler','iid','Oracle sampler','#222222','x')]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':True,'axes.spines.right':True})
fig,axes=plt.subplots(1,2,figsize=(7.4,3.4))
for ax,exp,landmarks,count in zip(axes,['Hard24','CR3'],[[0,20,23,24],[0,10,20]],[9,5]):
    lines=[]
    for model,method,label,color,marker in styles:
        oldvals=[float(next(z['mean_MAE'] for z in original_overview if z['experiment']==exp and int(z['landmark'])==L and z['model']==model and z['method']==method)) for L in landmarks]
        newvals=[float(next(z['mean_MAE'] for z in updated if z['experiment']==exp and int(z['landmark'])==L and z['model']==model and z['method']==method)) for L in landmarks]
        if model!='DynamicDeepHit':assert oldvals==newvals
        line,=ax.plot(np.arange(len(landmarks)),oldvals,color=color,marker=marker,linewidth=1.35,markersize=4.2,label=label);lines.append((line,newvals))
    ax.set_yscale('log');ylim=ax.get_ylim()
    for line,newvals in lines:line.set_ydata(newvals)
    ax.set_ylim(ylim);ax.set_xticks(np.arange(len(landmarks)),[str(x) for x in landmarks]);ax.set_xlabel('Landmark');ax.grid(True,which='both',linewidth=.45,alpha=.45);ax.spines[['right','top']].set_visible(False)
    ax.set_title(('Duration-dependent' if exp=='Hard24' else 'Multi-covariate competing risks')+f'\n{count} training runs per neural model',pad=8)
axes[0].set_ylabel('Mean survival MAE (log scale)');h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,ncol=3,loc='lower center',bbox_to_anchor=(.52,-.01),frameon=False,columnspacing=1.7);fig.subplots_adjust(left=.085,right=.985,top=.82,bottom=.30,wspace=.22)
save(fig,'atlas_survival_comparison_multiseed');write('overview_plot_values.csv',updated)

# Individual patients: exact same covariate ranks and unchanged original other curves.
run=runs['Hard24'];folder=run/'evaluation/L0';ref=npz(folder/'reference.npz');cohort=npz(run/'checkpoints/cohort.npz');x=cohort['covariate'][ref['patient_ids']];order=np.argsort(x);indices=[order[round(q*(len(x)-1))] for q in [.1,.5,.9]]
draws=np.stack([npz(folder/f'Glazier-Transformer_sobol_{seed}.npz')['prediction'] for seed in [41001,52001,63001,74001,85001]])
with zipfile.ZipFile(R/'review/S2_DDH_returned_results_audit_2026-11-02/ATLAS_DDH_FAST_v1_results.zip') as z:
    newddh=npz(io.BytesIO(z.read('evaluation/Hard24/seed_42_L0.npz')));np.testing.assert_array_equal(newddh['patient_ids'],ref['patient_ids']);newddh=newddh['prediction'][:,0]
oldpred={name:npz(folder/f'{name}_direct_-1.npz')['prediction'] for name in ['DynamicDeepHIt-Lite','DynamicDeepHit','SurvTransformer-Discrete']}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,3,figsize=(11.5,6),sharex=True,gridspec_kw={'height_ratios':[2,1]});patient_records=[]
for col,i in enumerate(indices):
    t=np.arange(31);truth=np.r_[1,ref['truth'][i]];at=np.r_[1,draws[0,i]]
    axes[0,col].step(t,truth,where='post',color='black',lw=2,label='SFSA reference');axes[0,col].step(t,at,where='post',color='#6d4fa4',lw=1.8,label='ATLAS-Transformer / Sobol');axes[0,col].fill_between(t,np.r_[1,draws[:,i].min(0)],np.r_[1,draws[:,i].max(0)],step='post',color='#6d4fa4',alpha=.18)
    axes[1,col].step(t,(at-truth)*100,where='post',color='#6d4fa4');ddh_lines=[]
    for model,color,ls in [('DynamicDeepHIt-Lite','#15968b','--'),('DynamicDeepHit','#ba4d3b',':'),('SurvTransformer-Discrete','#315f9b','-.')]:
        label='Dynamic-DeepHit (optimized)' if model=='DynamicDeepHit' else model.replace('DynamicDeepHIt','DynamicDeepHit')
        p=np.r_[1,oldpred[model][i]];line0,=axes[0,col].step(t,p,where='post',color=color,ls=ls,label=label);line1,=axes[1,col].step(t,(p-truth)*100,where='post',color=color,ls=ls)
        if model=='DynamicDeepHit':ddh_lines=[line0,line1]
    axes[0,col].set_title(f'Calcium rank {int([.1,.5,.9][col]*100)}%: x={x[i]:.2f}');axes[0,col].set_ylim(-.02,1.03);axes[1,col].axhline(0,color='grey',lw=.6)
    for ax in axes[:,col]:ax.axvline(24,color='grey',lw=.8,ls=':');ax.grid(alpha=.15)
    axes[1,col].set_xlabel('Absolute step');oldlim=axes[1,col].get_ylim()
    ddh_lines[0].set_ydata(np.r_[1,newddh[i]]);newerror=np.r_[0,newddh[i]-ref['truth'][i]]*100;ddh_lines[1].set_ydata(newerror)
    axes[1,col].set_ylim(oldlim)
    if newerror.min()<oldlim[0] or newerror.max()>oldlim[1]:
        # Keep all plotted errors visible if the new curve reaches beyond the old range.
        low=min(oldlim[0],float(newerror.min())-1);high=max(oldlim[1],float(newerror.max())+1);axes[1,col].set_ylim(low,high);axis_notes.append(dict(figure='individual',patient_id=int(ref['patient_ids'][i]),old_limits=list(oldlim),new_limits=[low,high]))
    for j,tt in enumerate(ref['times']):patient_records.append(dict(patient_id=int(ref['patient_ids'][i]),covariate_percentile=int([.1,.5,.9][col]*100),x=float(x[i]),time=int(tt),reference=float(ref['truth'][i,j]),ATLAS_Transformer_Sobol=float(draws[0,i,j]),ATLAS_min=float(draws[:,i,j].min()),ATLAS_max=float(draws[:,i,j].max()),DDH_optimized=float(newddh[i,j]),DDH_Lite=float(oldpred['DynamicDeepHIt-Lite'][i,j]),SurvTransformer_Discrete=float(oldpred['SurvTransformer-Discrete'][i,j])))
axes[0,0].set_ylabel('KM reference / prediction');axes[1,0].set_ylabel('Error (percentage points)');fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,-.045));fig.tight_layout(rect=[0,.075,1,1]);save(fig,'updated_individual_baseline',240);write('individual_baseline_plot_values.csv',patient_records)

# Full comparison panels, preserving the original figure format and axis ranges.
methods={'iid':'IID','sobol':'Sobol','stratified':'Stratified','antithetic':'Antithetic','importance':'Importance','weighted_branching':'WeightedBranching','conditional_mc':'ConditionalMC'}
def label(model,method,exp):
    if model=='DynamicDeepHit':return 'Dynamic-DeepHit (optimized)'
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
                if key[0]!='DynamicDeepHit':
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
(OUT/'validation.json').write_text(json.dumps(dict(figures=6,original_other_model_values_unchanged=True,patient_ids=[int(ref['patient_ids'][i]) for i in indices],patient_covariates=[float(x[i]) for i in indices],axis_adjustments=axis_notes,source='verified optimized_DDH_integrated_2026-11-02 metrics and original seed-42 patient curves',S8_figures_changed=False),indent=2))
print('Regenerated all six approved figures; non-DDH plotted values verified unchanged.');print('Axis adjustments:',axis_notes)
