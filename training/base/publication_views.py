"""Export the two cross-run/patient benchmark views without opening a manuscript."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from atlas_figures import display_name
ROLLOUT_SEEDS=[41001,52001,63001,74001,85001]

def export_views(hard_run,cr_run,destination):
    hard_run=Path(hard_run).resolve();cr_run=Path(cr_run).resolve();destination=Path(destination).resolve()
    if any(destination==p or p in destination.parents for p in (hard_run,cr_run)):
        raise ValueError('Use a separate export directory')
    destination.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        for suffix in ('.png','.svg'):fig.savefig(destination/(name+suffix),dpi=220,bbox_inches='tight')
        plt.close(fig)
    folder=hard_run/'evaluation/L0'
    with np.load(folder/'reference.npz') as z:ref={k:z[k] for k in z.files}
    with np.load(hard_run/'checkpoints/cohort.npz') as z:x=z['covariate'][ref['patient_ids']]
    ranked=np.argsort(x);quantiles=[.1,.5,.9];indices=[ranked[round(q*(len(x)-1))] for q in quantiles]
    def prediction(path):
        with np.load(path) as z:return z['prediction']
    samples=np.stack([prediction(folder/f'Glazier-Transformer_sobol_{s}.npz') for s in ROLLOUT_SEEDS])
    direct=[('DynamicDeepHIt-Lite','#15968b','--'),('DynamicDeepHit','#ba4d3b',':'),('SurvTransformer-Discrete','#315f9b','-.')]
    fig,axes=plt.subplots(2,3,figsize=(12,6.5),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    for j,i in enumerate(indices):
        t=np.r_[0,ref['times']];truth=np.r_[1,ref['truth'][i]]
        axes[0,j].step(t,truth,where='post',color='black',lw=2,label='SFSA reference')
        axes[0,j].step(t,np.r_[1,samples[0,i]],where='post',color='#6d4fa4',label=display_name('Glazier-Transformer','sobol'))
        axes[0,j].fill_between(t,np.r_[1,samples[:,i].min(0)],np.r_[1,samples[:,i].max(0)],step='post',color='#6d4fa4',alpha=.18)
        axes[1,j].step(t,np.r_[0,samples[0,i]-ref['truth'][i]]*100,where='post',color='#6d4fa4')
        for model,color,ls in direct:
            p=prediction(folder/f'{model}_direct_-1.npz')[i]
            axes[0,j].step(t,np.r_[1,p],where='post',color=color,ls=ls,label=display_name(model,'direct'))
            axes[1,j].step(t,np.r_[0,p-ref['truth'][i]]*100,where='post',color=color,ls=ls)
        axes[0,j].set_title(f'Calcium rank {int(quantiles[j]*100)}%: x={x[i]:.2f}')
        axes[0,j].set_ylim(-.02,1.03);axes[1,j].axhline(0,color='grey',lw=.6)
        for ax in axes[:,j]:ax.axvline(24,color='grey',lw=.8,ls=':');ax.grid(alpha=.15)
        axes[1,j].set_xlabel('Absolute step')
    axes[0,0].set_ylabel('KM reference / prediction');axes[1,0].set_ylabel('Error (percentage points)')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False)
    fig.tight_layout(rect=[0,.10,1,1]);save(fig,'updated_individual_baseline')
    hs=pd.read_csv(hard_run/'evaluation/summary.csv');cs=pd.read_csv(cr_run/'evaluation/summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,5.2))
    for ax,df,title in zip(axes,[hs,cs[cs.target=='Survival']],['Single-covariate semi-Markov','Multi-covariate competing risks']):
        styles=[('Glazier-GRU','sobol','#7a4fab','o'),('Glazier-Transformer','sobol','#ae77cf','s'),
            ('SurvTransformer-Discrete','direct','#176b9c','^'),('SurvTransformer','direct','#489db5','v'),
            ('DynamicDeepHit','direct','#bb593d','D'),('Oracle sampler','iid','#333333','x')]
        for model,method,color,marker in styles:
            d=df[(df.model==model)&(df.method==method)].sort_values('landmark')
            ax.plot(np.arange(len(d)),d.MAE,marker=marker,color=color,label=display_name(model,method))
        ax.set_yscale('log');ax.set_xlabel('Landmark');ax.set_title(title);ax.grid(alpha=.2)
        landmarks=sorted(df.landmark.unique());ax.set_xticks(np.arange(len(landmarks)),labels=landmarks)
    axes[0].set_ylabel('Mean survival-curve absolute error (log scale)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False)
    fig.tight_layout(rect=[0,.19,1,1]);save(fig,'updated_survival_comparison')
    (destination/'FIGURE_NOTES.txt').write_text(
        'Baseline individuals: fixed calcium ranks 10%, 50%, 90%; line is rollout seed 41001.\n'
        'Shading is the minimum-to-maximum range of five rollout predictions, not a confidence interval.\n'
        'Overview: per-checkpoint mean MAE; landmarks have changing eligible populations and windows.\n'
        'Saved checkpoint identifiers are unchanged; only displayed Glazier names become ATLAS.\n')
    print('Exported two additional benchmark views:',destination)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--hard-run',required=True,type=Path)
    p.add_argument('--cr-run',required=True,type=Path);p.add_argument('--destination',required=True,type=Path)
    a=p.parse_args();export_views(a.hard_run,a.cr_run,a.destination)
