"""Keep training replication distinct from repeated rollout estimation."""
from pathlib import Path
import argparse,copy,hashlib,json
import numpy as np
import pandas as pd
from atlas_figures import display_name
ROLLOUT_SEEDS=[41001,52001,63001,74001,85001]
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def aggregate(first,second,destination):
    first=Path(first).resolve();second=Path(second).resolve();destination=Path(destination).resolve()
    if any(destination==p or p in destination.parents for p in (first,second)):
        raise ValueError('Use a separate aggregation output directory')
    frames=[]
    for root in (first,second):
        if not (root/'evaluation/COMPLETE.json').is_file(): raise ValueError('Incomplete evaluation: '+str(root))
        t=json.loads((root/'training_settings.json').read_text())
        ev=json.loads((root/'evaluation/evaluation_settings.json').read_text())
        frame=pd.read_csv(root/'evaluation/replicate_metrics.csv').rename(columns={'seed':'rollout_seed'})
        frame['training_seed']=t.get('training_seed',t.get('seed'))
        frame['target']=frame['target'] if 'target' in frame else 'Survival'
        if ev['seeds']!=ROLLOUT_SEEDS:raise ValueError('Rollout seed set changed')
        frames.append((frame,t,ev))
    if frames[0][1].get('training_seed',frames[0][1].get('seed'))==frames[1][1].get('training_seed',frames[1][1].get('seed')):
        raise ValueError('Training seeds must differ')
    if digest(first/'checkpoints/cohort.npz')!=digest(second/'checkpoints/cohort.npz'):
        raise ValueError('Different cohorts cannot be treated as initialization-only repeats')
    for key in ('M','n_eval','seeds','landmarks','batch_size'):
        if frames[0][2][key]!=frames[1][2][key]:raise ValueError('Evaluation setting differs: '+key)
    for key in ('n','epochs','patience','cohort_seed','scenario','rule'):
        if frames[0][1].get(key)!=frames[1][1].get(key):
            raise ValueError('Training/data setting differs: '+key)
    def training_options(settings):
        options=copy.deepcopy(settings.get('history_options',settings.get('options',{})))
        seed=settings.get('training_seed',settings.get('seed'))
        for name in ('DynamicDeepHit','SurvTransformer','SurvTransformer-Discrete'):
            if name in options:
                options[name].setdefault('validation_index_seed',seed+103)
        return options
    if training_options(frames[0][1])!=training_options(frames[1][1]):
        raise ValueError('Model/training options or validation-prefix subset differ')
    for L in frames[0][2]['landmarks']:
        with np.load(first/f'evaluation/L{L}/reference.npz') as a,np.load(second/f'evaluation/L{L}/reference.npz') as b:
            for key in ('patient_ids','times','truth'):np.testing.assert_array_equal(a[key],b[key])
    group=['target','landmark','model','method']
    def signatures(f):return set(map(tuple,f[group+['rollout_seed']].to_numpy()))
    if signatures(frames[0][0])!=signatures(frames[1][0]):raise ValueError('Unmatched/missing evaluation configurations')
    full=pd.concat([f[0] for f in frames],ignore_index=True)
    if full.duplicated(['training_seed',*group,'rollout_seed']).any():raise ValueError('Duplicate metric rows')
    for keys,g in full.groupby(['training_seed',*group]):
        expected=set(ROLLOUT_SEEDS) if keys[3].startswith('Glazier-') or keys[3]=='Oracle sampler' else {-1}
        if set(g.rollout_seed)!=expected:raise ValueError('Incomplete rollout repetitions: '+str(keys))
    destination.mkdir(parents=True,exist_ok=True)
    full['display_name']=[display_name(m,s) for m,s in zip(full.model,full.method)]
    # Oracle and CPH are not independent stochastic neural-training replicates.
    neural=full[~full.model.isin(['Oracle sampler','CPH'])].copy()
    fixed=full[full.model.isin(['Oracle sampler','CPH'])].copy()
    full.to_csv(destination/'all_metrics_with_seed_types.csv',index=False)
    fixed.to_csv(destination/'training_independent_baselines.csv',index=False)
    per_checkpoint=neural.groupby(['training_seed',*group]).agg(
        rollout_count=('MAE','size'),MAE_mean=('MAE','mean'),rollout_SD=('MAE','std')).reset_index()
    records=[]
    for key,g in per_checkpoint.groupby(group):
        if len(g)!=2:raise ValueError('Expected two independent training checkpoints')
        within=g.rollout_SD.dropna().to_numpy()
        records.append(dict(zip(group,key),training_checkpoints=len(g),MAE_mean=g.MAE_mean.mean(),
            between_checkpoint_mean_SD=g.MAE_mean.std(ddof=1),
            pooled_within_checkpoint_rollout_SD=np.sqrt(np.mean(within**2)) if len(within) else np.nan,
            checkpoint_means_min=g.MAE_mean.min(),checkpoint_means_max=g.MAE_mean.max()))
    overall=pd.DataFrame(records)
    for table in (per_checkpoint,overall):
        table['display_name']=[display_name(m,s) for m,s in zip(table.model,table.method)]
    per_checkpoint.to_csv(destination/'per_checkpoint_MAE.csv',index=False)
    overall.to_csv(destination/'two_training_seed_MAE.csv',index=False)
    (destination/'INTERPRETATION.txt').write_text(
        'Two checkpoints, not ten independent fits. Within-checkpoint SD uses five rollout seeds.\n'
        'Between-checkpoint SD describes two rollout-averaged errors and retains finite-rollout error.\n'
        'Matched rollout seeds can correlate errors across checkpoints. These are descriptive summaries, not confidence intervals.\n'
        'Direct predictions have one deterministic prediction per checkpoint; rollout SD is not estimated.\n'
        'CPH is deterministic here, and the oracle does not train. Do not count either as replicated neural training.\n')
    print('Saved nested training/rollout summaries:',destination)
    return per_checkpoint,overall

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--first',required=True,type=Path)
    p.add_argument('--second',required=True,type=Path);p.add_argument('--destination',required=True,type=Path)
    a=p.parse_args();aggregate(a.first,a.second,a.destination)
