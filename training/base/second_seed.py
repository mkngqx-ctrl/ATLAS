"""A second independent fit on an unchanged cohort. No training occurs on import.

Run hard24 and cr3 in separate Python processes: their historical modules have
the same import names but different implementations.
"""
from pathlib import Path
import argparse, copy, datetime, hashlib, json, os, shutil, sys
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
if not hasattr(datetime,'UTC'): datetime.UTC=datetime.timezone.utc
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROLLOUT_SEEDS=[41001,52001,63001,74001,85001]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2,default=str),encoding='utf-8')

def prepare(kind, previous, out, training_seed=43, device='cuda'):
    previous=Path(previous).resolve();out=Path(out).resolve()
    if out==previous or previous in out.parents or out in previous.parents:
        raise ValueError('Use a separate sibling output folder, outside the archived run')
    config=json.loads((previous/'training_settings.json').read_text())
    evaluation=json.loads((previous/'evaluation/evaluation_settings.json').read_text())
    original_seed=config.get('training_seed',config.get('seed'))
    if original_seed!=42 or training_seed==original_seed:
        raise ValueError('This package targets the audited seed-42 archives; choose a different training seed')
    if not 0<=training_seed<2**32-103: raise ValueError('Seed out of supported range')
    if evaluation['seeds']!=ROLLOUT_SEEDS or evaluation['M']!=10000 or evaluation['n_eval']!=300:
        raise ValueError('Evaluation settings differ from the audited full runs')
    expected_landmarks=[0,20,23,24] if kind=='hard24' else [0,10,20]
    expected_cohort_seed=24001 if kind=='hard24' else 34001
    if evaluation['landmarks']!=expected_landmarks or config['cohort_seed']!=expected_cohort_seed or config['n']!=50000:
        raise ValueError('Wrong experiment archive')
    source=json.loads((HERE/'source_manifest.json').read_text())[kind]
    bundle=HERE/source['bundle']
    if config['source_hashes']!=source['original_hashes']:
        raise ValueError('Archive source hashes do not match the audited implementation')
    for name,wanted in source['revised_hashes'].items():
        if digest(bundle/name)!=wanted: raise ValueError('Rerun bundle was modified: '+name)
    cohort=previous/'checkpoints/cohort.npz'
    cohort_hash=digest(cohort)
    if evaluation['checkpoint_hashes']['cohort.npz']!=cohort_hash:
        raise ValueError('Archived cohort checksum does not match evaluation')
    for name,wanted in evaluation['checkpoint_hashes'].items():
        if digest(previous/'checkpoints'/name)!=wanted:
            raise ValueError('Archived checkpoint checksum mismatch: '+name)
    metadata=torch.load(previous/'checkpoints/DynamicDeepHit.pt',map_location='cpu',weights_only=True)['meta']
    val_seed=int(metadata['validation_index_seed'])
    if val_seed!=145: raise ValueError('Unexpected original validation-prefix seed')
    options=copy.deepcopy(config.get('history_options',config.get('options')))
    for name in ('DynamicDeepHit','SurvTransformer','SurvTransformer-Discrete'):
        options[name]['validation_index_seed']=val_seed
    cached=[]
    for L in expected_landmarks:
        folder=previous/'evaluation'/f'L{L}'
        cached.append(folder/'reference.npz')
        cached.extend(folder/f'Oracle sampler_iid_{seed}.npz' for seed in ROLLOUT_SEEDS)
    plan=dict(version=1,experiment=kind,previous=str(previous),training_seed=training_seed,
        original_training_seed=original_seed,validation_index_seed=val_seed,
        rollout_seeds=ROLLOUT_SEEDS,cohort_sha256=cohort_hash,
        source_manifest_sha256=digest(HERE/'source_manifest.json'),device=device,
        epochs=config['epochs'],patience=config['patience'],options=options,
        reused_files={str(p.relative_to(previous)):digest(p) for p in cached})
    marker=out/'second_seed_plan.json'
    if marker.exists():
        if json.loads(marker.read_text())!=plan:
            raise ValueError('Rerun settings changed; use a fresh output directory')
        if digest(out/'checkpoints/cohort.npz')!=cohort_hash:
            raise ValueError('Rerun cohort changed')
    else:
        if out.exists() and any(out.iterdir()):
            raise ValueError('A fresh output directory must be empty')
        (out/'checkpoints').mkdir(parents=True,exist_ok=True)
        shutil.copy2(cohort,out/'checkpoints/cohort.npz')
        if kind=='hard24':
            run_config=json.loads((previous/'checkpoints/run_config.json').read_text())
            run_config['training']={};run_config['evaluations']=[]
            for key in ('reused_from','reused_cohort_sha256'): run_config.pop(key,None)
            run_config['cohort_reused_from']=str(cohort)
            run_config['source_sha256']={p.name:digest(p) for p in (bundle/'source').glob('*.py')}
            run_config['torch']=str(torch.__version__);run_config['numpy']=np.__version__
            write_json(out/'checkpoints/run_config.json',run_config)
        else:
            shutil.copy2(previous/'checkpoints/cohort_manifest.json',out/'checkpoints/cohort_manifest.json')
        # No learned model checkpoint or learned prediction is copied.
        for p in cached:
            target=out/p.relative_to(previous)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,target)
        write_json(marker,plan)
    for relative,wanted in plan['reused_files'].items():
        if digest(out/relative)!=wanted: raise ValueError('Cached reference/oracle changed: '+relative)
    return bundle,config,evaluation,options,plan

def run(kind, previous, out, training_seed=43, device='cuda', prepare_only=False):
    bundle,config,ev,options,plan=prepare(kind,previous,out,training_seed,device)
    out=Path(out).resolve()
    print('Original training seed: 42; new training seed:',training_seed,flush=True)
    print('Unchanged validation-prefix seed: 145; rollout seeds:',ROLLOUT_SEEDS,flush=True)
    print('Output:',out,flush=True)
    if prepare_only: return plan
    if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('Select a GPU runtime before full training')
    for name in ('history_models','history_training','model_training','extension_workflow'):
        if name in sys.modules: raise RuntimeError('Run each experiment in a fresh Python process')
    sys.path.insert(0,str(bundle));sys.path.insert(0,str(bundle/'source'))
    for parent in HERE.parents:
        support=parent/'work/gpu_support'
        if support.is_dir(): sys.path.insert(0,str(support));break
    torch.set_num_threads(4)
    if kind=='hard24':
        from hard_rule_workflow import train,evaluate
        from extension_workflow import ALL_SM_LABELS
        experiment=train(out,n=config['n'],epochs=config['epochs'],patience=config['patience'],
            device=device,history_options=options,labels=list(ALL_SM_LABELS),reuse_from=None,
            training_seed=training_seed)
        fitted={k:v['seed'] for k,v in experiment.config['training'].items() if k!='CPH'}
    else:
        from cr_workflow import train,evaluate
        from sfsa_cr import Scenario
        experiment=train(out,n=config['n'],epochs=config['epochs'],patience=config['patience'],
            seed=training_seed,cohort_seed=config['cohort_seed'],device=device,
            scenario=Scenario(**config['scenario']),options=options)
        fitted={k:meta['seed'] for k,(_,meta) in experiment['models'].items() if k!='CPH'}
    if set(fitted.values())!={training_seed}: raise RuntimeError('A model retained a different training seed')
    changes={}
    for file in (out/'checkpoints').glob('*.pt'):
        first=torch.load(Path(previous)/'checkpoints'/file.name,map_location='cpu',weights_only=True)['state_dict']
        second=torch.load(file,map_location='cpu',weights_only=True)['state_dict']
        same=set(first)==set(second) and all(torch.equal(first[k],second[k]) for k in first)
        if same: raise RuntimeError('Learned parameters exactly match seed 42: '+file.name)
        changes[file.name]=dict(new_sha256=digest(file),different_learned_parameters=True)
    write_json(out/'checkpoint_independence_check.json',changes)
    evaluate(experiment,out/'evaluation',M=ev['M'],n_eval=ev['n_eval'],seeds=ev['seeds'],
             landmarks=ev['landmarks'],batch_size=ev['batch_size'])
    write_json(out/'second_seed_environment.json',dict(python=sys.version,torch=str(torch.__version__),
        numpy=np.__version__,cuda=torch.version.cuda,device=device,
        gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,pythonhashseed=os.environ.get('PYTHONHASHSEED')))
    from atlas_figures import export_figures
    export_figures(kind,out/'evaluation',out/'ATLAS_figures')
    print('Training, evaluation, and ATLAS figure export completed.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment',choices=['hard24','cr3'],required=True)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--training-seed',type=int,default=43)
    parser.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    parser.add_argument('--prepare-only',action='store_true')
    a=parser.parse_args()
    run(a.experiment,a.previous,a.out,a.training_seed,a.device,a.prepare_only)
