from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('experiment');p.add_argument('device');args=p.parse_args()
exp=args.experiment;previous=ROOT/'outputs/runs'/exp/'42';out=ROOT/'outputs/refits'/exp/'42'
config=json.loads((previous/'training_settings.json').read_text());ev=json.loads((previous/'evaluation/evaluation_settings.json').read_text())
src=ROOT/'training/seed42_hard' if exp=='Hard24' else previous/'source_snapshot'
sys.path[:0]=[str(src),str(src/'source')]
if exp=='Hard24':
 from hard_rule_workflow import train,evaluate
 model=train(out,n=config['n'],epochs=config['epochs'],patience=config['patience'],device=args.device,history_options=config.get('history_options'),reuse_from=None)
else:
 from cr_workflow import train,evaluate
 from sfsa_cr import Scenario
 model=train(out,n=config['n'],epochs=config['epochs'],patience=config['patience'],seed=42,cohort_seed=config['cohort_seed'],device=args.device,scenario=Scenario(**config['scenario']),options=config.get('options'))
evaluate(model,out/'evaluation',M=ev['M'],n_eval=ev['n_eval'],seeds=ev['seeds'],landmarks=ev['landmarks'],batch_size=ev['batch_size'])
