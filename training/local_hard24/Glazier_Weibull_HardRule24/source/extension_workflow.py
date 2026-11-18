"""Explicit, isolated training/evaluation for preserved semi-Markov and CR generators.
No training on import. Smoke results check execution only, not performance.
"""
from pathlib import Path
import json, runpy, sys, time, pickle, hashlib, random
import numpy as np
import pandas as pd
import torch
from lifelines import KaplanMeierFitter
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import sfsa_data_generation as base
import sfsa_semi_markov as sm
import sfsa_semi_markov_extras as extras
import sfsa_competing_risks as cr
import reference_recursion as reference

SPECS={
 'CPH':('cph_model','fit_cph','predict_survival','predict_cif_cph'),
 'DynamicDeepHIt-Lite':('dynamic_deephit_model','train_dynamic_deephit','predict_survival_dh','predict_cif_dh'),
 'SurvTransformer-Lite':('survtransformer_model','train_survtransformer','predict_survival_survtransformer','predict_cif_survtransformer'),
 'Weibull':('wtte_model','train_wtte','predict_survival_wtte','predict_cif_wtte'),
 'DRSA':('drsa_model','train_drsa','predict_survival_drsa','predict_cif_drsa'),
 'Glazier-GRU':('glazier_model','train_glazier','predict_survival_glazier','predict_cif_glazier'),
 'Glazier-RNN':('glazier_rnn_model','train_glazier_rnn','predict_survival_glazier_rnn','predict_cif_glazier_rnn'),
 'Glazier-Transformer':('glazier_transformer_model','train_glazier_transformer','predict_survival_glazier_transformer','predict_cif_glazier_transformer'),
}
HISTORY_SPECS={'DynamicDeepHit':'deephit', 'SurvTransformer':'survtrace',
               'SurvTransformer-Discrete':'survtrace'}
LEGACY_LABELS={'DynamicDeepHit':'DynamicDeepHIt-Lite', 'SurvTransformer':'SurvTransformer-Lite'}
ALL_SM_LABELS=list(SPECS)+list(HISTORY_SPECS)

def load_definitions(filename,kind):
    context=dict(vars(base))
    if kind=='sm':context.update(vars(sm));context.update(vars(extras))
    elif kind=='cr':context.update(vars(cr))
    else:raise ValueError('kind must be sm or cr')
    context={k:v for k,v in context.items() if not k.startswith('__')}
    if filename=='cph_model_cr.py':
        context['fit_cph']=load_definitions('cph_model_sm.py','sm')['fit_cph']
    # Dedicated globals prevent one pasted model cell from replacing another's loss.
    return runpy.run_path(str(ROOT/filename),init_globals=context,run_name='extension_definitions')

def make_cohort(kind,scenario_name,n,seed,out_dir):
    if n<100:raise ValueError('Use at least 100 people for an 80/10/10 split')
    scenario={'ReLU':base.RELU_SCENARIO,'Weibull':base.WEIBULL_SCENARIO}[scenario_name]
    generator=sm.generate_cohort_sm if kind=='sm' else cr.generate_cohort_cr
    cohort=generator(scenario,n=n,seed=seed,t_k=30)
    order=np.random.default_rng(seed+1).permutation(n)
    cohort.update(train_index=order[:int(.8*n)],val_index=order[int(.8*n):int(.9*n)],test_index=order[int(.9*n):])
    for absorbing in ([3,4] if kind=='sm' else [3,4,5]):
        for t in range(29):
            mask=cohort['states'][:,t]==absorbing
            assert np.all(cohort['states'][mask,t+1]==absorbing)
    assert len(set(cohort['train_index'])&set(cohort['test_index']))==0
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'cohort.npz',**cohort)
    return scenario,cohort

class Experiment:
    def __init__(self,kind,scenario='Weibull',n=1000,seed=21001,out_dir='extension_run'):
        self.kind=kind;self.out=Path(out_dir);self.seed=seed
        self.scenario,self.cohort=make_cohort(kind,scenario,n,seed,self.out)
        self.models={};self.config=dict(kind=kind,scenario=scenario,n=n,cohort_seed=seed,split_seed=seed+1,
            split='80/10/10 train/validation/test',target='population event-first KM/AJ reference of observed generator; latent biological identification unresolved',
            numpy=np.__version__,torch=torch.__version__,training={},evaluations=[])
        self.config['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}
        self.config['history_comparator_version']=2
        self._save_config()
    def _save_config(self):
        (self.out/'run_config.json').write_text(json.dumps(self.config,indent=2,default=str))
    def train(self,labels=None,epochs=2,patience=2,seed=42,device=None,history_options=None):
        if self.config.get('history_comparator_version',0)<2:
            raise ValueError('Archived runs are read-only. Use hard_rule_workflow.train with a fresh out and reuse_from.')
        labels=(ALL_SM_LABELS if self.kind=='sm' else list(SPECS)) if labels is None else labels
        history_options=history_options or {}
        c=self.cohort;tr=c['train_index'];va=c['val_index'];outcome='event' if self.kind=='sm' else 'cause'
        device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
        for label in labels:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            print(f'Training {self.kind}/{self.scenario.name}/{label}: up to {epochs} epochs',flush=True)
            start=time.perf_counter()
            if label in HISTORY_SPECS:
                if self.kind!='sm':raise ValueError('History comparators currently target single-event semi-Markov data')
                from history_training import train_history
                options=dict(history_options.get(label,{}))
                options['model_config']=dict(options.get('model_config',{}))
                if label.startswith('SurvTransformer'):
                    expected='discrete' if label.endswith('-Discrete') else 'pch'
                    if options['model_config'].get('likelihood',expected)!=expected:
                        raise ValueError('Likelihood must match the comparator label')
                    options['model_config']['likelihood']=expected
                model,meta=train_history(c['covariate'][tr],c['states'][tr],c['duration'][tr],c[outcome][tr],
                    c['covariate'][va],c['states'][va],c['duration'][va],c[outcome][va],
                    architecture=HISTORY_SPECS[label],max_epochs=epochs,patience=patience,
                    seed=seed,device=device,**options)
                ns={}
                torch.save({'state_dict':model.state_dict(),'meta':meta,'label':label,'kind':self.kind,
                            'class_name':type(model).__name__,'history_version':2},self.out/(label+'.pt'))
                pd.DataFrame(meta['training_log']).to_csv(self.out/(label+'_training.csv'),index=False)
                elapsed=time.perf_counter()-start
                self.models[label]=(model,meta,ns)
                self.config['training'][label]=dict(max_epochs=epochs,patience=patience,seed=seed,
                    device=device,seconds=elapsed,history_version=2,options=options,
                    best_epoch=meta['best_epoch'],monitor=meta['monitor'],
                    training_prefixes=meta['training_prefixes'],validation_prefixes=meta['validation_prefixes'])
                self._save_config();print(f'  Saved history checkpoint; {elapsed:.1f}s',flush=True)
                continue
            stem,train_name,_,_=SPECS[label];ns=load_definitions(stem+'_'+self.kind+'.py',self.kind)
            fn=ns[train_name+('_cr' if self.kind=='cr' else '')]
            if label=='CPH':
                model=fn(c['covariate'][tr],c['duration'][tr],c[outcome][tr],covariate_name=self.scenario.covariate_name)
                meta={}
                with (self.out/(label+'.pkl')).open('wb') as f:pickle.dump(model,f)
            else:
                if label.startswith('Glazier'):
                    args=[c['covariate'][tr],c['states'][tr],c['covariate'][va],c['states'][va]]
                else:args=[c['covariate'][tr],c['duration'][tr],c[outcome][tr],c['covariate'][va],c['duration'][va],c[outcome][va]]
                model,meta=fn(*args,max_epochs=epochs,patience=patience,seed=seed,device=device,verbose=True)
                torch.save({'state_dict':model.state_dict(),'meta':meta,'label':label,'kind':self.kind,'class_name':type(model).__name__},self.out/(label+'.pt'))
            elapsed=time.perf_counter()-start
            self.models[label]=(model,meta,ns)
            self.config['training'][label]=dict(max_epochs=epochs,patience=patience,seed=seed,device=device,seconds=elapsed)
            self._save_config();print(f'  Saved checkpoint; {elapsed:.1f}s',flush=True)
    @classmethod
    def restore(cls,out_dir,device=None):
        """Restore this package's saved models without retraining. Only load trusted runs."""
        self=cls.__new__(cls);self.out=Path(out_dir)
        self.config=json.loads((self.out/'run_config.json').read_text())
        self.kind=self.config['kind'];self.seed=self.config['cohort_seed']
        self.scenario={'ReLU':base.RELU_SCENARIO,'Weibull':base.WEIBULL_SCENARIO}[self.config['scenario']]
        with np.load(self.out/'cohort.npz') as data:self.cohort={k:data[k] for k in data.files}
        self.models={};device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
        old_format=self.config.get('history_comparator_version',0)<2
        renamed_training={}
        for saved_label,training_config in self.config['training'].items():
            label=LEGACY_LABELS.get(saved_label,saved_label) if old_format else saved_label
            renamed_training[label]=training_config
            if label in HISTORY_SPECS:
                from history_models import DynamicDeepHitHistory,SurvTRACEHistory
                checkpoint=torch.load(self.out/(saved_label+'.pt'),map_location=device,weights_only=True)
                if checkpoint.get('history_version')!=2 or checkpoint['meta'].get('history_version')!=2:
                    raise ValueError('A legacy checkpoint cannot be restored as a history comparator')
                classes={'DynamicDeepHitHistory':DynamicDeepHitHistory,'SurvTRACEHistory':SurvTRACEHistory}
                model=classes[checkpoint['class_name']](**checkpoint['meta']['model_config']).to(device)
                model.load_state_dict(checkpoint['state_dict']);model.eval()
                meta=checkpoint['meta'];meta['device']=device
                self.models[label]=(model,meta,{})
                continue
            ns=load_definitions(SPECS[label][0]+'_'+self.kind+'.py',self.kind)
            if label=='CPH':
                with (self.out/(saved_label+'.pkl')).open('rb') as f:model=pickle.load(f)
                meta={}
            else:
                checkpoint=torch.load(self.out/(saved_label+'.pt'),map_location=device,weights_only=True)
                model=ns[checkpoint['class_name']]().to(device)
                model.load_state_dict(checkpoint['state_dict']);model.eval()
                meta=checkpoint['meta'];meta['device']=device
            self.models[label]=(model,meta,ns)
        self.config['training']=renamed_training
        if old_format:self.config['legacy_label_mapping']=dict(LEGACY_LABELS)
        return self
    def predict(self,label,x,times,samples=150,seed=4242,prefix=None,L=None):
        model,meta,ns=self.models[label]
        if label in HISTORY_SPECS:
            from history_training import predict_history
            return predict_history(model,meta,x,times,prefix=prefix,landmark=0 if L is None else L)
        spec=SPECS[label]
        if prefix is not None and label.startswith('Glazier'):
            fn=ns[spec[2]+'_landmark_sm']
            return fn(model,meta,x,prefix,L,times,n_samples=samples,seed=seed,t_k=30)
        fn=ns[spec[2] if self.kind=='sm' else spec[3]]
        kwargs={'n_samples':samples,'seed':seed} if label.startswith('Glazier') else {}
        def call(grid):
            if label=='CPH':return fn(model,x,grid,covariate_name=self.scenario.covariate_name)
            return fn(model,meta,x,grid,**kwargs)
        if prefix is None:return call(times)
        full=call(np.arange(31));denom=full[:,L]
        if np.any(denom<=0):raise ValueError(f'{label}: S(L)=0; conditional ratio is undefined')
        return np.minimum(full[:,times]/denom[:,None],1.)
    def evaluate(self,n_eval=30,samples=150,gt_samples=1000,landmarks=(10,15,20),seed=4242,tag='pilot'):
        if not self.models:raise ValueError('Train models first')
        if n_eval<1 or samples<2 or gt_samples<2:raise ValueError('positive evaluation size and at least two trajectories required')
        if any(L<1 or L>=30 for L in landmarks):raise ValueError('landmarks must be between 1 and 29')
        c=self.cohort;idx=c['test_index'];idx=np.random.default_rng(self.seed+2).permutation(idx)[:n_eval]
        x=c['covariate'][idx];times=np.arange(31);rows=[];saved={'test_indices':idx,'times':times}
        start=time.perf_counter()
        if self.kind=='sm':
            truth=extras.ground_truth_km_sm_for_validation_set(self.scenario,x,base_seed=self.seed+100,n_matched=gt_samples)
            oracle=extras.ground_truth_km_sm_for_validation_set(self.scenario,x,base_seed=seed+90000,n_matched=samples)
        else:
            ref=cr.ground_truth_cif_for_validation_set(self.scenario,x,base_seed=self.seed+100,n_matched=gt_samples)
            truth=np.stack([ref['cif_disease'],ref['cif_other']],axis=1)
            ref=cr.ground_truth_cif_for_validation_set(self.scenario,x,base_seed=seed+90000,n_matched=samples)
            oracle=np.stack([ref['cif_disease'],ref['cif_other']],axis=1)
        print(f'References ready ({time.perf_counter()-start:.1f}s)',flush=True)
        saved['baseline_reference']=truth
        metrics=load_definitions('model_comparison_semimarkov.py' if self.kind=='sm' else 'model_comparison_competing_risks.py',self.kind)
        outcome=c['event'] if self.kind=='sm' else (c['cause']>0).astype(int)
        tr=c['train_index'];km=KaplanMeierFitter().fit(c['duration'][tr],event_observed=1-outcome[tr])
        G=lambda t:km.survival_function_at_times(t).to_numpy()
        def record(label,pred,target,grid,protocol,information,ids,seconds=0.,L=0):
            if not np.isfinite(pred).all():raise ValueError(f'{label}: nonfinite predictions')
            if pred.min() < -1e-6 or pred.max()>1+1e-6:raise ValueError(f'{label}: invalid probability range')
            variants={'survival':(pred,target)} if self.kind=='sm' else {'survival':(1-pred.sum(1),1-target.sum(1)),'disease':(pred[:,0],target[:,0]),'other':(pred[:,1],target[:,1])}
            for component,(p,t) in variants.items():
                if p.min() < -1e-6:raise ValueError(f'{label}: cause sum exceeds one')
                direction=-1 if component=='survival' else 1
                if np.any(direction*np.diff(p,axis=1)<-1e-6):raise ValueError(f'{label}: nonmonotone curve')
                keep=grid>L;error=abs(p[:,keep]-t[:,keep]);ibs=float('nan')
                if protocol=='baseline':
                    support=grid<30
                    if component=='survival' and self.kind=='sm':
                        _,ibs=metrics['integrated_brier_score'](p[:,support],grid[support],c['duration'][ids],c['event'][ids],G)
                    elif component!='survival':
                        _,ibs=metrics['integrated_brier_score_cr'](p[:,support],grid[support],c['duration'][ids],c['cause'][ids],1 if component=='disease' else 2,G)
                rows.append(dict(model=label,protocol=protocol,landmark=L,information=information,component=component,n=len(ids),MAE=error.mean(),max_horizon_MAE=error.mean(0).max(),IBS_0_29=ibs,samples=samples if label.startswith('Glazier') or label=='Oracle sampler' else 0,prediction_seconds=seconds))
        for label in list(self.models)+['Oracle sampler']:
            start=time.perf_counter();pred=oracle if label=='Oracle sampler' else self.predict(label,x,times,samples,seed)
            record(label,pred,truth,times,'baseline','baseline covariate',idx,time.perf_counter()-start)
            saved[label+'_baseline']=pred
        if self.kind=='cr':
            # Per-cause diagnostics from a known all-cause reference; these are oracle-assisted controls, not learned models.
            all_events=truth.sum(1);share=float(np.mean(c['cause'][tr][c['cause'][tr]>0]==1))
            for label,pi in [('Constant-share control',share),('Zero-other-cause control',1.)]:
                pred=np.stack([all_events*pi,all_events*(1-pi)],1)
                record(label,pred,truth,times,'baseline','oracle all-cause reference + training cause share',idx)
            self.config['test_cause_counts']={str(k):int(np.sum(c['cause'][idx]==k)) for k in [0,1,2]}
        else:
            for L in landmarks:
                live=c['states'][idx,L-1]<3;ids=idx[live]
                if not len(ids):continue
                xx=c['covariate'][ids];prefix=c['states'][ids,:L];grid=np.arange(L+1,31)
                # Exact observed-process reference for each actually observed augmented state.
                truth_L=[];state_control=[]
                sojourns=sm.current_state_sojourn_time(prefix)
                for cov,s,u in zip(xx,prefix[:,-1],sojourns):
                    states,init,k=reference.kernels(float(cov),self.scenario.name,'semi')
                    masses,_=reference.forward(init,k);indices=np.flatnonzero(masses[L]>0)
                    curves=reference.conditional_curves(k,L,indices)
                    z=states.index((int(s),int(u)));truth_L.append(reference.conditional_curves(k,L,[z])[0])
                    take=np.array([states[i][0]==s for i in indices]);w=masses[L][indices][take]
                    state_control.append(reference.weighted_median(curves[take],w))
                truth_L=np.asarray(truth_L);saved[f'landmark_{L}_ids']=ids;saved[f'landmark_{L}_reference']=truth_L
                for label in self.models:
                    start=time.perf_counter();pred=self.predict(label,xx,grid,samples,seed,prefix,L)
                    record(label,pred,truth_L,grid,'landmark','observed prefix' if label.startswith('Glazier') or label in HISTORY_SPECS else 'covariate + survival-only ratio',ids,time.perf_counter()-start,L)
                    saved[f'{label}_landmark_{L}']=pred
                sampled=sm.ground_truth_km_landmark_sm_for_validation_set(self.scenario,xx,prefix,L,base_seed=seed+90000,n_matched=samples)[:,L+1:]
                for label,pred,info in [('Oracle sampler',sampled,'true augmented state; finite sampling'),('Current-state oracle median',np.asarray(state_control),'covariate + current state'),('Augmented-state oracle',truth_L,'covariate + state + sojourn')]:
                    record(label,pred,truth_L,grid,'landmark',info,ids,L=L)
        df=pd.DataFrame(rows);df.to_csv(self.out/(tag+'_metrics.csv'),index=False)
        np.savez_compressed(self.out/(tag+'_curves.npz'),**saved)
        self.config['evaluations'].append(dict(tag=tag,n_eval=len(idx),samples=samples,gt_samples=gt_samples,landmarks=list(landmarks),sampling_seed=seed,reference_seed=self.seed+100))
        self._save_config();return df
