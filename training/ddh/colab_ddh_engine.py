"""ATLAS DDH follow-up: validation-only search, resumable fitting, gated evaluation."""
from pathlib import Path
import copy, csv, hashlib, io, json, math, os, random, time, zipfile
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import ddh_base as base

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def atomic_json(p, obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));os.replace(tmp,p)

def save_torch(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');torch.save(obj,tmp);os.replace(tmp,p)

def inputs(x, states, landmark, horizon=30, n_live=3):
    N,C=x.shape;dev=x.device;L=landmark.long()
    observed=torch.arange(1,horizon+1,device=dev)[None,:]<=L[:,None]
    safe=torch.where(observed,states,torch.zeros_like(states)).long()
    if bool(((safe<0)|(safe>=n_live)).any()):raise ValueError('Invalid observed state')
    valid=torch.arange(horizon+1,device=dev)[None,:]<=L[:,None]
    values=torch.zeros(N,horizon+1,C+n_live+1,device=dev)
    values[:,:,:C]=x[:,None,:]
    values[:,1:,C:C+n_live]=F.one_hot(safe,n_live).float()*observed[:,:,None]
    values[:,:,-1]=torch.arange(horizon+1,device=dev)/horizon
    missing=torch.zeros_like(values);missing[:,0,C:C+n_live]=1
    values*=valid[:,:,None];missing=torch.where(valid[:,:,None],missing,torch.ones_like(missing))
    gap=(torch.arange(horizon+1,device=dev)[None,:]<L[:,None]).float()/horizon
    return dict(x=x,states=safe,observed=observed,landmark=L,values=values,missing=missing,gap=gap,valid=valid)

class DDH(base.DynamicDeepHitHistory):
    def __init__(self,config):
        cfg=copy.deepcopy(config);self.readout=cfg.pop('readout','top');self.output_wiring=cfg.pop('output_wiring','independent')
        super().__init__(**cfg);self.full_config=copy.deepcopy(config)
        D=cfg['n_covariates']+cfg['n_live']+1;hidden=cfg['hidden'];layers=cfg['layers'];drop=cfg['dropout']
        self.layer_dropout=nn.Dropout(drop)
        if self.readout=='all':
            del self.gru
            self.gru_stack=nn.ModuleList([nn.GRU(2*D+1 if j==0 else hidden,hidden,1,batch_first=True) for j in range(layers)])
        context_width=hidden*(layers if self.readout=='all' else 1)
        if self.readout=='all':
            self.attention=nn.Sequential(nn.Linear(context_width+2*D,cfg['attention_hidden']),nn.Tanh(),nn.Linear(cfg['attention_hidden'],1))
        # Rebuild heads only when the shared input or output wiring changes.
        if self.readout=='all' or self.output_wiring=='shared':
            self.heads=nn.ModuleList()
            for _ in range(self.causes):
                blocks=[];width=context_width+2*D
                for _ in range(cfg['head_layers']):
                    blocks.extend([nn.Linear(width,cfg['head_hidden']),nn.ReLU(),nn.Dropout(drop)]);width=cfg['head_hidden']
                blocks.append(nn.Linear(width,self.horizon if self.output_wiring=='independent' else cfg['head_hidden']))
                self.heads.append(nn.Sequential(*blocks))
            if self.output_wiring=='shared':
                del self.tail
                self.joint_output=nn.Linear(self.causes*cfg['head_hidden'],self.causes*self.horizon+1)
            else:self.tail=nn.Linear(context_width+2*D,1)

    def forward(self,batch):
        if self.readout=='top' and self.output_wiring=='independent':return super().forward(batch)
        values,missing,L=batch['values'],batch['missing'],batch['landmark'];row=torch.arange(len(L),device=L.device)
        latest=torch.cat([values[row,L],missing[row,L]],-1)
        past=torch.cat([values[:,:-1],missing[:,:-1],batch['gap'][:,:-1,None]],-1)
        if self.readout=='all':
            state=past;all_hidden=[]
            for j,gru in enumerate(self.gru_stack):
                packed=nn.utils.rnn.pack_padded_sequence(state,L.clamp_min(1).cpu(),batch_first=True,enforce_sorted=False)
                packed,_=gru(packed);state,_=nn.utils.rnn.pad_packed_sequence(packed,batch_first=True,total_length=self.horizon)
                all_hidden.append(state)
                if j<len(self.gru_stack)-1:state=self.layer_dropout(state)
            hidden=torch.cat(all_hidden,-1);top=all_hidden[-1]
        else:
            packed=nn.utils.rnn.pack_padded_sequence(past,L.clamp_min(1).cpu(),batch_first=True,enforce_sorted=False)
            packed,_=self.gru(packed);hidden,_=nn.utils.rnn.pad_packed_sequence(packed,batch_first=True,total_length=self.horizon);top=hidden
        mask=torch.arange(self.horizon,device=L.device)[None,:]<L[:,None]
        scores=self.attention(torch.cat([hidden,latest[:,None,:].expand(-1,self.horizon,-1)],-1)).squeeze(-1).masked_fill(~mask,-1e9)
        weights=scores.softmax(-1)*mask;context=(weights[:,:,None]*hidden).sum(1);shared=torch.cat([context,latest],-1)
        heads=[h(shared) for h in self.heads]
        logits=self.joint_output(torch.cat(heads,-1)) if self.output_wiring=='shared' else torch.cat(heads+[self.tail(shared)],-1)
        return dict(log_pmf=logits.log_softmax(-1),next_logits=self.next_state(top),attention=weights)

def objective(model,out,batch,T,J,patient,config):
    loss,parts=base.deephit_objective(model,out,batch,T,J,patient=patient,rank_weight=config['rank_weight'],rank_sigma=config['rank_sigma'],auxiliary_weight=config['auxiliary_weight'],l1_weight=0.,rank_reduction=config['rank_reduction'])
    scope=config['l1_scope']
    penalty=sum(p.abs().sum() for name,p in model.named_parameters() if 'weight' in name and (scope=='all' or name.startswith(('heads.','tail.','joint_output.'))))
    parts['l1']=penalty
    return loss+config['l1_weight']*penalty,parts

def original_config(exp):
    return dict(model=dict(horizon=30,hidden=64,layers=4 if exp=='Hard24' else 2,attention_hidden=64,head_hidden=64,head_layers=4,dropout=.1,causes=1 if exp=='Hard24' else 2,n_live=3 if exp=='Hard24' else 7,n_covariates=1 if exp=='Hard24' else 3,readout='top',output_wiring='independent'),loss=dict(rank_weight=.1,rank_sigma=.1,auxiliary_weight=.1,l1_weight=1e-5,rank_reduction='per_patient',l1_scope='all'),lr=.001,batch_size=256)

def make_candidates(exp,n):
    candidates=[original_config(exp)]
    c=copy.deepcopy(candidates[0]);c['loss']['l1_scope']='heads';candidates.append(c)
    c=copy.deepcopy(candidates[1]);c['model'].update(readout='all',output_wiring='shared');candidates.append(c)
    c=copy.deepcopy(candidates[1]);c['loss'].update(rank_weight=.01,auxiliary_weight=.01);candidates.append(c)
    rng=np.random.default_rng(20261101+(0 if exp=='Hard24' else 1))
    while len(candidates)<n:
        c=copy.deepcopy(candidates[1]);c['model'].update(hidden=int(rng.choice([32,64,100])),layers=int(rng.choice([1,2,4])),head_hidden=int(rng.choice([32,64,100])),head_layers=int(rng.choice([1,2,4])),dropout=float(rng.choice([0,.1,.3])),readout=str(rng.choice(['top','all'])),output_wiring=str(rng.choice(['independent','shared'])))
        c['model']['attention_hidden']=c['model']['hidden'];c['lr']=float(np.exp(rng.uniform(np.log(1e-4),np.log(3e-3))))
        reduction=str(rng.choice(['per_patient','pairs']))
        c['loss'].update(rank_weight=float(rng.choice([0,.001,.01,.1] if reduction=='per_patient' else [0,.1,1.,10.])),rank_sigma=float(rng.choice([.05,.1,.2])),auxiliary_weight=float(rng.choice([0,.01,.1,1.])),l1_weight=float(rng.choice([0,1e-6,1e-5,1e-4])),rank_reduction=reduction)
        if digest(c) not in [digest(z) for z in candidates]:candidates.append(c)
    return candidates

def concordance(curves,T,J,L,patient):
    residual=T-L;wins=0.;pairs=0
    for start in range(0,len(T),128):
        idx=np.arange(start,min(start+128,len(T)))
        mask=(J[idx,None]>0)&(residual[idx,None]<residual[None,:])&(patient[idx,None]!=patient[None,:])
        ii,jj=np.nonzero(mask);ii=idx[ii]
        first=curves[ii,J[ii],T[ii]];second=curves[jj,J[ii],L[jj]+residual[ii]]
        wins+=np.sum(first>second)+.5*np.sum(first==second);pairs+=len(ii)
    return float(wins/pairs) if pairs else .5

class Cohort:
    def __init__(self,root,exp,device='cuda'):
        self.exp=exp;self.device=device;self.horizon=30;self.nlive=3 if exp=='Hard24' else 7
        with np.load(Path(root)/'data'/exp/'train.npz') as z:train={k:z[k] for k in z.files}
        with np.load(Path(root)/'data'/exp/'validation.npz') as z:val={k:z[k] for k in z.files}
        assert not set(train['patient_ids'])&set(val['patient_ids'])
        self.mean=train['x'].mean(0);self.sd=np.maximum(train['x'].std(0),1e-8)
        self.train=self.tensors(train);self.val=self.tensors(val)
        self.patient=np.repeat(np.arange(len(train['x'])),train['duration'])
        self.landmark=np.concatenate([np.arange(c) for c in train['duration']])
        self.vpatient=np.repeat(np.arange(len(val['x'])),val['duration'])
        self.vlandmark=np.concatenate([np.arange(c) for c in val['duration']])
        self.val_duration=val['duration'];self.train_ids=train['patient_ids'];self.val_ids=val['patient_ids']
    def tensors(self,z):
        return dict(x=torch.tensor((z['x']-self.mean)/self.sd,dtype=torch.float32,device=self.device),states=torch.tensor(z['states'],dtype=torch.long,device=self.device),duration=torch.tensor(z['duration'],dtype=torch.long,device=self.device),cause=torch.tensor(z['cause'],dtype=torch.long,device=self.device))
    def batch(self,which,patient,L):
        z=self.train if which=='train' else self.val
        patient=torch.as_tensor(patient,dtype=torch.long,device=self.device);L=torch.as_tensor(L,dtype=torch.long,device=self.device)
        batch=inputs(z['x'][patient],z['states'][patient],L,self.horizon,self.nlive)
        return batch,z['duration'][patient],z['cause'][patient],patient

def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def validate(model,data,cfg,index_seed,balanced_limit=512):
    model.eval();rng=np.random.default_rng(index_seed)
    indices=np.sort(rng.choice(len(data.vpatient),min(2048,len(data.vpatient)),replace=False))
    total=0.;nll=0.;N=0;collected=[]
    with torch.no_grad():
        for idx in np.array_split(indices,math.ceil(len(indices)/256)):
            batch,T,J,p=data.batch('validation',data.vpatient[idx],data.vlandmark[idx]);out=model(batch)
            loss,parts=objective(model,out,batch,T,J,p,cfg['loss']);N+=len(idx);total+=float(loss)*len(idx);nll+=float(parts['nll'])*len(idx)
            curves=base.pmf_curves(out['log_pmf'],batch['landmark'],30,model.causes)
            collected.append((curves.cpu().numpy(),T.cpu().numpy(),J.cpu().numpy(),batch['landmark'].cpu().numpy(),p.cpu().numpy()))
        scores=[]
        for L in ([0,20,23,24] if data.exp=='Hard24' else [0,10,20]):
            eligible=np.flatnonzero(data.val_duration>L)
            selected=np.sort(np.random.default_rng(20261101+L).choice(eligible,min(balanced_limit,len(eligible)),replace=False))
            losses=[]
            for idx in np.array_split(selected,max(1,math.ceil(len(selected)/256))):
                batch,T,J,p=data.batch('validation',idx,np.full(len(idx),L));out=model(batch)
                _,parts=objective(model,out,batch,T,J,p,cfg['loss']);losses.append((float(parts['nll']),len(idx)))
            scores.append(sum(x*n for x,n in losses)/sum(n for _,n in losses))
    merged=[np.concatenate([z[k] for z in collected]) for k in range(5)]
    return dict(total_loss=total/N,prefix_nll=nll/N,concordance=concordance(*merged),balanced_nll=float(np.mean(scores)),landmark_nll=scores)

def fit(data,cfg,seed,directory,joint_epochs,burnin=5,selection='balanced_nll',patience=None,epoch_batch_cap=None):
    """Resume exact epoch-boundary RNG/optimizer state; never reads evaluation payload."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    key=digest(dict(config=cfg,seed=seed,burnin=burnin,selection=selection,epoch_batch_cap=epoch_batch_cap))
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if data.device!='cpu':torch.cuda.manual_seed_all(seed)
    model=DDH(cfg['model']).to(data.device);optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'])
    generator=torch.Generator().manual_seed(seed);start=0;history=[];best={};stale=0
    resume=directory/'resume.pt'
    if resume.exists():
        saved=torch.load(resume,map_location=data.device,weights_only=False)
        if saved['key']!=key:raise RuntimeError('Configuration differs from saved run; use a new study folder')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);start=saved['epoch'];history=saved['history'];best=saved['best'];stale=saved['stale']
        torch.set_rng_state(saved['torch_rng'].cpu());generator.set_state(saved['shuffle_rng'].cpu())
        if data.device!='cpu':torch.cuda.set_rng_state_all([z.cpu() for z in saved['cuda_rng']])
        if saved.get('stopped'):return json.loads((directory/'summary.json').read_text())
    total_epochs=burnin+joint_epochs
    for epoch in range(start,total_epochs):
        begin=time.perf_counter();model.train();order=torch.randperm(len(data.patient),generator=generator).numpy();total=0.;count=0
        for j in range(0,len(order),cfg['batch_size']):
            if epoch_batch_cap is not None and j//cfg['batch_size']>=epoch_batch_cap:break
            idx=order[j:j+cfg['batch_size']];batch,T,J,p=data.batch('train',data.patient[idx],data.landmark[idx]);out=model(batch)
            loss,parts=objective(model,out,batch,T,J,p,cfg['loss'])
            if epoch<burnin:loss=parts['next_measurement']
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step();total+=float(loss.detach())*len(idx);count+=len(idx)
        val=validate(model,data,cfg,seed+103 if data.exp=='Hard24' else 145)
        if not all(np.isfinite(val[k]) for k in ['total_loss','prefix_nll','concordance','balanced_nll']):raise FloatingPointError('Nonfinite validation metric')
        row=dict(epoch=epoch+1,burnin=epoch<burnin,training_loss=total/count,seconds=time.perf_counter()-begin,**val);history.append(row)
        improved=False
        if epoch>=burnin:
            state=None
            for metric in ['total_loss','prefix_nll','concordance','balanced_nll']:
                score=val[metric] if metric=='concordance' else -val[metric]
                if metric not in best or score>best[metric]['score']+1e-6:
                    best[metric]=dict(score=score,epoch=epoch+1,value=val[metric]);state=cpu_state(model) if state is None else state
                    save_torch(directory/f'best_{metric}.pt',dict(state_dict=state,config=cfg,mean=data.mean.tolist(),sd=data.sd.tolist(),seed=seed,epoch=epoch+1,metric=metric,metric_value=val[metric],run_key=key))
                    if metric==selection:improved=True
            stale=0 if improved else stale+1
        stopped=epoch>=burnin and patience is not None and stale>=patience
        summary=dict(key=key,config=cfg,seed=seed,epochs_completed=epoch+1,requested_joint_epochs=joint_epochs,burnin=burnin,selection=selection,best=best,stopped=stopped,training_prefixes=len(data.patient),epoch_batch_cap=epoch_batch_cap)
        save_torch(resume,dict(key=key,model=cpu_state(model),optimizer=optimizer.state_dict(),epoch=epoch+1,history=history,best=best,stale=stale,stopped=stopped,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if data.device!='cpu' else [],shuffle_rng=generator.get_state()))
        atomic_json(directory/'history.json',history);atomic_json(directory/'summary.json',summary)
        print(f"{data.exp} {directory.name} epoch {epoch+1}: NLL={val['balanced_nll']:.5f}; C={val['concordance']:.4f}; {row['seconds']:.1f}s",flush=True)
        if stopped:break
    del model,optimizer
    if data.device!='cpu':torch.cuda.empty_cache()
    return json.loads((directory/'summary.json').read_text())

class Study:
    def __init__(self,bundle,output,mode='FAST',experiments=('Hard24','CR3'),device='cuda',smoke=False):
        self.bundle=Path(bundle);self.out=Path(output);self.out.mkdir(parents=True,exist_ok=True);self.device=device;self.smoke=smoke
        manifest=json.loads((self.bundle/'manifest.json').read_text())
        for name,sha in manifest['files'].items():
            if hashlib.sha256((self.bundle/name).read_bytes()).hexdigest()!=sha:raise RuntimeError('Input hash mismatch: '+name)
        fast=mode=='FAST'
        protocol=dict(version='ATLAS_DDH_COLAB_v1',mode=mode,experiments=list(experiments),candidates=12 if fast else 24,rungs=[5,15,30] if fast else [10,30,60],keep_fraction=.5,confirmation_top=2 if fast else 3,confirmation_seeds=[43,45],search_seed=42,burnin=5,final_max_joint_epochs=200,final_patience=15,batch_size=256,selection='balanced_validation_conditional_NLL',validation_patients_per_landmark=512,final_seeds={'Hard24':[42,43,44,45,50,51,52,62,63],'CR3':[42,43,45,46,47]},precision='float32',test_during_search=False,synthetic_reference_during_search=False,epoch_batch_cap=None,bundle_digest=digest(manifest),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),smoke=smoke)
        if smoke:protocol.update(candidates=4,rungs=[1,2],confirmation_top=1,confirmation_seeds=[43],burnin=0,final_max_joint_epochs=2,final_patience=2,epoch_batch_cap=1,final_seeds={e:[42,43] for e in experiments})
        self.protocol=protocol;self.protocol_hash=digest(protocol)
        p=self.out/'frozen_protocol.json'
        if p.exists() and digest(json.loads(p.read_text()))!=self.protocol_hash:raise RuntimeError('Frozen protocol differs. Start a separate study folder before changing settings.')
        atomic_json(p,protocol)
        env=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if device!='cpu' else 'CPU',protocol_hash=self.protocol_hash)
        envfile=self.out/'environments.json';envs=json.loads(envfile.read_text()) if envfile.exists() else [];envs.append(env);atomic_json(envfile,envs)
        self.data={}
    def cohort(self,exp):
        if exp not in self.data:self.data[exp]=Cohort(self.bundle,exp,self.device)
        return self.data[exp]
    def search(self):
        if (self.out/'evaluation_started.json').exists():raise RuntimeError('Test evaluation has started; start a separate documented study for new model development.')
        p=self.protocol
        for exp in p['experiments']:
            data=self.cohort(exp);candidates=make_candidates(exp,p['candidates']);atomic_json(self.out/exp/'candidates.json',candidates)
            active=list(range(len(candidates)));all_rows=[]
            for rung in p['rungs']:
                stage_file=self.out/exp/f'rung_{rung}_frozen.json'
                if stage_file.exists():
                    stage=json.loads(stage_file.read_text())
                    assert stage['protocol_hash']==self.protocol_hash
                    active=stage['survivors'];all_rows.extend(stage['rows'])
                    continue
                scores=[]
                stage_rows=[]
                for k in active:
                    d=self.out/exp/'search'/f'candidate_{k:02d}'
                    try:
                        z=fit(data,candidates[k],42,d,rung,p['burnin'],epoch_batch_cap=p['epoch_batch_cap']);score=-z['best']['balanced_nll']['score']
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache();atomic_json(d/'failure.json',dict(reason='CUDA out of memory'));score=1e30
                    scores.append((score,k));record=dict(candidate=k,rung=rung,best_validation_NLL=score);all_rows.append(record);stage_rows.append(record)
                    atomic_json(self.out/exp/'leaderboard.json',all_rows)
                active=[k for score,k in sorted(scores) if score<1e29][:max(p['confirmation_top'],math.ceil(len(active)*p['keep_fraction']))]
                atomic_json(stage_file,dict(protocol_hash=self.protocol_hash,survivors=active,rows=stage_rows))
            shortlist=active[:p['confirmation_top']];ranked=[]
            for k in shortlist:
                initial=json.loads((self.out/exp/'search'/f'candidate_{k:02d}'/'summary.json').read_text());scores=[initial['best']['balanced_nll']['value']]
                for seed in p['confirmation_seeds']:
                    z=fit(data,candidates[k],seed,self.out/exp/'confirmation'/f'candidate_{k:02d}_seed_{seed}',p['rungs'][-1],p['burnin'],epoch_batch_cap=p['epoch_batch_cap']);scores.append(z['best']['balanced_nll']['value'])
                ranked.append(dict(candidate=k,mean_validation_NLL=float(np.mean(scores)),seed_scores=scores))
            best=min(ranked,key=lambda z:(z['mean_validation_NLL'],z['candidate']))
            lock=dict(protocol_hash=self.protocol_hash,experiment=exp,candidate=best['candidate'],config=candidates[best['candidate']],selection='mean balanced validation NLL across fixed search/confirmation seeds',confirmation=ranked)
            atomic_json(self.out/exp/'selection_lock.json',lock)
        return self.status()
    def final_fits(self):
        if (self.out/'evaluation_started.json').exists():raise RuntimeError('Test evaluation has started; existing final fits are frozen.')
        for exp in self.protocol['experiments']:
            lock=json.loads((self.out/exp/'selection_lock.json').read_text());assert lock['protocol_hash']==self.protocol_hash
            for seed in self.protocol['final_seeds'][exp]:
                fit(self.cohort(exp),lock['config'],seed,self.out/exp/'final'/f'seed_{seed}',self.protocol['final_max_joint_epochs'],self.protocol['burnin'],patience=self.protocol['final_patience'],epoch_batch_cap=self.protocol['epoch_batch_cap'])
        return self.status()
    def sensitivity(self):
        inventory=json.loads((self.bundle/'sensitivity_schedule.json').read_text())
        for exp in self.protocol['experiments']:
            for z in inventory:
                if z['experiment']!=exp:continue
                if self.smoke and z['seed']!=42:continue
                fit(self.cohort(exp),original_config(exp),z['seed'],self.out/exp/'sensitivity'/f"seed_{z['seed']}",max(1,z['epochs_logged']-5) if not self.smoke else 1,5 if not self.smoke else 0,selection='concordance',epoch_batch_cap=self.protocol['epoch_batch_cap'])
        return self.status()
    def status(self):
        rows=[]
        for exp in self.protocol['experiments']:
            lock=self.out/exp/'selection_lock.json'
            rows.append(dict(experiment=exp,configuration_locked=lock.exists(),final_fits=sum((self.out/exp/'final'/f'seed_{s}'/'summary.json').exists() for s in self.protocol['final_seeds'][exp]),required_final_fits=len(self.protocol['final_seeds'][exp])))
        return rows

    def evaluate_final(self,resamples=5000):
        """Open sealed test payload only after all preset final fits finish."""
        for exp in self.protocol['experiments']:
            lock=json.loads((self.out/exp/'selection_lock.json').read_text());assert lock['protocol_hash']==self.protocol_hash
            for seed in self.protocol['final_seeds'][exp]:
                d=self.out/exp/'final'/f'seed_{seed}';summary=json.loads((d/'summary.json').read_text())
                assert summary['config']==lock['config'] and (summary['stopped'] or summary['epochs_completed']>=self.protocol['burnin']+self.protocol['final_max_joint_epochs']), 'Final fits incomplete'
                assert (d/'best_balanced_nll.pt').exists()
        if self.smoke and not (self.bundle/'SMOKE_ONLY').exists():raise RuntimeError('Smoke evaluation requires synthetic test fixture')
        atomic_json(self.out/'evaluation_started.json',dict(protocol_hash=self.protocol_hash,reason='All predetermined final fits complete; configuration locked before test access'))
        destination=self.out/'test_payload'
        if not destination.exists():
            destination.mkdir()
            with zipfile.ZipFile(self.bundle/'evaluation_payload.zip') as z:
                for name in z.namelist():
                    path=(destination/name).resolve()
                    if not path.is_relative_to(destination.resolve()):raise RuntimeError('Unsafe archive member')
                z.extractall(destination)
        metrics=[];comparisons=[]
        for exp in self.protocol['experiments']:
            with np.load(destination/exp/'patients.npz') as z:patients={k:z[k] for k in z.files}
            assert not set(patients['patient_ids'])&set(self.cohort(exp).train_ids)
            assert not set(patients['patient_ids'])&set(self.cohort(exp).val_ids)
            patient_map={int(v):j for j,v in enumerate(patients['patient_ids'])}
            with np.load(destination/exp/'comparators.npz') as z:comparators={k:z[k] for k in z.files}
            targets=comparators['target_names'].tolist();means={};saved_errors={}
            for L in ([0,20,23,24] if exp=='Hard24' else [0,10,20]):
                with np.load(destination/exp/f'reference_L{L}.npz') as z:ref={k:z[k] for k in z.files}
                ids=ref['patient_ids'];times=ref['times'];truth=ref['truth'];truth=truth[:,None,:] if truth.ndim==2 else truth
                np.testing.assert_array_equal(ids,comparators[f'patient_ids_L{L}'])
                index=np.array([patient_map[int(v)] for v in ids]);fit_errors=[]
                for seed in self.protocol['final_seeds'][exp]:
                    ck=torch.load(self.out/exp/'final'/f'seed_{seed}'/'best_balanced_nll.pt',map_location=self.device,weights_only=False)
                    model=DDH(ck['config']['model']).to(self.device);model.load_state_dict(ck['state_dict']);model.eval();pred=[]
                    with torch.no_grad():
                        for idx in np.array_split(index,max(1,math.ceil(len(index)/256))):
                            x=torch.tensor((patients['x'][idx]-np.array(ck['mean']))/np.array(ck['sd']),dtype=torch.float32,device=self.device)
                            states=torch.tensor(patients['states'][idx],dtype=torch.long,device=self.device);landmark=torch.full((len(idx),),L,dtype=torch.long,device=self.device)
                            batch=inputs(x,states,landmark,30,3 if exp=='Hard24' else 7);out=model(batch)
                            pred.append(base.pmf_curves(out['log_pmf'],landmark,30,model.causes)[:,:,:][:,:,times].cpu().numpy())
                    prediction=np.concatenate(pred)
                    if exp=='Hard24':prediction=prediction[:,:1,:]
                    assert prediction.shape==truth.shape and np.isfinite(prediction).all()
                    assert (prediction>=-1e-6).all() and (prediction<=1+1e-6).all()
                    assert (np.diff(prediction[:,0,:],axis=-1)<=1e-6).all()
                    if exp=='CR3':np.testing.assert_allclose(prediction.sum(1),1,atol=3e-6,rtol=0)
                    err=np.abs(prediction-truth).mean(-1);fit_errors.append(err)
                    d=self.out/'evaluation'/exp;d.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(d/f'seed_{seed}_L{L}.npz',patient_ids=ids,times=times,prediction=prediction,patient_MAE=err)
                    for k,target in enumerate(targets):metrics.append(dict(experiment=exp,seed=seed,landmark=L,target=target,MAE=float(err[:,k].mean()),n_patients=len(ids)))
                    del model
                means[L]=np.mean(fit_errors,axis=0);saved_errors[f'patient_MAE_L{L}']=means[L];saved_errors[f'patient_ids_L{L}']=ids
            baseline=patients['patient_ids'];multiplicity=np.random.default_rng(20261101).multinomial(len(baseline),np.full(len(baseline),1/len(baseline)),size=resamples)
            np.savez_compressed(self.out/'evaluation'/exp/'mean_patient_errors.npz',**saved_errors)
            for L,new_error in means.items():
                ids=comparators[f'patient_ids_L{L}'];loc=np.array([patient_map[int(v)] for v in ids]);weights=multiplicity[:,loc];denom=weights.sum(1)
                if not (denom>0).all():raise RuntimeError('Empty eligible bootstrap sample')
                old=comparators[f'patient_MAE_L{L}']
                for k,target in enumerate(targets):
                    for j,name in enumerate(comparators['model_names']):
                        differences=new_error[:,k]-old[:,k,j];boot=(weights@differences)/denom;ci=np.quantile(boot,[.025,.975]);old_mean=float(old[:,k,j].mean());new_mean=float(new_error[:,k].mean())
                        comparisons.append(dict(experiment=exp,landmark=L,target=target,model='Optimized-DDH',comparator=str(name),optimized_MAE=new_mean,comparator_MAE=old_mean,difference=new_mean-old_mean,CI_lower=float(ci[0]),CI_upper=float(ci[1]),relative_reduction_pct=100*(1-new_mean/old_mean),training_fits=len(self.protocol['final_seeds'][exp]),resamples=resamples,n_patients=len(ids)))
        for name,rows in [('per_seed_MAE.csv',metrics),('comparisons.csv',comparisons)]:
            with (self.out/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        atomic_json(self.out/'evaluation_complete.json',dict(protocol_hash=self.protocol_hash,comparison_rows=len(comparisons),test_evaluation_n=300 if not self.smoke else 'synthetic fixture',bootstrap_resamples=resamples,intervals='pointwise paired patient; fixed fits and configurations'))
        return comparisons

    def export(self):
        import shutil
        path=shutil.make_archive(str(self.out.parent/(self.out.name+'_results')),'zip',self.out)
        return path

    def evaluate_sensitivity(self):
        """Paired monitor effects on fixed rerun paths, separately from optimized DDH."""
        if not (self.out/'evaluation_complete.json').exists():raise RuntimeError('Complete locked final evaluation first')
        schedule=json.loads((self.bundle/'sensitivity_schedule.json').read_text());rows=[]
        for exp in self.protocol['experiments']:
            wanted=[z for z in schedule if z['experiment']==exp and (not self.smoke or z['seed']==42)]
            destination=self.out/'test_payload'/exp
            with np.load(destination/'patients.npz') as z:patients={k:z[k] for k in z.files}
            patient_map={int(v):j for j,v in enumerate(patients['patient_ids'])}
            targets=['Survival'] if exp=='Hard24' else ['Survival','CIF_Death_A','CIF_Death_B']
            for scheduled in wanted:
                seed=scheduled['seed'];d=self.out/exp/'sensitivity'/f'seed_{seed}'
                if not (d/'summary.json').exists():raise RuntimeError('Run complete sensitivity phase first')
                summary=json.loads((d/'summary.json').read_text())
                assert summary['epochs_completed']>=scheduled['epochs_logged'] or self.smoke
                for L in ([0,20,23,24] if exp=='Hard24' else [0,10,20]):
                    with np.load(destination/f'reference_L{L}.npz') as z:ref={k:z[k] for k in z.files}
                    ix=np.array([patient_map[int(v)] for v in ref['patient_ids']]);truth=ref['truth'];truth=truth[:,None,:] if truth.ndim==2 else truth
                    errors={};epochs={}
                    for monitor in ['concordance','total_loss','prefix_nll']:
                        ck=torch.load(d/f'best_{monitor}.pt',map_location=self.device,weights_only=False);model=DDH(ck['config']['model']).to(self.device);model.load_state_dict(ck['state_dict']);model.eval();pred=[]
                        with torch.no_grad():
                            for indices in np.array_split(ix,max(1,math.ceil(len(ix)/256))):
                                x=torch.tensor((patients['x'][indices]-np.array(ck['mean']))/np.array(ck['sd']),dtype=torch.float32,device=self.device);states=torch.tensor(patients['states'][indices],dtype=torch.long,device=self.device);landmark=torch.full((len(indices),),L,dtype=torch.long,device=self.device)
                                batch=inputs(x,states,landmark,30,3 if exp=='Hard24' else 7);out=model(batch);curves=base.pmf_curves(out['log_pmf'],landmark,30,model.causes)
                                pred.append(curves[:,:1 if exp=='Hard24' else 3,ref['times']].cpu().numpy())
                        prediction=np.concatenate(pred);errors[monitor]=np.abs(prediction-truth).mean(-1);epochs[monitor]=ck['epoch'];del model
                    np.savez_compressed(d/f'test_errors_L{L}.npz',patient_ids=ref['patient_ids'],**errors)
                    for monitor in ['total_loss','prefix_nll']:
                        for k,target in enumerate(targets):rows.append(dict(experiment=exp,seed=seed,landmark=L,target=target,selection=monitor,concordance_epoch=epochs['concordance'],alternative_epoch=epochs[monitor],concordance_MAE=float(errors['concordance'][:,k].mean()),alternative_MAE=float(errors[monitor][:,k].mean()),difference=float((errors[monitor][:,k]-errors['concordance'][:,k]).mean()),interpretation='matched rerun path; not recovered original intermediate weights'))
        with (self.out/'checkpoint_sensitivity.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        return rows
