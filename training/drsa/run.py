"""Frozen validation-selected DRSA study; run locally in gpu_env.
No analytical references or test outcomes are used before selections/fits lock.
"""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
from pathlib import Path
import argparse,copy,csv,gc,hashlib,json,math,platform,random,shutil,sys,time,traceback
from datetime import datetime,timezone
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/drsa'
SEEDS={'Hard24':[42,43,44,45,50,51,52,62,63],'CR3':[42,43,45,46,47]}
ALPHAS=[.05,.10,.25,.50,.75,.90]
LANDMARKS={'Hard24':[0,20,23,24],'CR3':[0,10,20]}
def stamp():return datetime.now(timezone.utc).isoformat()
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8');os.replace(tmp,p)
def save_torch(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');torch.save(obj,tmp);os.replace(tmp,p)
def status(**kw):
 write_json(OUT/'progress.json',dict(timestamp_UTC=stamp(),pid=os.getpid(),**kw))
def log(message):print(stamp(),message,flush=True)

class DRSA(nn.Module):
 def __init__(self,ncov,causes,config):
  super().__init__();p=config['projection'];drop=config['dropout']
  self.projection=nn.Sequential(nn.Linear(ncov,p),nn.ReLU(),nn.Dropout(drop)) if p else nn.Identity()
  self.lstm=nn.LSTM((p or ncov)+1,config['hidden'],config['layers'],batch_first=True,dropout=drop if config['layers']>1 else 0)
  self.output_dropout=nn.Dropout(drop);self.head=nn.Linear(config['hidden'],1 if causes==1 else causes+1);self.causes=causes
  self.register_buffer('clock',torch.arange(1,31,dtype=torch.float32).reshape(1,30,1)/30)
 def forward(self,x):
  e=self.projection(x);inp=torch.cat([e[:,None,:].expand(-1,30,-1),self.clock.expand(len(x),-1,-1)],-1)
  h,_=self.lstm(inp);a=self.head(self.output_dropout(h))
  if self.causes==1:
   a=a[:,:,0];return torch.stack([F.logsigmoid(-a),F.logsigmoid(a)],-1)
  return a.log_softmax(-1)

def terms(logp,d,c):
 row=torch.arange(len(d),device=d.device);s=F.pad(logp[:,:,0].cumsum(1),(1,0));at=s[row,d]
 event=-(s[row,d-1]+logp[row,d-1,c])
 status_loss=-torch.log((-torch.expm1(at)).clamp_min(torch.finfo(at.dtype).tiny))
 return s,event,status_loss,at
def composite(logp,d,c,alpha):
 s,event,st,at=terms(logp,d,c)
 return (alpha*torch.where(c>0,event,0)+(1-alpha)*torch.where(c>0,st,-at)).mean()
def l2_penalty(model,coefficient):
 # Explicit TF-style L2, 0.5*sum(theta^2), scaled per nominal batch128.
 # A fixed128 denominator prevents batch-size changes from changing penalty strength.
 return coefficient*.5*sum(p.square().sum() for p in model.parameters())/128 if coefficient else 0.
def probability_curves(logp,L):
 # Float64 propagation avoids numerical underflow in landmark division.
 p=logp.double();N,H,_=p.shape
 tail=p[:,L:];logS=F.pad(tail[:,:,0].cumsum(1),(1,0));S=logS.exp()
 cif=(S[:,:-1,None]*tail[:,:,1:].exp()).cumsum(1)
 return torch.cat([S[:,None,:],F.pad(cif.transpose(1,2),(1,0))],1)

def config(hidden,layers,projection,alpha,lr=.001,dropout=0,l2=0,batch=256):
 return dict(hidden=hidden,layers=layers,projection=projection,alpha=alpha,lr=lr,dropout=dropout,l2=l2,batch=batch)
def make_protocol():
 return dict(created_UTC=stamp(),study='Approved DRSA composite-objective optimization',date='2026-11-04',
  inputs='Fixed standardized covariates and normalized time only; no health-state history',horizon=30,
  objective='alpha*uncensored event-time NLL +(1-alpha)*event-status/censoring loss; mean over minibatch',
  alpha_values=ALPHAS,screen_architectures=[dict(hidden=h,layers=l,projection=p) for h,l,p in [(32,1,0),(128,1,0),(128,1,64),(256,2,128)]],
  screen=dict(seed=42,cumulative_epochs=[5,15,30],keep_per_alpha=[2,1],ranking='minimum validation score seen through current rung; ties candidate ID'),
  refinement=dict(top_alphas=2,recipes=[
   {'rule':'reuse winning screen configuration for this alpha'},
   config(64,1,64,0,lr=.0003,dropout=0,l2=0,batch=256),
   config(128,1,128,0,lr=.0001,dropout=.1,l2=.000001,batch=128),
   config(128,2,64,0,lr=.0007,dropout=.1,l2=.00001,batch=256),
   config(256,1,128,0,lr=.003,dropout=.2,l2=.0001,batch=128),
   config(256,2,128,0,lr=.0015,dropout=.1,l2=.00001,batch=256)],
   recipe_alpha_zero_is_placeholder=True,cumulative_epochs=[10,30,60],keep=[6,4],seed=42,
   clarification='Two alpha candidates are crossed with the same five new recipes plus each alpha screen winner; fixed balanced design, not exhaustive factorial'),
  confirmation=dict(finalists=2,seeds=[42,43,45],maximum_epochs=200,patience=15,minimum_improvement=1e-6,decision='lowest mean best validation score over three seeds'),
  selection_metric='Mean observed-outcome conditional NLL over landmarks with equal landmark weights; all eligible validation patients; no analytical curves/test MAE',
  validation_eligibility='recorded duration > landmark; event/censor at landmark excluded',
  optimizer=dict(name='Adam',betas=[.9,.99],gradient_clip=5),l2_definition='explicit0.5*sum(all learned parameters squared)*coefficient/128; no Adam weight decay',
  final_seeds=SEEDS,final_epochs=200,final_patience=15,confirmation_fits_reused='Selected configuration seeds42/43/45 reused unchanged for final evaluation',
  final_test='Original300baseline test patients and original eligible subsets from archived reference files; survival both experiments, both CIFs CR3',
  resampling='Use saved5000paired patient-bootstrap multiplicities for updated ATLAS-versus-DRSA comparisons',
  transparency='Motivated by original test findings. No original results/manuscript/export ZIPs overwritten. Tiny original saved-inference discrepancies remain separately documented; do not condition new search on test outputs.',
  precision='float32; no AMP, TF32 disabled; float64 curve propagation',resume='Last state/model/optimizer/RNG at every epoch; best state saved separately; immutable rung decision JSONs',
  data_inventory=[dict(entry, cohort=str(ROOT/'outputs/runs'/entry['experiment']/'42/checkpoints/cohort.npz'), evaluation_ids=str(ROOT/'outputs/runs'/entry['experiment']/'42/evaluation/L0/reference.npz')) for entry in json.loads((Path(__file__).parent/'inventory.json').read_text())])

class Data:
 def __init__(self,experiment,inventory):
  self.experiment=experiment;self.K=1 if experiment=='Hard24' else 2;self.landmarks=LANDMARKS[experiment]
  entry=next(x for x in inventory if x['experiment']==experiment);p=Path(entry['cohort'])
  assert digest(p)==entry['cohort_sha256']
  with np.load(p) as z:
   x=z['covariate' if self.K==1 else 'covariates'];d=z['duration'];c=z['event' if self.K==1 else 'cause'];tr=z['train_index'];va=z['val_index'];te=z['test_index']
  assert len(tr)==40000 and len(va)==5000 and not(set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te))
  if x.ndim==1:x=x[:,None]
  self.mean=x[tr].mean(0);self.sd=x[tr].std(0);self.ncov=x.shape[1]
  def tensor(ids):return (torch.tensor((x[ids]-self.mean)/self.sd,dtype=torch.float32,device='cuda'),torch.tensor(d[ids],dtype=torch.long,device='cuda'),torch.tensor(c[ids],dtype=torch.long,device='cuda'))
  self.train=tensor(tr);self.val=tensor(va);self.source=entry
  write_json(OUT/experiment/'data_provenance.json',{**entry,'standardization_mean':self.mean.tolist(),'standardization_SD':self.sd.tolist(),'training_n':len(tr),'validation_n':len(va),'input_columns':['calcium'] if self.K==1 else ['calcium','age','drug']})

def validate(model,data,batch):
 model.eval();sums=torch.zeros(len(data.landmarks),device='cuda',dtype=torch.float64);counts=torch.zeros_like(sums)
 with torch.no_grad():
  for start in range(0,len(data.val[0]),batch):
   xx,dd,cc=(a[start:start+batch] for a in data.val);logp=model(xx);S,event,st,at=terms(logp,dd,cc)
   nll=torch.where(cc>0,event,-at)
   for k,L in enumerate(data.landmarks):
    eligible=dd>L;sums[k]+=(nll[eligible]+S[eligible,L]).double().sum();counts[k]+=eligible.sum()
 assert bool((counts>0).all());scores=(sums/counts).cpu().tolist()
 assert all(math.isfinite(x) for x in scores)
 return float(np.mean(scores)),dict(zip(map(str,data.landmarks),scores))

def set_rng(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def capture_rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(r):
 random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda'])
def fit(data,cfg,fit_id,seed,budget,early_stopping,phase):
 directory=OUT/data.experiment/'fits'/fit_id/f'seed_{seed}';directory.mkdir(parents=True,exist_ok=True)
 if (directory/'config.json').exists():assert json.loads((directory/'config.json').read_text())==cfg
 else:write_json(directory/'config.json',cfg)
 set_rng(seed);model=DRSA(data.ncov,data.K,cfg).cuda();opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'],betas=(.9,.99));epoch=0;best=float('inf');stale=0;best_epoch=0
 if (directory/'last.pt').exists():
  state=torch.load(directory/'last.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);epoch=state['epoch'];best=state['best'];stale=state['stale'];best_epoch=state['best_epoch'];restore_rng(state['rng'])
  for optstate in opt.state.values():
   for k,v in optstate.items():
    if torch.is_tensor(v) and k!='step':optstate[k]=v.cuda()
 while epoch<budget and not(early_stopping and stale>=15):
  epoch+=1;start=time.perf_counter();model.train();torch.cuda.reset_peak_memory_stats();order=torch.randperm(len(data.train[0]),device='cuda');train_sum=0.
  for start_index in range(0,len(order),cfg['batch']):
   ids=order[start_index:start_index+cfg['batch']];xx,dd,cc=(a[ids] for a in data.train);loss=composite(model(xx),dd,cc,cfg['alpha'])+l2_penalty(model,cfg['l2'])
   if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss {fit_id} epoch{epoch}')
   opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step();train_sum+=float(loss.detach())*len(ids)
  score,parts=validate(model,data,cfg['batch'])
  if score<best-1e-6:
   best=score;stale=0;best_epoch=epoch
   save_torch(directory/'best.pt',dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},config=cfg,experiment=data.experiment,seed=seed,epoch=epoch,validation_score=score,validation_components=parts,x_mean=data.mean,x_sd=data.sd))
  else:stale+=1
  save_torch(directory/'last.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,best=best,stale=stale,best_epoch=best_epoch,rng=capture_rng()))
  torch.cuda.synchronize();seconds=time.perf_counter()-start
  row=dict(epoch=epoch,train_composite_with_penalty=train_sum/len(order),validation_NLL=score,best_validation_NLL=best,best_epoch=best_epoch,stale=stale,seconds=seconds,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,**{'NLL_L'+k:v for k,v in parts.items()})
  logfile=directory/'training.csv';exists=logfile.exists()
  with logfile.open('a',newline='') as f:w=csv.DictWriter(f,fieldnames=list(row));(None if exists else w.writeheader());w.writerow(row)
  status(phase=phase,experiment=data.experiment,fit=fit_id,seed=seed,epoch=epoch,budget=budget,best_validation_NLL=best,best_epoch=best_epoch,epoch_seconds=seconds)
 result=dict(fit_id=fit_id,seed=seed,config=cfg,best_validation_NLL=best,best_epoch=best_epoch,epochs_completed=epoch,checkpoint=str(directory/'best.pt'),stop='patience' if early_stopping and stale>=15 else 'budget')
 write_json(directory/'result.json',result);del model,opt;gc.collect();torch.cuda.empty_cache();return result

def decision(path,builder):
 if path.exists():return json.loads(path.read_text())
 result=builder();write_json(path,result);return result
def ranking(rows):return sorted(rows,key=lambda r:(r['best_validation_NLL'],r['fit_id']))

def search(experiment,protocol):
 data=Data(experiment,protocol['data_inventory']);exdir=OUT/experiment;decisions=exdir/'decisions';decisions.mkdir(exist_ok=True)
 screen={f'A{ai:02d}_M{mi}':config(a['hidden'],a['layers'],a['projection'],alpha) for ai,alpha in enumerate(ALPHAS) for mi,a in enumerate(protocol['screen_architectures'])}
 write_json(exdir/'screen_candidates.json',screen);active=list(screen);screen_last=[]
 for rung,budget in enumerate([5,15,30]):
  def build_screen(active=active,rung=rung,budget=budget):
   rows=[fit(data,screen[cid],cid,42,budget,False,f'screen_rung_{rung+1}') for cid in active]
   keep=2 if rung==0 else 1;selected=[]
   for alpha in ALPHAS:selected.extend(r['fit_id'] for r in ranking([r for r in rows if r['config']['alpha']==alpha])[:keep])
   return dict(rows=rows,selected=selected,budget=budget)
  state=decision(decisions/f'screen_rung_{rung+1}.json',build_screen);active=state['selected'];screen_last=state['rows']
 leaders=ranking(screen_last);top=leaders[:2];refine={}
 for ai,row in enumerate(top):
  alpha=row['config']['alpha']
  for ri,recipe in enumerate(protocol['refinement']['recipes']):
   cfg=copy.deepcopy(row['config'] if ri==0 else recipe);cfg['alpha']=alpha;refine[f'B{ai}_R{ri}']=cfg
 write_json(exdir/'refinement_candidates.json',refine);active=list(refine);ref_last=[]
 for rung,budget in enumerate([10,30,60]):
  def build_ref(active=active,rung=rung,budget=budget):
   rows=[fit(data,refine[cid],cid,42,budget,False,f'refinement_rung_{rung+1}') for cid in active]
   keep=[6,4,2][rung];return dict(rows=rows,selected=[r['fit_id'] for r in ranking(rows)[:keep]],budget=budget)
  state=decision(decisions/f'refinement_rung_{rung+1}.json',build_ref);active=state['selected'];ref_last=state['rows']
 finalists=ranking(ref_last)[:2]
 def confirm():
  candidates=[]
  for row in finalists:
   cid=row['fit_id'];fits=[fit(data,row['config'],'confirm_'+cid,seed,200,True,'confirmation') for seed in [42,43,45]]
   candidates.append(dict(candidate=cid,config=row['config'],fits=fits,mean_validation_NLL=float(np.mean([r['best_validation_NLL'] for r in fits]))))
  candidates.sort(key=lambda r:(r['mean_validation_NLL'],r['candidate']));return dict(candidates=candidates,selected=candidates[0])
 selected=decision(decisions/'confirmation.json',confirm)['selected'];write_json(exdir/'selected_configuration.json',selected)
 log(experiment+' configuration selected '+json.dumps(selected['config']))
 del data;gc.collect();torch.cuda.empty_cache();return selected

def final_training(experiment,selected,protocol):
 data=Data(experiment,protocol['data_inventory']);fits=[]
 for seed in SEEDS[experiment]:
  fits.append(fit(data,selected['config'],'confirm_'+selected['candidate'],seed,200,True,'final_training'))
 write_json(OUT/experiment/'final_fit_inventory.json',fits)
 del data;gc.collect();torch.cuda.empty_cache()

def run_checks():
 checks=[]
 def check(name,passed):
  checks.append(dict(name=name,passed=bool(passed)))
  if not passed:raise AssertionError(name)
 for K in [1,2]:
  N,H=5,30;torch.manual_seed(73);p=torch.randn(N,H,K+1,dtype=torch.float64).log_softmax(-1).requires_grad_();d=torch.tensor([1,2,17,30,30]);c=torch.tensor([1,0,K,1,0])
  for alpha in ALPHAS:
   expected=[]
   for i in range(N):
    logS=p[i,:d[i],0].sum();event=-(p[i,:d[i]-1,0].sum()+p[i,d[i]-1,c[i]]) if c[i]>0 else 0
    st=-torch.log1p(-logS.exp()) if c[i]>0 else -logS;expected.append(alpha*event+(1-alpha)*st)
   check(f'{K}causes alpha{alpha} independent loss',torch.allclose(composite(p,d,c,alpha),torch.stack(expected).mean(),atol=1e-10))
  composite(p,d,c,.5).backward();check(f'{K} finite gradients',torch.isfinite(p.grad).all())
  for L in [0,20,29]:
   curves=probability_curves(p.detach(),L);check(f'{K} L{L} mass conserved',torch.allclose(curves.sum(1),torch.ones(N,31-L,dtype=torch.float64)))
   check(f'{K} L{L} initialized',bool((curves[:,0,0]==1).all() and (curves[:,1:,0]==0).all()))
  m=DRSA(1 if K==1 else 3,K,config(128,1,64,.5));m.eval();x=torch.randn(N,1 if K==1 else 3)
  check(f'{K} projected shape',m(x).shape==(N,30,K+1));check(f'{K} interval normalized',torch.allclose(m(x).exp().sum(-1),torch.ones(N,30),atol=1e-6))
 write_json(OUT/'implementation_checks.json',dict(passed=sum(t['passed'] for t in checks),total=len(checks),tests=checks))

def evaluate(protocol):
 # This is deliberately called only after BOTH selections and all final fits exist.
 assert all((OUT/e/'final_fit_inventory.json').exists() for e in SEEDS)
 outdir=OUT/'evaluation';outdir.mkdir(exist_ok=True);rows=[];verification=[]
 for experiment in SEEDS:
  data=next(x for x in protocol['data_inventory'] if x['experiment']==experiment);K=1 if experiment=='Hard24' else 2
  with np.load(data['cohort']) as z:
   x=z['covariate' if K==1 else 'covariates'];tr=set(z['train_index']);va=set(z['val_index']);te=set(z['test_index'])
  if x.ndim==1:x=x[:,None]
  fits=json.loads((OUT/experiment/'final_fit_inventory.json').read_text());outcomes=[]
  for fitinfo in fits:
   cp=torch.load(fitinfo['checkpoint'],map_location='cpu',weights_only=False);model=DRSA(x.shape[1],K,cp['config']).cuda();model.load_state_dict(cp['model']);model.eval()
   for L in LANDMARKS[experiment]:
    ref=Path(data['evaluation_ids']).parent.parent/f'L{L}/reference.npz'
    with np.load(ref) as z:ids=z['patient_ids'];times=z['times'];truth=z['truth']
    assert set(ids)<=te and not(set(ids)&(tr|va));assert (times>L).all();assert len(ids)==300 if L==0 else len(ids)<=300
    xx=torch.tensor((x[ids]-cp['x_mean'])/cp['x_sd'],dtype=torch.float32,device='cuda')
    with torch.no_grad():curves=probability_curves(model(xx),L).cpu().numpy()
    pred=curves[:,:1,times-L] if K==1 else curves[:,:,times-L];targets=['Survival'] if K==1 else ['Survival','CIF_Death_A','CIF_Death_B']
    truth3=truth[:,None,:] if K==1 else truth
    assert pred.shape==truth3.shape and np.isfinite(pred).all()
    assert np.max(np.abs(curves.sum(1)-1))<1e-6
    dest=outdir/experiment/f'seed_{fitinfo["seed"]}'/f'L{L}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest,patient_ids=ids,times=times,prediction=pred,truth=truth3)
    for ti,target in enumerate(targets):
     err=pred[:,ti]-truth3[:,ti];post=times>=24
     rows.append(dict(experiment=experiment,training_seed=fitinfo['seed'],landmark=L,target=target,model='DRSA-optimized',method='direct',MAE=float(np.abs(err).mean()),RMSE=float(np.sqrt(np.square(err).mean())),post24_MAE=float(np.abs(err[:,post]).mean()) if experiment=='Hard24' and post.any() else '',n_patients=len(ids),selected_epoch=cp['epoch'],validation_NLL=cp['validation_score']))
    verification.append(dict(experiment=experiment,seed=fitinfo['seed'],landmark=L,patients=len(ids),test_membership=True,probability_mass_max_error=float(np.max(np.abs(curves.sum(1)-1))),checkpoint_sha256=digest(fitinfo['checkpoint'])))
   del model;gc.collect();torch.cuda.empty_cache()
  log(experiment+' test evaluation complete')
 with (outdir/'optimized_DRSA_per_seed_metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summary=[]
 for exp in SEEDS:
  for L in LANDMARKS[exp]:
   for target in (['Survival'] if exp=='Hard24' else ['Survival','CIF_Death_A','CIF_Death_B']):
    rr=[r for r in rows if r['experiment']==exp and r['landmark']==L and r['target']==target];a=np.array([r['MAE'] for r in rr])
    summary.append(dict(experiment=exp,landmark=L,target=target,mean_MAE=float(a.mean()),between_SD=float(a.std(ddof=1)),minimum=float(a.min()),maximum=float(a.max()),n_runs=len(a),mean_RMSE=float(np.mean([r['RMSE'] for r in rr]))))
 with (outdir/'optimized_DRSA_summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 write_json(outdir/'verification.json',verification)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--evaluate-only',action='store_true');args=parser.parse_args()
 OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);assert torch.cuda.is_available();torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 if not (OUT/'frozen_protocol.json').exists():write_json(OUT/'frozen_protocol.json',make_protocol())
 protocol=json.loads((OUT/'frozen_protocol.json').read_text())
 frozen=OUT/'run_drsa_optimization.py'
 if Path(__file__).resolve()!=frozen.resolve():
  if frozen.exists():assert digest(frozen)==digest(__file__),'Frozen source changed'
  else:shutil.copy2(__file__,frozen)
 write_json(OUT/'environment.json',dict(timestamp_UTC=stamp(),GPU=torch.cuda.get_device_name(0),GPU_GiB=torch.cuda.get_device_properties(0).total_memory/2**30,torch=torch.__version__,CUDA=torch.version.cuda,cudnn=torch.backends.cudnn.version(),python=platform.python_version(),cpu_threads=2,source_sha256=digest(frozen),protocol_sha256=digest(OUT/'frozen_protocol.json')))
 run_checks()
 if args.prepare:log('Protocol frozen and implementation checks passed');return
 if not args.evaluate_only:
  selections={exp:search(exp,protocol) for exp in SEEDS}
  for exp in SEEDS:final_training(exp,selections[exp],protocol)
 status(phase='test_evaluation');evaluate(protocol);status(phase='complete');write_json(OUT/'COMPLETED.json',dict(completed_UTC=stamp(),selection='validation only',manuscript_changed=False));log('DRSA STUDY COMPLETE')
if __name__=='__main__':
 try:main()
 except Exception:
  OUT.mkdir(parents=True,exist_ok=True);(OUT/'ERROR.txt').write_text(traceback.format_exc());status(phase='error',error=traceback.format_exc());raise
