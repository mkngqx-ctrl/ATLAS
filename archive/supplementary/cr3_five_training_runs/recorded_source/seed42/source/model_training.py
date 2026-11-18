"""Training-only model code: no simulator kernels or reference curves imported."""
import copy, random
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from history_models import (conditional_log_pmf,pmf_curves,dynamic_rank,hazard_curves,
                            DynamicDeepHitHistory,SurvTRACEHistory)
from history_training import train_history,predict_history
from legacy_models import (DynamicDeepHitCRModel,SurvTransformerCRModel,WTTEModelCR,
                           DRSACRModel,GlazierModel,GlazierTransformerModel)

LABELS=['CPH','DynamicDeepHIt-Lite','SurvTransformer-Lite','Weibull','DRSA',
        'Glazier-GRU','Glazier-Transformer','DynamicDeepHit','SurvTransformer','SurvTransformer-Discrete']
HISTORY=('DynamicDeepHit','SurvTransformer','SurvTransformer-Discrete')
GLAZIER=('Glazier-GRU','Glazier-Transformer')
CLASSES={c.__name__:c for c in (DynamicDeepHitCRModel,SurvTransformerCRModel,WTTEModelCR,
         DRSACRModel,GlazierModel,GlazierTransformerModel,DynamicDeepHitHistory,SurvTRACEHistory)}


def pmf_loss(logp,d,c,H,rank_weight=1.,rank_sigma=1.):
    row=torch.arange(len(d),device=d.device); L=torch.zeros_like(d)
    idx=(c-1).clamp_min(0)*H+d-1
    times=torch.arange(1,H+1,device=d.device).repeat(2)
    after=torch.cat([times[None,:]>d[:,None],torch.ones((len(d),1),dtype=torch.bool,device=d.device)],1)
    censor=torch.logsumexp(logp.masked_fill(~after,-torch.inf),1)
    nll=-torch.where(c>0,logp[row,idx],censor).mean()
    return nll+rank_weight*dynamic_rank(logp,d,c,L,horizon=H,causes=2,sigma=rank_sigma,reduction='pairs')


def drsa_loss(logprob,d,c,alpha=.5):
    row=torch.arange(len(d),device=d.device)
    logS=logprob[:,:,0].cumsum(1)
    at=logS[row,d-1]
    previous=F.pad(logS,(1,0))[row,d-1]
    death=-(previous+logprob[row,d-1,c])
    uncensored=-torch.log((-torch.expm1(at)).clamp_min(torch.finfo(at.dtype).tiny))
    return (alpha*torch.where(c>0,death,0.)+(1-alpha)*torch.where(c>0,uncensored,-at)).mean()


def weibull_loss(params,d,c):
    ad,bd,ab,bb=params; y=d.float()
    def log_h(a,b):return b.log()-a.log()+(b-1)*(y/a).log()
    cumulative=(y/ad).pow(bd)+(y/ab).pow(bb)
    return (cumulative-(c==1)*log_h(ad,bd)-(c==2)*log_h(ab,bb)).mean()


def glazier_loss(logits,target,previous):
    valid=(previous<7)|(previous==10)
    loss=F.cross_entropy(logits.reshape(-1,10),target.reshape(-1),reduction='none').reshape_as(target)
    return (loss*valid).sum()/valid.sum().clamp_min(1)


def cox_fit(x,d,c):
    from lifelines import CoxPHFitter
    models=[]
    for k in (1,2):
        if not np.any(c==k):raise ValueError(f'No training deaths for cause {k}; increase cohort size')
        frame=pd.DataFrame(x,columns=['Calcium','Age','Drug'])
        frame['duration']=d;frame['event']=(c==k).astype(int)
        model=CoxPHFitter(penalizer=.1).fit(frame,'duration','event')
        models.append(model)
    return models


def fit(label,cohort,horizon=30,epochs=200,patience=15,seed=42,device='cpu',options=None):
    if label not in LABELS:raise ValueError(label)
    options=copy.deepcopy(options or {})
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    tr,va=cohort['train_index'],cohort['val_index']
    x=cohort['covariates'];states=cohort['states'];d=cohort['duration'];c=cohort['cause']
    mean=x[tr].mean(0);sd=np.maximum(x[tr].std(0),1e-8)
    if label=='CPH':
        model=cox_fit((x[tr]-mean)/sd,d[tr],c[tr])
        return model,dict(x_mean=mean.tolist(),x_sd=sd.tolist(),horizon=horizon,device=device,training_log=[])
    if label in HISTORY:
        config=dict(options.pop('model_config',{}));config.update(horizon=horizon,causes=2,n_live=7,n_covariates=3)
        if label!='DynamicDeepHit':config['likelihood']='discrete' if label.endswith('-Discrete') else 'pch'
        return train_history(x[tr],states[tr],d[tr],c[tr],x[va],states[va],d[va],c[va],
            architecture='deephit' if label=='DynamicDeepHit' else 'survtrace',model_config=config,
            max_epochs=epochs,patience=patience,seed=seed,device=device,**options)
    config=dict(options.pop('model_config',{}))
    if label=='Glazier-GRU':constructor=GlazierModel;config['n_states']=10
    elif label=='Glazier-Transformer':constructor=GlazierTransformerModel;config.update(n_states=10,max_len=horizon+1)
    elif label=='DynamicDeepHIt-Lite':constructor=DynamicDeepHitCRModel;config['L']=horizon
    elif label=='SurvTransformer-Lite':constructor=SurvTransformerCRModel;config['L']=horizon
    elif label=='DRSA':constructor=DRSACRModel;config['L']=horizon
    else:constructor=WTTEModelCR;config.setdefault('init_alpha',max(float(np.median(d[tr])),2.))
    model=constructor(**config).to(device)
    lr=options.pop('lr',1e-3);batch_size=options.pop('batch_size',256)
    loss_options=options.pop('loss_config',{})
    if options:raise ValueError('Unrecognized options: '+str(options))
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    def tensors(ids):
        return (torch.tensor((x[ids]-mean)/sd,dtype=torch.float32,device=device),
                torch.tensor(states[ids],dtype=torch.long,device=device),
                torch.tensor(d[ids],dtype=torch.long,device=device),
                torch.tensor(c[ids],dtype=torch.long,device=device))
    train=tensors(tr);val=tensors(va)
    def loss(batch):
        xx,ss,dd,cc=batch
        if label in GLAZIER:
            previous=torch.cat([torch.full((len(xx),1),10,device=device,dtype=torch.long),ss[:,:-1]],1)
            return glazier_loss(model.forward_teacher_forced(previous,xx),ss,previous)
        output=model(xx)
        if label=='DynamicDeepHIt-Lite':output=output[0]
        if label in ('DynamicDeepHIt-Lite','SurvTransformer-Lite'):return pmf_loss(output,dd,cc,horizon,**loss_options)
        if label=='DRSA':return drsa_loss(output,dd,cc,**loss_options)
        return weibull_loss(output,dd,cc)
    best=float('inf');stale=0;best_state=None;logs=[]
    for epoch in range(epochs):
        model.train();order=torch.randperm(len(tr),device=device);total=0.
        for start in range(0,len(tr),batch_size):
            ids=order[start:start+batch_size];value=loss([a[ids] for a in train])
            if not torch.isfinite(value):raise FloatingPointError(label+' nonfinite loss')
            optimizer.zero_grad(set_to_none=True);value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
            total+=float(value.detach())*len(ids)
        model.eval();v=0.
        with torch.no_grad():
            for start in range(0,len(va),batch_size):
                b=[a[start:start+batch_size] for a in val];value=loss(b)
                if not torch.isfinite(value):raise FloatingPointError(label+' nonfinite validation loss')
                v+=float(value)*len(b[0])
        v/=len(va);logs.append(dict(epoch=epoch+1,train_loss=total/len(tr),validation_loss=v))
        print(f'  {label} epoch {epoch+1}: train={total/len(tr):.5f}, validation={v:.5f}',flush=True)
        if v<best-1e-6:
            best=v;stale=0;best_epoch=epoch+1
            best_state={k:t.detach().cpu().clone() for k,t in model.state_dict().items()}
        else:stale+=1
        if stale>=patience:break
    if best_state is None:raise RuntimeError('No finite checkpoint')
    model.load_state_dict(best_state);model.eval()
    return model,dict(model_config=config,x_mean=mean.tolist(),x_sd=sd.tolist(),horizon=horizon,
                      device=device,training_log=logs,best_epoch=best_epoch,seed=seed,
                      lr=lr,batch_size=batch_size,loss_config=loss_options)


def drsa_curves(logprob,L):
    H=logprob.shape[1];mask=torch.arange(1,H+1,device=logprob.device)>L
    logS=(logprob[:,:,0]*mask).cumsum(1);S=logS.exp()
    previous=F.pad(S[:,:-1],(1,0),value=1.)
    cif=(previous[:,:,None]*logprob[:,:,1:].exp()*mask[None,:,None]).cumsum(1).transpose(1,2)
    return torch.cat([F.pad(S,(1,0),value=1.)[:,None,:],F.pad(cif,(1,0))],1)


def weibull_curves(params,L,H):
    from scipy.integrate import quad
    result=np.zeros((len(params[0]),3,H-L))
    for i,(ad,bd,ab,bb) in enumerate(zip(*params)):
        base=(L/ad)**bd+(L/ab)**bb
        def logS(t):return base-(t/ad)**bd-(t/ab)**bb
        def densityA(t):return np.exp(logS(t)+np.log(bd/ad)+(bd-1)*np.log(t/ad))
        s_previous=1.; fa=fb=0.
        for j,t in enumerate(range(L+1,H+1)):
            s=np.exp(logS(t));mass=max(0.,s_previous-s)
            a,error=quad(densityA,t-1,t,epsabs=1e-10,epsrel=1e-9,limit=200)
            if error>1e-7 or a < -1e-8 or a>mass+1e-8:raise ArithmeticError('Weibull quadrature accuracy failure')
            a=min(max(a,0.),mass);fa+=a;fb+=mass-a
            result[i,:,j]=s,fa,fb;s_previous=s
    return result


def predict(label,model,meta,x,prefix,L,horizon=30,batch_size=512):
    x=np.asarray(x);xx=(x-meta['x_mean'])/meta['x_sd']
    if label in HISTORY:
        return predict_history(model,meta,x,np.arange(L+1,horizon+1),prefix=prefix,landmark=L,batch_size=batch_size)
    if label=='CPH':
        frame=pd.DataFrame(xx,columns=['Calcium','Age','Drug']);hazards=[]
        for fit_model in model:
            series=fit_model.baseline_cumulative_hazard_.iloc[:,0]
            times=series.index.to_numpy();index=np.searchsorted(times,np.arange(horizon+1),side='right')-1
            baseline=np.where(index>=0,series.to_numpy()[index.clip(min=0)],0.)
            risk=fit_model.predict_partial_hazard(frame).to_numpy()
            hazards.append(risk[:,None]*np.diff(baseline)[None,:])
        a=torch.tensor(np.stack(hazards,1),dtype=torch.float64)
        return hazard_curves(a,torch.full((len(x),),L))[:,:,L+1:].numpy()
    device=next(model.parameters()).device;result=[];model.eval()
    with torch.no_grad():
        for start in range(0,len(x),batch_size):
            xb=torch.tensor(xx[start:start+batch_size],dtype=torch.float32,device=device)
            output=model(xb)
            if label=='Weibull':
                result.append(weibull_curves([p.double().cpu().numpy() for p in output],L,horizon));continue
            if label=='DynamicDeepHIt-Lite':output=output[0]
            if label=='DRSA':curves=drsa_curves(output,L)
            else:curves=pmf_curves(output,torch.full((len(xb),),L,device=device),horizon,2)
            result.append(curves[:,:,L+1:].cpu().numpy())
    return np.concatenate(result)
