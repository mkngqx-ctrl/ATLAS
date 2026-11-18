"""Compare new sampling pipeline directly with archived ATLAS samplers on toys."""
import run_classical_sampling as run
import numpy as np
import torch,importlib.util,sys


def main():
    for exp,nlive in [('Hard24',3),('CR3',7)]:
        nstates=5 if nlive==3 else 10
        rng=np.random.default_rng(778)
        p=rng.dirichlet(np.ones(nstates),size=nlive)
        initial=rng.dirichlet(np.ones(nstates))
        toy=run.Model.__new__(run.Model);toy.exp=exp;toy.kind='Markov';toy.nlive=nlive;toy.nstates=nstates;toy.initial=initial[None,:];toy.q=p[None,:,:]
        class Adapter(torch.nn.Module):
            START_TOKEN=nstates
            def __init__(self):super().__init__();self.dummy=torch.nn.Parameter(torch.zeros(1,dtype=torch.float64))
            def next_state_logits(self,h,x):
                last=h[:,-1].cpu().numpy();out=np.zeros((len(last),nstates))
                for j,s in enumerate(last):out[j]=initial if s==nstates else p[s] if s<nlive else np.eye(nstates)[s]
                return torch.tensor(np.log(out,where=out>0,out=np.full_like(out,-np.inf)),dtype=torch.float64)
        if exp=='Hard24':
            path=run.ROOT/'history/2026-09-13/thi/outputs/Glazier_Weibull_HardRule24/autoregressive_samplers.py'
        else:
            path=run.ROOT/'current/working/ATLAS_SiM_data_and_code/supplementary/cr3_five_training_runs/recorded_source/seed42/source/samplers_cr.py'
            sys.path.insert(0,str(path.parent))
        spec=importlib.util.spec_from_file_location('archived_sampler_'+exp,path);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        for method in ['iid','sobol']:
            for L in [0,20]:
                seed=123456
                u=run.uniforms(method,30-L,seed)
                np.testing.assert_array_equal(u,old.uniforms(method,10000,30-L,seed))
                actual=toy.sample(0,L,1,L,u)[0]
                prefix=np.ones(L,dtype=int)
                if exp=='Hard24':
                    expected,_=old.sample_curve(Adapter(),dict(T_K=30,device='cpu',x_mean=0,x_sd=1),0,M=10000,method=method,seed=seed,prefix=prefix)
                    expected=expected[None,:]
                else:
                    expected,_=old.sample_curves(Adapter(),dict(horizon=30,x_mean=[0,0,0],x_sd=[1,1,1]),np.zeros(3),M=10000,method=method,seed=seed,prefix=prefix)
                np.testing.assert_allclose(actual,expected,atol=1e-14,rtol=0)
        # Duration-independent probabilities in a semi-Markov table should
        # produce exactly the same sampled outcomes, including prefix updates.
        semi=run.Model.__new__(run.Model);semi.__dict__.update(toy.__dict__);semi.kind='SemiMarkov'
        semi.q=np.broadcast_to(p[None,None,:,None,:],(1,30,nlive,30,nstates)).copy()
        for L in [0,20]:
            u=run.uniforms('sobol',30-L,789)
            np.testing.assert_array_equal(toy.sample(0,L,1,L,u)[0],semi.sample(0,L,1,L,u)[0])
            actual=toy.exact(np.array([0]),L,np.array([1]),np.array([L]))
            np.testing.assert_allclose(actual,semi.exact(np.array([0]),L,np.array([1]),np.array([L])),atol=1e-14,rtol=0)
            mass=np.eye(nlive)[1] if L else None;survival=1.;cif=np.zeros(1 if exp=='Hard24' else 2);expected=[]
            for dest in range(L+1,31):
                nxt=initial.copy() if dest==1 else mass@p
                hazard=nxt[nlive:nlive+len(cif)]/nxt.sum();cif+=survival*hazard;survival*=1-hazard.sum();mass=nxt[:nlive]
                expected.append([survival] if exp=='Hard24' else [survival,*cif])
            np.testing.assert_allclose(actual[0],np.array(expected).T,atol=1e-14,rtol=0)
        print('PASS',exp,'IID/Sobol uniforms and curves match archived ATLAS samplers; semi-Markov reduction agrees',flush=True)


if __name__=='__main__':main()
