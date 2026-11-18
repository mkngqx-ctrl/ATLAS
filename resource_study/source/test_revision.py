from pathlib import Path
import ast,sys,types,unittest,json
import numpy as np
import torch
from torch import nn
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.resolve()))
import reference_recursion as ref

def selected(path,names,extra=None):
    s=Path(path).read_text(encoding='utf-8');tree=ast.parse(s)
    ns={'np':np,'torch':torch,'nn':nn,'F':torch.nn.functional,'HEALTHY':0,'SICK':1,'SICKER':2,'DEAD':3,'CENSORED':4,'N_STATES':5,'T_K':30,
        'DEFAULT_HAZARD_PARAMS':dict(lambda_sick_to_sicker=.35,kappa_sick_to_sicker=2.5,lambda_sicker_to_dead=.97,kappa_sicker_to_dead=3.)}
    if extra:ns.update(extra)
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    exec(compile('from __future__ import annotations\n'+'\n'.join(ast.get_source_segment(s,n) for n in nodes),str(path),'exec'),ns)
    return ns

class RevisionChecks(unittest.TestCase):
    def test_syntax(self):
        for p in R.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),str(p))
    def test_common_time_ranking(self):
        ns=selected(R/'dynamic_deephit_model.py',['l2_loss'])
        cif=torch.tensor([[.3,.6,.8],[.1,.9,.95]],requires_grad=True)
        loss=ns['l2_loss'](cif,torch.tensor([1,2]),torch.tensor([1,1]))
        self.assertAlmostEqual(loss.item(),np.exp(-.2),places=6)
        loss.backward();self.assertNotEqual(cif.grad[1,0].item(),0);self.assertEqual(cif.grad[1,1].item(),0)
    def test_cr_cross_cause_pair(self):
        ns=selected(R/'dynamic_deephit_model_cr.py',['l2_loss_cr'])
        cif=torch.tensor([[[.3,.4],[.1,.2]],[[.1,.2],[.4,.6]]])
        for later in [0,1,2]:
            loss=ns['l2_loss_cr'](cif,torch.tensor([0,max(0,later-1)]),torch.tensor([1,2]),torch.tensor([1,later]))
            self.assertAlmostEqual(loss.item(),np.exp(-.2),places=6)
    def test_tail_likelihood(self):
        ns=selected(R/'dynamic_deephit_model.py',['l1_loss'])
        logits=torch.zeros((1,31),requires_grad=True);pmf=logits.softmax(1)
        loss,_=ns['l1_loss'](pmf,torch.tensor([30]),torch.tensor([0]));self.assertAlmostEqual(loss.item(),np.log(31),places=5)
        loss.backward();self.assertLess(logits.grad[0,-1].item(),0)
        ns=selected(R/'dynamic_deephit_model_cr.py',['l1_loss_cr'])
        logits=torch.zeros((1,61),requires_grad=True);pmf=logits.softmax(1)[:,:-1].reshape(1,2,30)
        loss,*_=ns['l1_loss_cr'](pmf,torch.tensor([30]),torch.tensor([0]));self.assertAlmostEqual(loss.item(),np.log(61),places=5)
        loss.backward();self.assertLess(logits.grad[0,-1].item(),0)
    def test_corrected_model_forward_and_gradient(self):
        for stem,clsname in [('dynamic_deephit_model','DynamicDeepHitModel'),('survtransformer_model','SurvTransformerModel')]:
            for suffix in ['', '_sm','_cr']:
                cls=clsname.replace('Model','CRModel') if suffix=='_cr' else clsname
                path=R/(stem+suffix+'.py')
                ns=selected(path,['PositionalEncoding',cls,'l1_loss','l2_loss','combined_loss','l1_loss_cr','l2_loss_cr','combined_loss_cr'])
                model=ns[cls](L=3);out=model(torch.tensor([0.,1.]))
                pmf=out[0] if isinstance(out,tuple) else out
                if suffix=='_cr':
                    self.assertEqual(tuple(pmf.shape),(2,2,3));self.assertTrue((pmf.sum((1,2))<1).all())
                    loss,*_=ns['combined_loss_cr'](pmf,torch.tensor([1,3]),torch.tensor([1,0]))
                else:
                    self.assertEqual(tuple(pmf.shape),(2,4));self.assertTrue((pmf[:,-1]>0).all())
                    loss,*_=ns['combined_loss'](pmf,torch.tensor([1,3]),torch.tensor([1,0]))
                loss.backward();self.assertTrue(torch.isfinite(model.head.weight.grad).all())
    def test_cr_brier_other_event(self):
        ns=selected(R/'model_comparison_competing_risks.py',['integrated_brier_score_cr'])
        g=lambda t:np.ones_like(t,dtype=float)
        bs,ibs=ns['integrated_brier_score_cr'](np.full((3,2),.2),[1,2],[1,1,3],[1,2,0],1,g)
        np.testing.assert_allclose(bs,[(.64+.04+.04)/3]*2)
    def test_discrete_weibull(self):
        ns=selected(R/'wtte_model.py',['wtte_loss']);a=torch.tensor([2.,2.],requires_grad=True);b=torch.ones(2,requires_grad=True)
        loss=ns['wtte_loss'](a,b,torch.tensor([1,3]),torch.tensor([1,0]))
        self.assertAlmostEqual(loss.item(),(-np.log(1-np.exp(-.5))+1.5)/2,places=6)
        loss.backward();self.assertTrue(torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all())
    def test_ipcw_left_limit_and_support(self):
        ns=selected(R/'model_comparison.py',['integrated_brier_score'])
        def g(t):return np.where(t<1,1.,np.where(t<3,.5,0.))
        bs,ibs=ns['integrated_brier_score'](np.full((2,2),.5),[1,2],[1,3],[1,0],g)
        np.testing.assert_allclose(bs,[.375,.375]);self.assertAlmostEqual(ibs,.375)
        bs,ibs=ns['integrated_brier_score'](np.full((2,2),.5),[2,3],[1,3],[1,0],g)
        self.assertTrue(np.isnan(bs[-1]) and np.isnan(ibs))
    def test_independent_seeds(self):
        seeds=np.random.SeedSequence(42).spawn(20);self.assertEqual(len(set(int(s.generate_state(1)[0]) for s in seeds)),20)
        for p in R.glob('*sfsa*.py'):self.assertNotIn('child_seed.entropy',p.read_text(encoding='utf-8'))
    def test_exact_kernel_agreement(self):
        base=selected(R/'sfsa_data_generation.py',['transition_matrix'])
        sm=selected(R/'sfsa_semi_markov.py',['sojourn_boosted_transition_matrix'],base)
        for scenario in ref.SCENARIOS:
            for x in ([30.,70.,120.] if scenario=='ReLU' else [7.,10.,13.]):
                states,init,k=ref.kernels(x,scenario,'semi')
                for t in [2,19,20,30]:
                    u=np.array([z[1] for z in states]);s=np.array([z[0] for z in states]);n=len(s)
                    m=sm['sojourn_boosted_transition_matrix'](np.full(n,ref.covariate_F(x,scenario)),ref.SCENARIOS[scenario]['p'],u,t)[np.arange(n),s]
                    q,d,c=k[t-2];np.testing.assert_allclose(d,m[:,3],atol=1e-14);np.testing.assert_allclose(c,m[:,4],atol=1e-14)
                    for dest in range(3):np.testing.assert_allclose(q[:,np.array([z[0]==dest for z in states])].sum(1),m[:,dest],atol=1e-14)
    def test_simulator_reference_agreement(self):
        base=selected(R/'sfsa_data_generation.py',['transition_matrix','_simulate_states'])
        sm=selected(R/'sfsa_semi_markov.py',['sojourn_boosted_transition_matrix','_simulate_from_sm','_simulate_states_sm'],base)
        report=[]
        for scenario,x in [('ReLU',70.),('Weibull',12.5)]:
            config=types.SimpleNamespace(F=lambda v:ref.covariate_F(v,scenario) if scenario!='ReLU' else np.where(v<=70,.75,-.012*v+1.59),p=ref.SCENARIOS[scenario]['p'])
            for kind in ['base','semi']:
                fn=base['_simulate_states'] if kind=='base' else sm['_simulate_states_sm']
                paths=fn(np.full(80000,x),config,np.random.default_rng(823),30)
                states,init,k=ref.kernels(x,scenario,kind);masses,truth=ref.forward(init,k)
                est=[1.,1.]
                for t in range(1,30):
                    at_risk=paths[:,t-1]<3;events=at_risk&(paths[:,t]==3)
                    est.append(est[-1]*(1-events.sum()/at_risk.sum()))
                error=float(np.max(abs(np.array(est)-truth)));self.assertLess(error,.012)
                for L in [5,20,25]:self.assertLess(abs(np.mean(paths[:,L-1]<3)-masses[L].sum()),.012)
                report.append(dict(scenario=scenario,generator=kind,n=80000,max_KM_difference=error))
        (R/'simulation_validation.json').write_text(json.dumps(report,indent=2))
    def test_competing_risk_collapse(self):
        base=selected(R/'sfsa_data_generation.py',['transition_matrix','_simulate_states'])
        cr=selected(R/'sfsa_competing_risks.py',['default_cause_a_share','competing_risks_transition_matrix','_simulate_from_cr','_simulate_states_cr'],{'CENSORED':5,'DEAD_DISEASE':3,'DEAD_OTHER':4,'N_STATES_CR':6})
        for scenario in ref.SCENARIOS:
            cfg=ref.SCENARIOS[scenario]
            config=types.SimpleNamespace(F=(lambda v:np.where(v<=70,.75,-.012*v+1.59)) if scenario=='ReLU' else (lambda v:ref.covariate_F(v,'Weibull')),p=cfg['p'],covariate_mean=cfg['mean'],covariate_sd=cfg['sd'])
            x=np.linspace(cfg['low'],cfg['high'],4000)
            for seed in [0,17]:
                a=base['_simulate_states'](x,config,np.random.default_rng(seed),30)
                b=cr['_simulate_states_cr'](x,config,np.random.default_rng(seed),30)
                collapse=np.where(b==5,4,np.where(b==4,3,b))
                np.testing.assert_array_equal(a,collapse)
    def test_glazier_causality(self):
        path=R/'glazier_transformer_model.py';tree=ast.parse(path.read_text())
        names=[n.name for n in tree.body if isinstance(n,ast.ClassDef)]
        ns=selected(path,names);model=ns['GlazierTransformerModel'](dropout=0.).eval()
        x=torch.tensor([[5,0,1,2],[5,0,2,0]])
        with torch.no_grad():y=model.forward_teacher_forced(x,torch.zeros(2))
        torch.testing.assert_close(y[0,:2],y[1,:2])
    def test_information_order(self):
        for scenario,x in [('ReLU',70.),('Weibull',10.)]:
            for kind in ['base','semi']:
                for r in ref.risks_at_x(x,scenario,kind):
                    self.assertLessEqual(r['state_floor'],r['floor']+1e-12);self.assertLessEqual(r['floor'],r['ratio']+1e-12)
if __name__=='__main__':unittest.main(verbosity=2)
