"""Independent numerical, information-boundary and checkpoint integration checks.

Run: python -m unittest test_history_models -v
"""
from pathlib import Path
import sys, os, unittest, json, datetime, uuid, contextlib
if not hasattr(datetime, 'UTC'):
    datetime.UTC = datetime.timezone.utc
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'source'), str(ROOT)]
for parent in [ROOT, *ROOT.parents]:
    if (parent/'work/gpu_support').exists():
        sys.path.insert(0, str(parent/'work/gpu_support'))
        break
import numpy as np
import torch
from history_models import (history_inputs, DynamicDeepHitHistory, SurvTRACEHistory,
    conditional_log_pmf, pmf_survival, dynamic_rank, deephit_objective, survtrace_objective)
from history_training import PrefixDataset, train_history, predict_history, dynamic_concordance

torch.set_num_threads(2)


class HistoryMathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(193)

    def test_conditional_pmf_and_tail_by_hand(self):
        logp = torch.tensor([[.1, .2, .3, .4]]).log()
        L = torch.tensor([1])
        np.testing.assert_allclose(conditional_log_pmf(logp,L,3).exp(), [[0,2/9,3/9,4/9]], atol=1e-7)
        np.testing.assert_allclose(pmf_survival(logp,L,3), [[1,1,7/9,4/9]], atol=1e-7)

    def test_joint_causes_share_tail(self):
        p = torch.tensor([[.1,.2,.05,.15,.5]])
        conditional = conditional_log_pmf(p.log(),torch.tensor([1]),2,2).exp()
        np.testing.assert_allclose(conditional,[[0,.2/.85,0,.15/.85,.5/.85]],atol=1e-7)

    def test_deephit_conditional_event_and_censor_likelihood(self):
        model = DynamicDeepHitHistory(horizon=3,hidden=4,attention_hidden=4,head_hidden=4)
        batch = history_inputs(torch.zeros(2),torch.zeros((2,3)),torch.ones(2,dtype=torch.long),3)
        out = dict(log_pmf=torch.tensor([[.1,.2,.3,.4],[.1,.2,.3,.4]]).log(),
                   next_logits=torch.zeros((2,3,3)))
        loss, parts = deephit_objective(model,out,batch,torch.tensor([2,3]),torch.tensor([1,0]),
                                        rank_weight=0,auxiliary_weight=0,l1_weight=0)
        self.assertAlmostEqual(float(loss),(-np.log(2/9)-np.log(4/9))/2,places=6)

    def test_ranking_uses_elapsed_not_absolute_time(self):
        p = torch.tensor([[.1,.2,.3,.4],[.4,.1,.1,.4]]).log()
        got = dynamic_rank(p,torch.tensor([2,2]),torch.tensor([1,0]),torch.tensor([1,0]),
                           torch.tensor([11,12]),3,1,sigma=1)
        self.assertAlmostEqual(float(got),np.exp(-((2/9)-.4))/2,places=6)
        same = dynamic_rank(p,torch.tensor([2,2]),torch.tensor([1,0]),torch.tensor([1,0]),
                            torch.tensor([11,11]),3,1,sigma=1)
        self.assertEqual(float(same),0)

    def test_hazard_likelihoods_independent_hand_calculation(self):
        raw = torch.log(torch.expm1(torch.tensor([[.1,.2,.3],[.1,.2,.3]])))
        batch = {'landmark':torch.tensor([1,1])}
        out = dict(hazard_logits=raw,hazard=torch.nn.functional.softplus(raw),
                   mortality_logits=torch.zeros(2),duration_prediction=torch.zeros(2))
        for kind,expected in [('pch',(.2-np.log(.2)+.5)/2),
                               ('discrete',(-np.log1p(-np.exp(-.2))+.5)/2)]:
            model = SurvTRACEHistory(horizon=3,likelihood=kind)
            got,_ = survtrace_objective(model,out,batch,torch.tensor([2,3]),torch.tensor([1,0]),
                                        mortality_weight=0,duration_weight=0)
            self.assertAlmostEqual(float(got),expected,places=6)
            np.testing.assert_allclose(model.survival(out,batch['landmark']),
                                       [[1,1,np.exp(-.2),np.exp(-.5)]]*2,atol=1e-7)

    def test_conditioning_stays_finite_when_raw_tail_tiny(self):
        logp=torch.tensor([[0.,-1000.,-1001.,-1002.]]).log_softmax(1)
        s=pmf_survival(logp,torch.tensor([1]),3)
        self.assertTrue(torch.isfinite(s).all())
        self.assertAlmostEqual(float(s[0,1]),1,places=6)

    def test_auxiliary_mask_excludes_future_targets(self):
        model=DynamicDeepHitHistory(horizon=3,hidden=4,attention_hidden=4,head_hidden=4)
        batch=history_inputs(torch.zeros(1),torch.tensor([[0,2,1]]),torch.tensor([1]),3)
        pred=torch.zeros((1,3,3),requires_grad=True)
        out=dict(log_pmf=torch.full((1,4),.25).log(),next_logits=pred)
        _,parts=deephit_objective(model,out,batch,torch.tensor([3]),torch.tensor([0]))
        parts['next_measurement'].backward()
        self.assertGreater(float(pred.grad[0,0].abs().sum()),0)
        self.assertEqual(float(pred.grad[0,1:].abs().sum()),0)

    def test_prefix_expansion_excludes_terminal_and_future(self):
        ds=PrefixDataset([1,2],[[0,3,3],[1,2,0]],[2,3],[1,0],0,1,horizon=3)
        self.assertEqual(list(zip(ds.patient,ds.landmark)),[(0,0),(0,1),(1,0),(1,1),(1,2)])
        for i in range(len(ds)):
            rec=ds[i]
            self.assertLess(rec['landmark'],int(rec['duration']))

    def test_future_suffix_is_ignored_even_if_nan_or_absorbing(self):
        for model in [DynamicDeepHitHistory(dropout=0),SurvTRACEHistory(dropout=0)]:
            model.eval()
            a=torch.zeros((2,30))
            b=a.clone();b[:,5:]=float('nan')
            ba=history_inputs(torch.tensor([0.,1.]),a,torch.tensor([5,5]))
            bb=history_inputs(torch.tensor([0.,1.]),b,torch.tensor([5,5]))
            key='log_pmf' if isinstance(model,DynamicDeepHitHistory) else 'hazard_logits'
            torch.testing.assert_close(model(ba)[key],model(bb)[key],rtol=0,atol=0)

    def test_observed_history_changes_both_models(self):
        for model in [DynamicDeepHitHistory(dropout=0),SurvTRACEHistory(dropout=0)]:
            model.eval()
            paths=torch.zeros((2,30),dtype=torch.long)
            paths[1,:5]=torch.tensor([1,2,0,1,2])
            batch=history_inputs(torch.zeros(2),paths,torch.tensor([5,5]))
            out=model(batch)
            key='log_pmf' if isinstance(model,DynamicDeepHitHistory) else 'hazard_logits'
            self.assertGreater(float((out[key][0]-out[key][1]).abs().max()),1e-6)

    def test_padding_batch_composition_invariance(self):
        for model in [DynamicDeepHitHistory(dropout=0),SurvTRACEHistory(dropout=0)]:
            model.eval()
            batch=history_inputs(torch.tensor([0.,1.]),torch.zeros((2,30)),torch.tensor([0,23]))
            single=history_inputs(torch.tensor([0.]),torch.zeros((1,30)),torch.tensor([0]))
            key='log_pmf' if isinstance(model,DynamicDeepHitHistory) else 'hazard_logits'
            torch.testing.assert_close(model(batch)[key][0],model(single)[key][0],rtol=1e-5,atol=1e-6)

    def test_attention_uses_past_and_latest_separately(self):
        model=DynamicDeepHitHistory(dropout=0).eval()
        batch=history_inputs(torch.zeros(2),torch.zeros((2,30)),torch.tensor([0,3]))
        a=model(batch)['attention']
        self.assertEqual(float(a[0].detach().sum()),0)
        self.assertAlmostEqual(float(a[1,:3].detach().sum()),1,places=6)
        self.assertEqual(float(a[1,3:].detach().sum()),0)

    def test_invalid_observed_dead_state_rejected(self):
        with self.assertRaisesRegex(ValueError,'live states'):
            history_inputs(torch.zeros(1),torch.tensor([[3]]),torch.tensor([1]))

    def test_dynamic_concordance_hand_case(self):
        s=np.array([[1,1,.7,.5],[1,.8,.6,.4]])
        c=dynamic_concordance(s,np.array([2,2]),np.array([1,0]),np.array([1,0]),np.array([1,2]))
        self.assertEqual(c,1)


class HistoryIntegrationTests(unittest.TestCase):
    def test_train_restore_predict_and_legacy_relabel(self):
        from extension_workflow import Experiment
        import sfsa_data_generation as base
        import sfsa_semi_markov as sm
        c=sm.generate_cohort_sm(base.WEIBULL_SCENARIO,n=100,seed=773)
        scratch=ROOT/'validation'/'temporary'/('history_test_'+uuid.uuid4().hex)
        scratch.mkdir(parents=True)
        with contextlib.nullcontext(scratch) as folder:
            e=Experiment('sm','Weibull',n=100,seed=24001,out_dir=folder)
            options={'batch_size':64,'validation_prefix_limit':64,'monitor':'loss',
                     'model_config':{'hidden':8,'attention_hidden':8,'head_hidden':8,'dropout':0}}
            st_options={'batch_size':64,'validation_prefix_limit':64,
                        'model_config':{'width':8,'heads':2,'layers':1,'intermediate':16,'dropout':0}}
            opts={'DynamicDeepHit':options,'SurvTransformer':st_options,'SurvTransformer-Discrete':st_options}
            labels=['DynamicDeepHIt-Lite','SurvTransformer-Lite',*opts]
            e.train(labels,epochs=1,patience=1,device='cpu',history_options=opts)
            restored=Experiment.restore(folder,device='cpu')
            for name in labels:
                for L in [0,23,24]:
                    prefix=np.zeros((2,L),dtype=int) if L else None
                    times=np.arange(L,31)
                    a=e.predict(name,[9.,11.],times,prefix=prefix,L=L)
                    b=restored.predict(name,[9.,11.],times,prefix=prefix,L=L)
                    np.testing.assert_allclose(a,b,atol=1e-7)
                    self.assertTrue(np.isfinite(a).all())
                    self.assertTrue(np.all(np.diff(a,axis=1)<=1e-6))
                    self.assertTrue(np.all((a>=0)&(a<=1.000001)))
                    np.testing.assert_allclose(a[:,0],1,atol=1e-6)
            # Simulate an archived v1 registry: its old name must become Lite.
            cfg=json.loads((Path(folder)/'run_config.json').read_text())
            cfg.pop('history_comparator_version')
            cfg['training']={'DynamicDeepHit':cfg['training']['DynamicDeepHIt-Lite']}
            import shutil
            shutil.copyfile(Path(folder)/'DynamicDeepHIt-Lite.pt',Path(folder)/'DynamicDeepHit.pt')
            (Path(folder)/'run_config.json').write_text(json.dumps(cfg))
            legacy=Experiment.restore(folder,device='cpu')
            self.assertIn('DynamicDeepHIt-Lite',legacy.models)
            self.assertNotIn('DynamicDeepHit',legacy.models)
            with self.assertRaisesRegex(ValueError,'Archived runs'):
                legacy.train(['DynamicDeepHit'],epochs=1)


if __name__=='__main__':
    unittest.main(verbosity=2)
