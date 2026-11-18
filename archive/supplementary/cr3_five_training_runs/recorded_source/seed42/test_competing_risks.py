"""Numerical/reference checks independent of fitted benchmark performance."""
from pathlib import Path
import sys,os,datetime,itertools,unittest
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
if not hasattr(datetime,'UTC'):datetime.UTC=datetime.timezone.utc
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'source')]
for parent in [ROOT,*ROOT.parents]:
    if (parent/'work/gpu_support').exists():sys.path.insert(0,str(parent/'work/gpu_support'));break
import numpy as np
import torch
from sfsa_cr import *
from samplers_cr import weighted_aj,sample_curves,Oracle,METHODS
from history_models import history_inputs,DynamicDeepHitHistory,SurvTRACEHistory,pmf_curves,survtrace_objective
from legacy_models import GlazierModel,GlazierTransformerModel
from history_training import PrefixDataset
torch.set_num_threads(2)


class GeneratorTests(unittest.TestCase):
    def test_transition_rows_graph_and_death_preference(self):
        x=np.array(list(itertools.product(np.linspace(8,13,9),np.linspace(30,90,9),[0,.5,1.])))
        q=transitions(x);np.testing.assert_allclose(q.sum(-1),1.,atol=1e-14)
        self.assertGreaterEqual(q.min(),0.)
        self.assertTrue((q[:,AABB,DA]>q[:,AABB,DB]).all())
        np.testing.assert_array_equal(q[:,[H,A,B,AB],7:9],0)
        np.testing.assert_array_equal(q[:,AA,DB],0);np.testing.assert_array_equal(q[:,BB,DA],0)
        np.testing.assert_allclose(initial(x).sum(1),1.)

    def test_drug_transfers_only_self_mass_to_specified_recovery(self):
        lo=transitions([[11,70,0]])[0];hi=transitions([[11,70,1]])[0]
        expected=np.zeros_like(lo)
        expected[BB,B]=.35;expected[BB,BB]=-.35
        expected[AABB,AB]=.30;expected[AABB,AABB]=-.30
        np.testing.assert_allclose(hi-lo,expected,atol=1e-14)

    def test_covariate_response_shapes(self):
        ca,age,_=effects([[8,30,0],[10.5,50,0],[13,90,0]])
        np.testing.assert_allclose(ca,[0,.5**2.5,1]);np.testing.assert_allclose(age,[0,0,1])

    def test_cohort_endpoints_absorption_and_split(self):
        c=generate(1000,7)
        for i,(d,k) in enumerate(zip(c['duration'],c['cause'])):
            self.assertTrue((c['states'][i,:d-1]<7).all())
            if k:self.assertEqual(c['states'][i,d-1],6+k)
            if d<30 or k:self.assertTrue((c['states'][i,d-1:]==c['states'][i,d-1]).all())
        groups=[set(c[k]) for k in ('train_index','val_index','test_index')]
        self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
        self.assertEqual(len(set.union(*groups)),1000)
        self.assertGreater(np.sum(c['cause']==1),0);self.assertGreater(np.sum(c['cause']==2),0)

    def test_forward_reference_against_path_enumeration(self):
        x=np.array([[11,70,.4]]);q=np.eye(9);q[:7]=transitions(x)[0];p0=initial(x)[0]
        expected=np.zeros((3,3))
        for path in itertools.product(range(9),repeat=3):
            p=p0[path[0]]*q[path[0],path[1]]*q[path[1],path[2]]
            for j,s in enumerate(path):expected[0 if s<7 else s-6,j]+=p
        truth=reference_curves(x,scenario=Scenario(horizon=3))[0]
        np.testing.assert_allclose(truth,expected,atol=1e-13);validate_curves(truth)

    def test_observed_aj_equals_independent_censor_biological_reference(self):
        x=np.array([[12,85,.2]]);cp=.13;scenario=Scenario(censor_probability=cp)
        p=observed(initial(x),cp)[0];q=observed(transitions(x),cp)[0]
        live=None;s=1.;f=np.zeros(2);curves=[]
        for t in range(30):
            risk=1. if t==0 else live.sum()
            mass=p if t==0 else live@q
            h=mass[7:9]/risk;f+=s*h;s*=1-h.sum();curves.append([s,*f]);live=mass[:7]
        np.testing.assert_allclose(np.array(curves).T,reference_curves(x,scenario=scenario)[0],atol=1e-12)

    def test_landmark_reference_uses_current_state(self):
        x=np.array([[11,70,.4],[11,70,.4]])
        pre=np.array([[A,AB,AABB],[B,BB,BB]])
        truth=reference_curves(x,pre,3)
        self.assertGreater(abs(truth[0]-truth[1]).max(),.01)
        validate_curves(truth)


class SamplerTests(unittest.TestCase):
    def test_event_first_tie_and_empty_risk(self):
        result=weighted_aj([1,1,2,3],[1,0,2,0],np.ones(4),H=4)
        np.testing.assert_allclose(result,[[.75,.375,.375,.375],[.25,.25,.25,.25],[0,.375,.375,.375]])

    def test_importance_weighted_count_example(self):
        result=weighted_aj([1,1,2],[1,2,0],[2,1,1],H=2)
        np.testing.assert_allclose(result,[[.25,.25],[.5,.5],[.25,.25]])

    def test_all_seven_oracle_samplers_against_exact_reference(self):
        x=np.array([12,80,.4]);scenario=Scenario(horizon=12)
        meta=dict(horizon=12,x_mean=[0,0,0],x_sd=[1,1,1])
        prefix=[AB,AABB];truth=reference_curves(x,np.array([prefix]),2,scenario)[0]
        for method in METHODS:
            with self.subTest(method=method):
                curve,diag=sample_curves(Oracle(scenario),meta,x,M=4096,method=method,seed=72,prefix=prefix)
                self.assertLess(abs(curve-truth).max(),.04);validate_curves(curve)
                self.assertGreater(diag['min_ESS'],0)


class ModelTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(7)

    def test_joint_pmf_tail_and_landmark(self):
        # A masses .1,.2; B masses .05,.15; shared tail .5.
        logp=torch.tensor([[.1,.2,.05,.15,.5]]).log()
        curve=pmf_curves(logp,torch.tensor([1]),2,2).numpy()
        np.testing.assert_allclose(curve[0,:,2],[.5/.85,.2/.85,.15/.85],atol=1e-7)
        np.testing.assert_allclose(curve[0,:,1],[1,0,0],atol=1e-7)

    def test_both_survtrace_likelihoods_by_hand(self):
        batch=history_inputs(torch.zeros(1,3),torch.zeros(1,2),torch.tensor([0]),2)
        a=torch.tensor([[[.1,.2],[.3,.4]]]);output=dict(hazard=a,mortality_logits=torch.zeros(1),duration_prediction=torch.zeros(1))
        for likelihood in ('pch','discrete'):
            model=SurvTRACEHistory(horizon=2,width=4,heads=1,layers=1,intermediate=4,likelihood=likelihood)
            value,_=survtrace_objective(model,output,batch,torch.tensor([2]),torch.tensor([1]),mortality_weight=0,duration_weight=0)
            expected=1-np.log(.2) if likelihood=='pch' else .4-np.log((1-np.exp(-.6))*.2/.6)
            self.assertAlmostEqual(float(value),expected,places=6)

    def test_future_mask_and_full_history_sensitivity(self):
        states=torch.tensor([[1,3,6,7,8],[1,2,6,9,7]])
        batch=history_inputs(torch.zeros(2,3),states,torch.tensor([3,3]),5)
        suffix=states.clone();suffix[:,3:]=99
        other=history_inputs(torch.zeros(2,3),suffix,torch.tensor([3,3]),5)
        models=[DynamicDeepHitHistory(horizon=5,hidden=8,attention_hidden=8,head_hidden=8,dropout=0),
                SurvTRACEHistory(horizon=5,width=4,heads=1,layers=1,intermediate=8,dropout=0)]
        for model in models:
            model.eval();key='log_pmf' if isinstance(model,DynamicDeepHitHistory) else 'hazard'
            a=model(batch)[key];b=model(other)[key]
            torch.testing.assert_close(a,b)
            self.assertGreater(float((a[0]-a[1]).abs().max()),1e-8)

    def test_glazier_causality(self):
        a=torch.tensor([[10,1,3,6]]);b=torch.tensor([[10,1,5,2]]);x=torch.zeros(1,3)
        models=[GlazierModel(),GlazierTransformerModel(dropout=0)]
        for model in models:
            model.eval()
            torch.testing.assert_close(model.forward_teacher_forced(a,x)[:,:2],model.forward_teacher_forced(b,x)[:,:2])

    def test_prefix_dataset_multicovariate_causes(self):
        c=generate(100,91,Scenario(horizon=6));idx=c['train_index'];x=c['covariates'][idx]
        ds=PrefixDataset(x,c['states'][idx],c['duration'][idx],c['cause'][idx],x.mean(0),x.std(0),horizon=6)
        self.assertEqual(len(ds),int(c['duration'][idx].sum()))
        self.assertEqual(ds[0]['x'].shape,(3,))


if __name__=='__main__':unittest.main(verbosity=2)
