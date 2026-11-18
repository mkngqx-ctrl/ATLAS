"""Small synthetic cache tests; no cohort simulation or model training."""
from pathlib import Path
import os, sys, datetime, json, unittest, uuid, contextlib, io
from types import SimpleNamespace
from unittest.mock import patch
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
if not hasattr(datetime, 'UTC'): datetime.UTC = datetime.timezone.utc
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT/'source')]
for parent in [ROOT, *ROOT.parents]:
    if (parent/'work/gpu_support').exists():
        sys.path.insert(0, str(parent/'work/gpu_support')); break
import numpy as np
import torch
import hard_rule_workflow as workflow
from reuse_results import plan_reuse, apply_reuse, sha


class SavedPredictionTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT/'validation/temporary'/('cache_'+uuid.uuid4().hex)
        self.source = self.root/'previous'
        ckpt = self.source/'checkpoints'; ckpt.mkdir(parents=True)
        self.out = self.root/'new/evaluation'
        cohort = dict(states=np.zeros((2,30),dtype=int),covariate=np.array([.2,.7]),
                      test_index=np.array([0,1]))
        np.savez_compressed(ckpt/'cohort.npz',**cohort)
        model = torch.nn.Linear(1,1)
        meta = {'device':'cpu'}
        torch.save(dict(state_dict=model.state_dict(),class_name='Linear',meta=meta),ckpt/'DynamicDeepHit.pt')
        (ckpt/'run_config.json').write_text(json.dumps(dict(kind='sm',scenario='Weibull',cohort_seed=24001,
                                                         training={'DynamicDeepHit':{}})))
        self.e = SimpleNamespace(kind='sm',scenario=SimpleNamespace(name='Weibull'),seed=24001,
                                 out=ckpt,cohort=cohort,models={'DynamicDeepHIt-Lite':(model,meta,{})})
        self.settings = dict(M=16,n_eval=2,seeds=[11,22],landmarks=[0,24],batch_size=8,
                             rule=workflow.RULE,importance_death_tilt=2.,source_hashes=workflow.hashes(),
                             checkpoint_hashes={p.name:sha(p) for p in ckpt.iterdir() if p.suffix in ('.pt','.npz')})
        ids = np.random.default_rng(24003).permutation(cohort['test_index'])
        for L in self.settings['landmarks']:
            folder = self.source/f'evaluation/L{L}'; folder.mkdir(parents=True)
            np.savez_compressed(folder/'reference.npz',patient_ids=ids,times=np.arange(L+1,31),truth=self.reference(self.e,ids,L))
            for name in ['DynamicDeepHit_direct_-1','Oracle sampler_iid_11','Oracle sampler_iid_22']:
                np.savez_compressed(folder/(name+'.npz'),prediction=self.reference(self.e,ids,L),seconds=.25,min_ESS=16.)
        (self.source/'evaluation/evaluation_settings.json').write_text(json.dumps(self.settings))

    @staticmethod
    def reference(e,ids,L):
        return np.broadcast_to(np.linspace(.9,.1,30-L),(len(ids),30-L)).copy()

    def plan(self,**kwargs):
        return plan_reuse(self.e,self.source,self.out,dict(self.settings,**kwargs),self.reference)

    def test_label_mapping_and_exact_copy(self):
        before = {str(p):sha(p) for p in self.source.rglob('*') if p.is_file()}
        plan = self.plan()
        self.assertEqual(plan['saved_predictions'],6)
        self.assertEqual(plan['label_mapping'],{'DynamicDeepHit':'DynamicDeepHIt-Lite'})
        with contextlib.redirect_stdout(io.StringIO()): apply_reuse(plan,self.out)
        self.assertTrue((self.out/'L0/DynamicDeepHIt-Lite_direct_-1.npz').exists())
        self.assertFalse((self.out/'L0/DynamicDeepHit_direct_-1.npz').exists())
        for item in plan['files']: self.assertEqual(sha(item['destination']),item['sha256'])
        self.assertEqual(before,{str(p):sha(p) for p in self.source.rglob('*') if p.is_file()})

    def test_budget_mismatch_is_read_only(self):
        with self.assertRaisesRegex(ValueError,'M differs'): self.plan(M=32)
        self.assertFalse(self.out.exists())

    def test_changed_weights_rejected(self):
        with torch.no_grad(): self.e.models['DynamicDeepHIt-Lite'][0].weight.add_(1.)
        with self.assertRaisesRegex(ValueError,'weights differ'): self.plan()

    def test_changed_reference_rejected(self):
        with self.assertRaises(AssertionError):
            plan_reuse(self.e,self.source,self.out,self.settings,lambda e,i,L: self.reference(e,i,L)+.01)

    def test_prediction_shape_rejected(self):
        file = self.source/'evaluation/L0/DynamicDeepHit_direct_-1.npz'
        np.savez_compressed(file,prediction=np.ones((1,30)),seconds=.25,min_ESS=16)
        with self.assertRaisesRegex(ValueError,'Invalid saved prediction'): self.plan()

    def test_destination_conflict_rejected(self):
        file = self.out/'L0/DynamicDeepHIt-Lite_direct_-1.npz'
        file.parent.mkdir(parents=True); file.write_bytes(b'different')
        with self.assertRaisesRegex(ValueError,'different cached file'): self.plan()

    def test_original_destination_rejected(self):
        with self.assertRaisesRegex(ValueError,'outside the previous'):
            plan_reuse(self.e,self.source,self.source/'evaluation',self.settings,self.reference)

    def test_only_new_models_predicted_and_resume_uses_cache(self):
        # Deterministic stand-ins test dispatch/cache behavior, not model quality.
        names=list(workflow.HISTORY_SPECS)
        self.e.models.update({name:(None,{},None) for name in names})
        calls=[]
        def predict(name,x,grid,prefix=None,L=None):
            self.assertIn(name,names)
            calls.append((name,L))
            return self.reference(self.e,np.arange(len(x)),L or 0)
        self.e.predict=predict
        args={k:self.settings[k] for k in ('M','n_eval','seeds','landmarks','batch_size')}
        with patch.object(workflow,'references',side_effect=self.reference), \
             patch.object(workflow,'sample_curve',side_effect=AssertionError('Unexpected rollout')), \
             patch.object(workflow.extras,'ground_truth_km_sm_for_validation_set',side_effect=AssertionError('Unexpected oracle')), \
             patch.object(workflow.sm,'ground_truth_km_landmark_sm_for_validation_set',side_effect=AssertionError('Unexpected oracle')), \
             contextlib.redirect_stdout(io.StringIO()):
            rows,summary=workflow.evaluate(self.e,self.out,reuse_from=self.source,**args)
            self.assertEqual(len(calls),6)
            self.assertEqual(int(rows.reused_prediction.sum()),6)
            self.assertEqual(len(rows),12)
            self.assertTrue(set(names).issubset(summary.model))
            workflow.evaluate(self.e,self.out,reuse_from=self.source,**args)
            self.assertEqual(len(calls),6)


if __name__=='__main__': unittest.main(verbosity=2)
