from pathlib import Path
import unittest,tempfile,numpy as np,torch,contextlib,uuid,os
from extension_workflow import Experiment,SPECS

class ExtensionIntegrationTests(unittest.TestCase):
    def test_sm_training_scoring_and_restore(self):self.check_kind('sm')
    def test_cr_training_scoring_and_restore(self):self.check_kind('cr')
    def test_relu_sm(self):self.check_kind('sm','ReLU')
    def test_relu_cr(self):self.check_kind('cr','ReLU')
    def check_kind(self,kind,scenario='Weibull'):
        torch.set_num_threads(2)
        with contextlib.nullcontext(str(Path(os.environ.get('GLAZIER_TEST_OUTPUT','extension_test_runs'))/uuid.uuid4().hex)) as out:
            e=Experiment(kind,scenario=scenario,n=1000,out_dir=out)
            e.train(epochs=2,patience=2,device='cpu')
            before={label:e.predict(label,np.array([70. if scenario=='ReLU' else 10.]),np.arange(31),samples=16,seed=42) for label in e.models}
            restored=Experiment.restore(out,device='cpu')
            self.assertEqual(set(restored.models),set(SPECS))
            for label in restored.models:
                after=restored.predict(label,np.array([70. if scenario=='ReLU' else 10.]),np.arange(31),samples=16,seed=42)
                np.testing.assert_allclose(before[label],after,atol=1e-6)
            df=restored.evaluate(n_eval=12,samples=16,gt_samples=128,landmarks=(10,20),tag='smoke')
            self.assertTrue(np.isfinite(df['MAE']).all())
            self.assertTrue((Path(out)/'smoke_curves.npz').exists())
            if kind=='sm':
                oracle=df[df['model']=='Augmented-state oracle'];self.assertTrue((oracle['MAE']==0).all())
                self.assertEqual(set(oracle['landmark']),{10,20})
            else:self.assertEqual(set(df['component']),{'survival','disease','other'})
    def test_colab_notebook_syntax(self):
        import json,ast
        for p in Path(__file__).parent.glob('Glazier_*Colab.ipynb'):
            n=json.loads(p.read_text())
            for c in n['cells']:
                if c['cell_type']=='code':
                    source=''.join(c['source'])
                    ast.parse('\n'.join(line for line in source.splitlines() if not line.startswith(('%','!'))))

if __name__=='__main__':unittest.main(verbosity=2)

