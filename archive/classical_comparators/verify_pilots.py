"""Read-only verification of the existing classical-comparator pilots.

Reference-generator probabilities are used only in the propagation unit test.
They are never used to train or select either fitted comparator.
"""
import hashlib
import importlib.util
import sys
from pathlib import Path
import numpy as np
import investigate_cox as c
import investigate_markov as m


def main():
    c.self_test()
    print('PASS: Cox tied-event fit, numerical gradient/Hessian, run length')
    rng=np.random.default_rng(123)
    x=np.column_stack([np.ones(20),rng.normal(size=(20,2))])
    counts=rng.integers(0,5,(20,3));beta=rng.normal(size=(3,2))*.1
    _,g,h=m.objective(beta,x,counts,.01);eps=1e-5
    for j in range(beta.size):
        delta=np.zeros_like(beta);delta.flat[j]=eps
        lp,gp,_=m.objective(beta+delta,x,counts,.01)
        lm,gm,_=m.objective(beta-delta,x,counts,.01)
        np.testing.assert_allclose(g.flat[j],(lp-lm)/(2*eps),atol=1e-7)
        np.testing.assert_allclose(h[:,j],((gp-gm)/(2*eps)).ravel(),atol=1e-7)
    print('PASS: multinomial numerical gradient/Hessian, tolerance 1e-7')
    ref=Path(__file__).resolve().parents[1]/'reference_code/competing_risks_generator_reference.py'
    spec=importlib.util.spec_from_file_location('sfsa_test_reference',ref)
    sfsa=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=sfsa;spec.loader.exec_module(sfsa)
    xx=np.array([[9,40,.2],[12,80,.8]])
    original=m.expanded
    try:
        m.expanded=lambda x,s:sfsa.observed(sfsa.initial(xx) if s==7 else sfsa.transitions(xx)[:,s],.008)
        for L in [0,10,20]:
            state=np.array([4,6]);prefix=np.tile(state[:,None],(1,L))
            actual=m.propagate(xx,list(range(8)),state,L)
            expected=sfsa.reference_curves(xx,prefix,L)
            np.testing.assert_allclose(actual,expected,atol=1e-13,rtol=0)
    finally:
        m.expanded=original
    print('PASS: event-first propagation versus archived reference at L=0,10,20; tolerance 1e-13')
    nfiles=0
    for name,(folder,landmarks,nlive,expected_hash) in c.SPECS.items():
        source=c.BASE/folder
        path=source/'checkpoints/cohort.npz'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected_hash
        with np.load(path,allow_pickle=False) as z:cohort={k:z[k] for k in z.files}
        tr,va,te=[set(cohort[k].tolist()) for k in ['train_index','val_index','test_index']]
        assert not (tr&va or tr&te or va&te)
        assert len(tr|va|te)==50000
        for L in landmarks:
            files=[c.OUT/f'{name}_L{L}.npz']
            if name=='CR3':files.append(c.OUT/f'CR3_markov_L{L}.npz')
            with np.load(source/f'evaluation/L{L}/reference.npz') as z:
                ids=z['patient_ids'];times=z['times'];truth=z['truth']
            if truth.ndim==2:truth=truth[:,None,:]
            assert set(ids.tolist())<=te
            assert L==0 or (cohort['states'][ids,L-1]<nlive).all()
            for file in files:
                with np.load(file) as z:
                    np.testing.assert_array_equal(z['patient_ids'],ids)
                    np.testing.assert_array_equal(z['times'],times)
                    np.testing.assert_array_equal(z['truth'],truth)
                    c.check_curves(z['prediction'])
                    np.testing.assert_allclose(z['patient_mae'],np.abs(z['prediction']-truth).mean(-1),atol=1e-15,rtol=0)
                nfiles+=1
    print(f'PASS: cohort hashes; disjoint splits; test eligibility and IDs; {nfiles} saved prediction files; curve constraints; recomputed patient MAEs')


if __name__=='__main__':main()
