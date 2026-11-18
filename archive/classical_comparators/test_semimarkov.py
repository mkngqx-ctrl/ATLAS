"""Verification of duration/time indexing, augmented propagation and saved scores."""
import importlib.util,json
from pathlib import Path
import fit_semimarkov as sm
import numpy as np
from scipy import sparse


def main():
    import investigate_markov as dense
    rng=np.random.default_rng(42)
    X=np.column_stack([np.ones(150),rng.normal(size=(150,2))])
    yy=rng.integers(0,3,150)
    fitted=sm.fit(sparse.csr_matrix(X),yy,ridge=1.)
    counts=np.eye(3)[yy]
    labels,beta=dense.fit(X,counts,ridge=1.)
    np.testing.assert_allclose(sm.predict(sparse.csr_matrix(X),fitted)[:,:3],dense.probabilities(X,beta),atol=1e-6,rtol=0)
    print('PASS: sparse scaled optimizer matches independent dense Newton implementation, tolerance 1e-6')
    # Verify (source time, source duration, next state) extraction explicitly.
    states=np.array([[0,1,1,2,2]+[3]*25,[0,0,1,1,4]+[4]*25])
    ids,s,u,t,y=sm.transition_rows(dict(states=states),np.array([0,1]))
    actual=list(zip(ids.tolist(),s.tolist(),u.tolist(),t.tolist(),y.tolist()))
    assert actual==[(0,0,1,1,1),(1,0,1,1,0),(0,1,1,2,1),(1,0,2,2,1),(0,1,2,3,2),(1,1,1,3,1),(0,2,1,4,2),(1,1,2,4,4),(0,2,2,5,3)]
    print('PASS: source-time indexing, run lengths, terminal outcomes, no absorbing repeats')
    # Test-only known generator; never used in fitting or model selection.
    path=Path(__file__).resolve().parents[1]/'reference_code/single_covariate_reference.py'
    spec=importlib.util.spec_from_file_location('semi_test_reference',path)
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    xs=np.array([9.,11.]);kernels=[ref.kernels(float(x),'Weibull','semi') for x in xs]
    old_design,old_predict=sm.design,sm.predict
    def known_predict(X,model):
        if model==3:return np.tile([2/3,1/4,1/12,0,0],(X.shape[0],1))
        X=X.toarray();out=np.zeros((len(X),5))
        for i,(x,u,t) in enumerate(X):
            states,initial,k=kernels[int(np.flatnonzero(xs==x)[0])]
            q,d,c=k[int(t)-1];z=states.index((model,int(u)))
            for j in range(3):out[i,j]=q[z,[a for a,state in enumerate(states) if state[0]==j]].sum()
            out[i,3]=d[z];out[i,4]=c[z]
        return out
    try:
        sm.design=lambda x,u,t,rich:sparse.csr_matrix(np.column_stack([x,u,t]))
        sm.predict=known_predict
        for L in [0,20,23,24]:
            ss=np.array([1,2]);uu=np.array([3,2])
            actual=sm.propagate(xs,3,[0,1,2],False,ss,uu,L)
            expected=[]
            for i,(states,initial,k) in enumerate(kernels):
                expected.append(ref.forward(initial,k)[1][1:] if not L else ref.conditional_curves(k,L,[states.index((int(ss[i]),int(uu[i])))])[0])
            np.testing.assert_allclose(actual,np.array(expected),atol=1e-13,rtol=0)
    finally:sm.design,sm.predict=old_design,old_predict
    print('PASS: augmented-state propagation matches independent reference at all four landmarks, tolerance 1e-13')
    if (sm.OUT/'metrics.csv').exists():
        import csv
        rows=list(csv.DictReader((sm.OUT/'metrics.csv').open()))
        for r in rows:
            with np.load(sm.OUT/f"L{r['landmark']}.npz") as z:
                sm.c.check_curves(z['prediction'][:,None,:])
                errors=np.abs(z['prediction']-z['truth']).mean(1)
                np.testing.assert_allclose(errors,z['patient_mae'],atol=1e-15,rtol=0)
                np.testing.assert_allclose(errors.mean(),float(r['MAE']),atol=1e-15,rtol=0)
        print('PASS: saved curves and patient/aggregate errors recomputed')


if __name__=='__main__':main()
