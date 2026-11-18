"""Boundary, conservation, absorption and simulation/reference checks."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
import sfsa_data_generation as base
import sfsa_semi_markov as sm
import reference_recursion as ref

def run_tests():
    states,init,k=ref.kernels(10.,'Weibull','semi')
    z=states.index((2,1))
    q23,d23,_=k[21];q24,d24,_=k[22]
    destinations=[i for i,(s,u) in enumerate(states) if s==2]
    assert q23[z,destinations].sum()>0
    assert q24[z,destinations].sum()==0
    np.testing.assert_allclose(d24[z]-d23[z],q23[z,destinations].sum())
    assert not np.allclose(k[17][1],k[18][1]),'t=20 regime change missing'
    for q,d,c in k:np.testing.assert_allclose(q.sum(1)+d+c,1,atol=1e-12)
    cohort=sm.generate_cohort_sm(base.WEIBULL_SCENARIO,n=10000,seed=27351)
    s=cohort['states']
    assert not np.any((s[:,22:-1]==2)&(s[:,23:]==2))
    for absorbing in [3,4]:
        assert not np.any((s[:,:-1]==absorbing)&(s[:,1:]!=absorbing))
    checks=[]
    for L in [22,23,24]:
        truth=ref.conditional_curves(k,L,[z])[0]
        _,sim=sm.ground_truth_km_landmark_sm(base.WEIBULL_SCENARIO,10.,2,1,L,seed=13457,n_matched=50000)
        error=float(abs(truth-sim[L+1:]).max())
        assert error<.015,(L,error)
        checks.append(dict(landmark=L,max_error_50k=error))
    result=dict(boundary_checks='passed',regime20='preserved',absorption='passed',probability_conservation='passed',simulation_checks=checks)
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':run_tests()
