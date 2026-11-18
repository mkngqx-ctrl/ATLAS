"""Patient-paired fixed-fit bootstrap versus all archived fitted procedures."""
import csv,json
from pathlib import Path
import fit_semimarkov as sm
from workspace_paths import relocated
import numpy as np


def main():
    base=Path(__file__).resolve().parents[1]
    with (base/'supplementary/hard24_nine_training_runs/source_settings.json').open() as f:sources=json.load(f)
    summary=list(csv.DictReader((base/'supplementary/hard24_nine_training_runs/hard24_verified_summary.csv').open()))
    configs=sorted({(r['model'],r['method']) for r in summary})
    bykey={(r['model'],r['method'],int(r['landmark'])):float(r['mean']) for r in summary}
    rng=np.random.default_rng(20261028);results=[]
    for L in [0,20,23,24]:
        with np.load(sm.OUT/f'L{L}.npz') as z:ids=z['patient_ids'];truth=z['truth'];semi=z['patient_mae']
        matrix=[];keys=[]
        for model,method in configs+[('CPH','direct')]:
            errors=[]
            use_sources=sources if model!='CPH' else sources[:1]
            for source in use_sources:
                root=relocated(source['root'])/f'evaluation/L{L}'
                with np.load(root/'reference.npz') as z:
                    np.testing.assert_array_equal(z['patient_ids'],ids)
                    np.testing.assert_allclose(z['truth'],truth,atol=1e-15,rtol=0)
                label=model.replace('ATLAS-','Glazier-')
                candidates=[label]
                if model=='DynamicDeepHit-Lite':candidates=['DynamicDeepHIt-Lite','DynamicDeepHit-Lite']
                files=[]
                for label in candidates:
                    files=sorted(root.glob(f'{label}_{method}_*.npz'))
                    if files:break
                assert len(files)==(5 if method!='direct' else 1),(root,model,method,len(files))
                run=[]
                for file in files:
                    with np.load(file) as z:p=z['prediction']
                    assert p.shape==truth.shape
                    run.append(np.abs(p-truth).mean(1))
                errors.append(np.mean(run,axis=0))
            patient_error=np.mean(errors,axis=0)
            if model!='CPH':np.testing.assert_allclose(patient_error.mean(),bykey[model,method,L],atol=1e-12,rtol=0)
            matrix.append(semi-patient_error);keys.append((model,method))
        # Include the other deterministic classical fits using saved patient errors.
        for folder,template,label in [('hard24_markov_results','L{L}.npz','Fitted-Markov'),('pilot_results','Hard24_L{L}.npz','Landmark-CPH-supermodel')]:
            with np.load(sm.OUT.parent/folder/template.format(L=L)) as z:
                np.testing.assert_array_equal(z['patient_ids'],ids)
                error=z['patient_mae'].reshape(len(ids),-1).mean(1)
            matrix.append(semi-error);keys.append((label,'exact'))
        differences=np.column_stack(matrix)
        weights=rng.multinomial(len(ids),np.ones(len(ids))/len(ids),size=5000)/len(ids)
        boot=weights@differences
        for j,(model,method) in enumerate(keys):
            low,high=np.quantile(boot[:,j],[.025,.975])
            results.append(dict(landmark=L,model=model,method=method,n_patients=len(ids),difference_semi_minus_comparator=float(differences[:,j].mean()),CI_lower=float(low),CI_upper=float(high),resamples=5000))
        print('Bootstrap completed L=',L,flush=True)
    sm.save_csv(sm.OUT/'paired_bootstrap.csv',results)
    (sm.OUT/'bootstrap_settings.json').write_text(json.dumps(dict(seed=20261028,resamples=5000,
        estimand='semi-Markov patient MAE minus comparator patient MAE averaged over saved sampling seeds then training runs',
        intervals='pointwise paired percentile bootstrap, 95%; no multiplicity adjustment',
        limitations='conditional on fitted models, selected specification and fixed cohort; no training-sample or new-dataset uncertainty'),indent=2))


if __name__=='__main__':main()
