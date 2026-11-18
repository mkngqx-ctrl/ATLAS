import run_classical_sampling as run
from pathlib import Path
import csv,json,html,hashlib,shutil,sys
import numpy as np,scipy,torch


def main():
    out=run.OUT
    verification=json.loads((out/'verification.json').read_text())
    assert verification['sampled_files_checked']==140
    summary=list(csv.DictReader((out/'sampled_summary.csv').open(encoding='utf-8-sig')))
    scores=list(csv.DictReader((out/'all_model_scores.csv').open(encoding='utf-8-sig')))
    assert len(summary)==52
    checks=[]
    for exp in ['Hard24','CR3']:
        for L in run.co.SPECS[exp][1]:
            kinds=['Markov','SemiMarkov'] if exp=='Hard24' else ['Markov']
            for kind in kinds:
                old=(run.CODE/f'hard24_markov_results/L{L}.npz' if kind=='Markov' else run.CODE/f'semimarkov_results/L{L}.npz') if exp=='Hard24' else run.CODE/f'pilot_results/CR3_markov_L{L}.npz'
                with np.load(old) as a,np.load(out/exp/f'L{L}/{kind}_exact.npz') as b:
                    p=a['prediction'];p=p[:,None,:] if p.ndim==2 else p
                    np.testing.assert_array_equal(a['patient_ids'],b['patient_ids']);np.testing.assert_allclose(p,b['prediction'],atol=1e-13,rtol=0)
                checks.append(dict(experiment=exp,landmark=L,model=kind,prior_exact_curves_reproduced=True))
    counts=[]
    for r in summary:
        atlas=[s for s in scores if s['experiment']==r['experiment'] and s['landmark']==r['landmark'] and s['target']==r['target'] and s['model'].startswith('ATLAS-')]
        matched=[s for s in atlas if s['method']==r['method']]
        counts.append(dict(experiment=r['experiment'],model=r['model'],method=r['method'],landmark=r['landmark'],target=r['target'],mean_MAE=float(r['mean_MAE']),ATLAS_configurations=len(atlas),ATLAS_means_beaten=sum(float(r['mean_MAE'])<float(s['mean_MAE']) for s in atlas),matched_sampler_ATLAS_configurations=len(matched),matched_sampler_ATLAS_means_beaten=sum(float(r['mean_MAE'])<float(s['mean_MAE']) for s in matched),best_ATLAS_mean_MAE=min(float(s['mean_MAE']) for s in atlas)))
    run.csvwrite(out/'atlas_comparison_counts.csv',counts)
    src=Path(__file__).resolve().parent
    for name in ['run_classical_sampling.py','test_classical_sampling.py','finalize_classical_sampling.py']:shutil.copy2(src/name,out/name)
    manifest=dict(M=10000,seeds=run.SEEDS,python=sys.version,numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,new_CR3_semimarkov_fit='15 fitted regressions including one common initial fit and two sets of seven state regressions; additive candidate chosen by observed validation transition log loss',n_sampled_prediction_files=140,n_exact_prediction_files=14,n_seed_level_metric_rows=260,n_summary_rows=52,total_trajectories=253400000,exact_regression_checks=checks,source_code_hashes={n:run.sha(out/n) for n in ['run_classical_sampling.py','test_classical_sampling.py','finalize_classical_sampling.py']})
    run.savejson(out/'reproducibility_manifest.json',manifest)
    lines=['IID AND SCRAMBLED SOBOL COMPARISON OF CLASSICAL TRANSITION MODELS','Saved October 28, 2026','',
        'M=10,000 trajectories per eligible patient, each model, sampler and sampling seed.',
        'Sampling seeds: 41001, 52001, 63001, 74001, 85001.',
        'Original ATLAS per-patient child-seed mappings and PyTorch Sobol implementation matched.',
        '253,400,000 trajectories across both experiments; 140 sampled prediction files.',
        'No neural models retrained. Prior classical fits retained. A missing CR3 semi-Markov fit was added using original train/validation splits.',
        'Hard24 has survival only. CR3 has total survival and both cause-specific CIFs.',
        'All model probabilities were sampled in original state order, with original event-first KM/AJ conventions.',
        'Classical means average five sampling-seed MAEs on a single fitted model. ATLAS means average sampling seeds within fit, then nine Hard24 or five CR3 training fits.',
        'Sampling-seed SD is numerical variability, not training-run or patient uncertainty.',
        'All comparisons are descriptive; no new statistical superiority tests were performed.',
        'M=10,000 is not a power of two; power-of-two Sobol balance guarantees do not apply.',
        'A Hard24 exact-curve array-view storage bug in this new pipeline was corrected; sampled predictions and truth-based errors were unaffected. See diagnostic_correction.json.',
        'Rare float32 Sobol values rounded to 1.0 were handled as censored terminal outcomes to match archived CR3 code. Affected censor-count records were replayed; curves and MAEs did not change. See sobol_boundary_audit.json.',
        'Reconstruction of risk/death/censor accounting, KM/AJ curves, patient MAEs and all saved curve constraints passed.',
        'Controlled tests reproduce archived ATLAS sampler curves and uniforms, and exact propagation agrees with independent scalar recursion.',
        'Previously fitted exact curves reproduced at all available experiment/model/landmark combinations (tolerance 1e-13).',
        '', 'FILES',
        'comparison_report.html: readable full model comparison tables.',
        'sampled_summary.csv: 52 model/sampler/landmark/target summaries with sampling SD and exact-reference diagnostics.',
        'Hard24_replicate_metrics.csv and CR3_replicate_metrics.csv: 260 seed-level metric rows.',
        'all_model_scores.csv: every original fitted comparator, ATLAS sampler, classical exact fit and new sampled fit; oracle controls identified separately.',
        'all_pairwise_comparisons.csv: every requested sampled classical result against all available model scores for the same target and landmark.',
        'atlas_comparison_counts.csv: counts of ATLAS means beaten, including matched samplers.',
        'Hard24/ and CR3/: patient IDs, truth, sampled curves, patient errors, risks, deaths, censor counts and effective seeds.',
        'CR3_semiMarkov_fit/: new model specification, training records, validation selection and coefficients.',
        'sampling_plan.json, verification.json and reproducibility_manifest.json: methods and reproducibility records.',
        '', 'MEAN ERRORS']
    sections=[]
    for exp,targets in [('Hard24',['Survival']),('CR3',['Survival','CIF_Death_A','CIF_Death_B'])]:
        Ls=run.co.SPECS[exp][1]
        for target in targets:
            lines += ['',f'{exp} {target}: landmarks '+','.join(map(str,Ls))]
            for model in ['Markov','SemiMarkov']:
                for method in ['iid','sobol']:
                    rr=[next(r for r in summary if r['experiment']==exp and r['target']==target and r['model']==model and r['method']==method and int(r['landmark'])==L) for L in Ls]
                    lines.append(model+' '+method+' '+', '.join(f"{float(r['mean_MAE']):.9f}" for r in rr))
            rows=[]
            for model,method in sorted({(s['model'],s['method']) for s in scores if s['experiment']==exp and s['target']==target}):
                ss=[next(s for s in scores if s['experiment']==exp and s['target']==target and s['model']==model and s['method']==method and int(s['landmark'])==L) for L in Ls]
                new=model in ['Markov','SemiMarkov'] and method in ['iid','sobol']
                rows.append('<tr'+(' class="new"' if new else '')+'><td>'+html.escape(model)+'</td><td>'+html.escape(method)+'</td>'+''.join(f'<td>{float(s["mean_MAE"]):.6f}</td>' for s in ss)+'<td>'+html.escape(ss[0]['replicate_type'])+'</td></tr>')
            sections.append(f'<h2>{exp} {target}</h2><table><thead><tr><th>Model</th><th>Method</th>'+''.join(f'<th>L={L}</th>' for L in Ls)+'<th>Replicates represented</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>')
    (out/'RESULTS.txt').write_text('\n'.join(lines),encoding='utf-8')
    body='''<!doctype html><html><head><meta charset="utf-8"><title>ATLAS matched sampling comparison</title><style>body{font:16px/1.5 Arial,sans-serif;margin:35px auto;max-width:1250px;color:#111}table{border-collapse:collapse;width:100%;margin:15px 0 40px}th,td{padding:7px 10px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child,td:nth-child(2),td:last-child{text-align:left}th{background:#e9edf2}.new{background:#e8f3ec}h2{margin-top:40px}code{font-size:14px}@media print{h2{break-before:page}table{font-size:10px}}</style></head><body><h1>Matched IID and scrambled Sobol comparison</h1><p>10,000 trajectories per patient. Sampling seeds 41001, 52001, 63001, 74001 and 85001, using ATLAS child-seed rules and archived sampling conventions. Lower full-curve MAE is better.</p><p>New sampled classical results are shaded green. Classical sampling SD and ATLAS between-training-run SD describe different sources of variation; full details and sampling-error diagnostics are in the CSV files. Exact propagation and oracle sampling controls are shown separately. The CR3 semi-Markov counterpart was newly fitted and validation-selected; other classical fits are unchanged. These are descriptive comparisons, conditional on fitted models and the original test subsets.</p>'''
    (out/'comparison_report.html').write_text(body+''.join(sections)+'</body></html>',encoding='utf-8')
    print('\n'.join(lines[-30:]))
    print('Final artifacts saved to',out)


if __name__=='__main__':main()
