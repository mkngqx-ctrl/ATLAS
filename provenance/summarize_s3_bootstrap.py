from pathlib import Path
import csv,json,hashlib,html,shutil
import numpy as np

OUT=Path(__file__).resolve().parent/'S3_paired_patient_bootstrap_2026-10-30'
def readcsv(p):return list(csv.DictReader(p.open(encoding='utf8')))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
heads=readcsv(OUT/'headline_comparisons.csv');pairs=readcsv(OUT/'paired_comparisons.csv');summary=json.loads((OUT/'summary_counts.json').read_text())
# Independent explicit patient-list check of weighted bootstrap arithmetic.
explicit_checks=0
for exp in ['Hard24','CR3']:
 with np.load(OUT/f'{exp}_patient_resamples.npz') as z:base=z['baseline_patient_ids'];weights=z['multiplicities']
 with np.load(OUT/f'{exp}_patient_errors.npz') as z:stored={k:z[k] for k in z.files}
 with np.load(OUT/f'{exp}_bootstrap_model_means.npz') as z:boot={k:z[k] for k in z.files}
 Ls=[0,20,23,24] if exp=='Hard24' else [0,10,20]
 for L in Ls:
  ids=stored[f'patient_ids_L{L}'];e=stored[f'patient_MAE_L{L}'];index={int(pid):i for i,pid in enumerate(ids)}
  for b in [0,2499,4999]:
   drawn=np.repeat(base,weights[b].astype(int))
   row_indices=np.array([index[int(pid)] for pid in drawn if int(pid) in index])
   direct=e[row_indices].mean(0)
   np.testing.assert_allclose(direct,boot[f'model_MAE_L{L}'][b],atol=1e-14,rtol=0)
   explicit_checks+=1
v=json.loads((OUT/'verification.json').read_text());v['independent_expanded_patient_list_checks']=explicit_checks
(OUT/'verification.json').write_text(json.dumps(v,indent=2),encoding='utf8')

def pretty(x):return f'{float(x):.6f}'
def label(m):return 'Weibull-MLP' if m=='Weibull' else m
def exp_label(e):return 'Duration-dependent' if e=='Hard24' else 'Competing risks'
def target_label(t):return {'Survival':'Survival','CIF_Death_A':'Death_A CIF','CIF_Death_B':'Death_B CIF'}[t]
body=['<!doctype html><html><head><meta charset="utf-8"><title>S3 paired patient bootstrap</title><style>body{font:16px/1.5 Arial;max-width:1250px;margin:40px auto;color:#183c49}h1,h2{color:#164e61}table{border-collapse:collapse;width:100%;margin:20px 0 35px;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ccd8df;text-align:right}td:first-child,th:first-child,td:nth-child(3),th:nth-child(3){text-align:left}th{background:#e8f1f5}.uncertain{background:#fff1d9}code{font-size:14px}.note{color:#536b75}</style></head><body>',
 '<h1>S3: paired patient bootstrap, 5,000 resamples</h1><p>30 October 2026. Results saved separately; the manuscript is unchanged. No models were retrained and no trajectories were generated.</p>',
 '<h2>What was compared</h2><p>Every original ATLAS architecture using the common Sobol sampler was compared with each of the eight original direct models: 96 duration-dependent and 144 competing-risk comparisons. Each patient\'s full-curve MAE was averaged over five saved sampling seeds within each ATLAS fit, then over nine or five training fits. Direct neural predictions were averaged over their training fits; CPH was a numerically unchanged control. Curves were scored before averaging errors.</p>',
 '<p>Within each experiment, the original 300 held-out test patient IDs were resampled 5,000 times with replacement. The same patient multiplicities were reused for every model, landmark and target, with the original live/observed eligibility rules applied at each landmark. The resulting 95% percentile intervals describe patient-sampling uncertainty conditional on the saved fits and cohort. Random seed: 20261030.</p>',
 '<h2>Main findings</h2><ul><li>All 168 post-baseline comparisons had pointwise intervals entirely below zero for ATLAS minus comparator MAE.</li><li>Duration-dependent experiment: 95 of 96 intervals were below zero; the baseline Transformer versus DynamicDeepHit-Lite comparison included zero.</li><li>Competing risks: 143 of 144 intervals were below zero; the baseline Transformer versus Weibull-MLP Death_B CIF comparison included zero.</li><li>The headline Transformer duration-dependent comparison favored ATLAS at all three later landmarks, but was inconclusive at baseline. The headline GRU competing-risk comparison favored ATLAS in all nine landmark/target settings.</li></ul>',
 '<h2>Headline comparisons against the strongest original direct model</h2><p>Negative MAE differences favor ATLAS. The ATLAS architecture is fixed to Transformer for the duration-dependent experiment and GRU for competing risks. The strongest direct comparator at each setting is fixed by its original mean MAE, not reselected inside each resample. These are descriptive, previously selected comparisons; the intervals do not account for model-selection uncertainty.</p>',
 '<table><thead><tr><th>Experiment</th><th>L / n</th><th>Target / comparator</th><th>ATLAS MAE</th><th>Direct MAE</th><th>Difference</th><th>95% interval</th><th>Reduction % [95% interval]</th></tr></thead><tbody>']
for r in heads:
 row=[exp_label(r['experiment']),r['landmark']+' / '+r['n_patients'],target_label(r['target'])+' / '+label(r['comparator']),pretty(r['ATLAS_MAE']),pretty(r['comparator_MAE']),pretty(r['difference_ATLAS_minus_comparator']),'['+pretty(r['CI_lower'])+', '+pretty(r['CI_upper'])+']',f"{float(r['relative_reduction_pct']):.1f} [{float(r['relative_CI_lower']):.1f}, {float(r['relative_CI_upper']):.1f}]"]
 body.append('<tr'+(' class="uncertain"' if r['interval_position']=='includes_zero' else '')+'>'+''.join('<td>'+html.escape(c)+'</td>' for c in row)+'</tr>')
body+=['</tbody></table>',
 '<h2>Brief interpretation</h2><p>The post-baseline error differences remain clear after accounting for which patients appear in the evaluation sample. Thus those differences are not described solely by training-seed variability: they also persist under a paired patient-sampling analysis of the saved procedures. This supports the reported benchmark comparisons, rather than establishing an advantage over models outside the original comparison set.</p>',
 '<p>Baseline results need more qualified wording. The duration-dependent Transformer has a slightly higher average error than DynamicDeepHit-Lite, but its interval crosses zero; competing-risk Transformer Death_B results are likewise inconclusive against Weibull-MLP. The duration-dependent GRU and RNN average-fit comparisons still have intervals favoring ATLAS at baseline. That does not contradict variation in the rankings of individual training runs: the bootstrap here assesses the mean error across the saved fits, not the probability that a newly trained individual model will win.</p>',
 '<p>Fewer patients remain at later landmarks (57 at duration-dependent L=24). Their resampled counts vary, and confidence intervals capture that patient-sampling variability. Nevertheless, surviving patient populations and scored time windows differ across landmarks, so changes in error or interval width cannot isolate the effect of supplying more history.</p>',
 '<h2>Scope of uncertainty</h2><p>All intervals are pointwise, with no adjustment for 240 comparisons; the counts of intervals excluding zero are not a simultaneous confidence statement or a family-wise significance test. Models, feature choices, architecture choices, archived sampling realizations and training data are held fixed. The analysis does not include new-training-cohort uncertainty, a full re-selection procedure, clinical transport, or uncertainty in the known synthetic reference curves. The reported absolute-error reductions are not reductions in mortality risk or evidence of clinical calibration.</p>',
 '<h2>Files</h2><p><code>headline_comparisons.csv</code> contains the 13 headline estimates and intervals; <code>paired_comparisons.csv</code> contains all 240 comparisons; <code>model_MAE_intervals.csv</code> contains 134 per-model intervals. Patient-error arrays, bootstrap multiplicities and bootstrap model means are saved so each calculation can be reproduced. The analysis plan and verification details remain separate from any proposed paper text.</p></body></html>']
(OUT/'RESULTS.html').write_text('\n'.join(body),encoding='utf8')
lines=['S3: paired patient bootstrap, 5,000 resamples','30 October 2026. Manuscript unchanged.','',
 'Random seed 20261030; CPU computation including loading took about 7 seconds.',
 'All original ATLAS architectures with common Sobol versus all eight original direct comparators.',
 'One shared set of resampled baseline patient IDs per experiment, with landmark eligibility retained.',
 'Per-patient errors averaged over sampling seeds within training run and then over training runs.',
 'Pointwise percentile 95% intervals; no multiple-comparison adjustment.',
 '', 'RESULTS',
 'Duration-dependent: 95/96 intervals favor ATLAS; all 72 post-baseline intervals favor ATLAS.',
 'Competing risks: 143/144 intervals favor ATLAS; all 96 post-baseline intervals favor ATLAS.',
 'Two intervals include zero: baseline Hard24 Transformer versus DynamicDeepHit-Lite; baseline CR3 Transformer Death_B CIF versus Weibull-MLP.',
 '', 'HEADLINE: ATLAS minus strongest original direct comparator; negative favors ATLAS.']
for r in heads:lines.append(f"{r['experiment']} L={r['landmark']} {r['target']}, n={r['n_patients']}: {r['ATLAS_model']} vs {label(r['comparator'])}; difference={pretty(r['difference_ATLAS_minus_comparator'])}, 95% interval [{pretty(r['CI_lower'])}, {pretty(r['CI_upper'])}].")
lines+=['','DISCUSSION',
 'Later-landmark differences remain clear when patient-sampling uncertainty is considered, conditional on the saved fitted procedures.',
 'The baseline exceptions reinforce caution about broad baseline superiority claims.',
 'Patient-bootstrap results for the training-run average do not guarantee that every individual training run or a new fitted model will win.',
 'The same eligible patients and prediction windows are used for every model within a setting, but these populations/windows change across landmarks.',
 'These are pointwise intervals, not joint claims across all comparisons. Architecture and strongest-comparator selection uncertainty is not included.',
 'New training cohorts, training-data uncertainty, clinical generalization and clinical calibration remain unassessed.',
 'The MMLR/HAMMLR exploratory comparisons are outside this S3 analysis and remain shelved.']
(OUT/'RESULTS.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
for name in ['run_s3_bootstrap.py','summarize_s3_bootstrap.py']:shutil.copy2(Path(__file__).resolve().parent/name,OUT/name)
manifest={'date':'2026-10-30','status':'completed; manuscript unchanged','files':[{'path':p.relative_to(OUT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='manifest.json']}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
print('Independent explicit patient-list checks passed:',explicit_checks)
print('Report and',len(manifest['files']),'manifest entries written.')
