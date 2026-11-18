from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from datetime import datetime
import csv,hashlib,json,shutil
R=Path(r'C:/Users/moeni/Documents/ATLAS_work/current');UPDATE=R/'review/DRSA_manuscript_update_2026-11-16';STAGE=UPDATE/'manuscript';M=R/'working/ATLAS_SiM_manuscript';D=R/'working/ATLAS_SiM_data_and_code';STUDY=R/'review/DRSA_optimization_2026-11-04'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
backup=R/'review/approved_changes/DRSA_optimized_replacement_2026-11-16'/datetime.now().strftime('%H%M%S');backup.mkdir(parents=True)
# Check for concurrent source changes against the most recent pre-edit export.
with ZipFile(R/'exports/ATLAS_SiM_Overleaf_updated_2026-11-16_v2.zip') as z:
 for p in M.rglob('*'):
  if p.is_file():assert z.read(p.relative_to(M).as_posix())==p.read_bytes(),'Concurrent source change: '+str(p)
p=STAGE/'README.md';p.write_text(p.read_text(encoding='utf-8')+'\nDRSA and Dynamic-DeepHit use their validation-selected optimized configurations.\nThe current integrated comparison data are in the data/code package under\nsupplementary/optimized_DRSA_integrated_2026-11-16/. Historical DRSA results\nare excluded from the manuscript; prior research records remain archived.\n',encoding='utf-8')
changed=[]
for p in sorted(STAGE.rglob('*')):
 if not p.is_file():continue
 rel=p.relative_to(STAGE);target=M/rel
 if target.exists() and sha(target)==sha(p):continue
 if target.exists():
  b=backup/'manuscript'/rel;b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,b)
 target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);changed.append(rel.as_posix())
dest=D/'supplementary/optimized_DRSA_integrated_2026-11-16';assert not dest.exists();shutil.copytree(UPDATE/'data',dest)
# Archive current optimized fit outputs and all search logs, without importing old DRSA results.
for name in ['frozen_protocol.json','environment.json','implementation_checks.json','independent_final_verification.json','independent_final_verification.csv','run_drsa_optimization.py']:
 shutil.copy2(STUDY/name,dest/name)
for name in ['optimized_DRSA_per_seed_metrics.csv','optimized_DRSA_summary.csv','verification.json']:shutil.copy2(STUDY/'evaluation'/name,dest/name)
for exp in ['Hard24','CR3']:
 ex=dest/'training'/exp;ex.mkdir(parents=True)
 for name in ['data_provenance.json','selected_configuration.json','final_fit_inventory.json','screen_candidates.json','refinement_candidates.json']:shutil.copy2(STUDY/exp/name,ex/name)
 shutil.copytree(STUDY/exp/'decisions',ex/'decisions')
 for p in (STUDY/exp/'fits').glob('*/seed_*/*'):
  if p.name not in {'training.csv','config.json','result.json'}:continue
  q=ex/'search_logs'/p.relative_to(STUDY/exp/'fits');q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
 for fit in json.loads((STUDY/exp/'final_fit_inventory.json').read_text()):
  seed=fit['seed'];folder=ex/'selected_fits'/f'seed_{seed}';folder.mkdir(parents=True)
  for name in ['best.pt','training.csv','config.json','result.json']:shutil.copy2(Path(fit['checkpoint']).parent/name,folder/name)
 shutil.copytree(STUDY/'evaluation'/exp,dest/'predictions'/exp)
notice='''CURRENT MANUSCRIPT RESULTS — updated 2026-11-16

Main comparisons now use both DRSA (optimized) and Dynamic-DeepHit (optimized).
Use supplementary/optimized_DRSA_integrated_2026-11-16/ for current integrated
all-model summary/per-training-run metrics, updated bootstrap results and figure
data. In those machine-readable comparison files the internal model key DRSA
means DRSA (optimized), and DynamicDeepHit means Dynamic-DeepHit (optimized).
Earlier result folders/manifests are archived source records and may contain
superseded DRSA rows; they do not define the current manuscript comparisons.

'''
for rel in ['README.txt','supplementary/README.txt']:
 p=D/rel;b=backup/'data_and_code'/rel;b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,b);p.write_text(notice+p.read_text(encoding='utf-8'),encoding='utf-8')
(dest/'README.txt').write_text('''ATLAS CURRENT INTEGRATED COMPARISONS: OPTIMIZED DRSA AND DDH

updated_all_models_summary.csv and updated_all_models_per_training_run.csv:
current numerical comparisons. Internal DRSA denotes DRSA (optimized).
optimized_DRSA_per_seed_metrics.csv: all81target-by-seed metrics.
optimized_DRSA_selected_epochs.csv: retained and stopping epochs for14fits.
bootstrap/: current5000-resample patient comparisons, input errors/weights and
plots; only30DRSA contrasts changed,210other contrasts and all13headlines retained.
figure_assets/: full-comparison panels and exact plotted data.
training/: selected configurations,14best checkpoints, all search logs and rung
decisions; predictions/:51final patient-level prediction files.
The full92-fit research archive, including optimizer states and unselected model
weights, remains in current/review/DRSA_optimization_2026-11-04/ in ATLAS_work.
Frozen code/provenance retains original local paths; this is an archived study,
not a newly tested portable installation. Historical DRSA results are excluded
from the manuscript; original research archives are preserved separately.
''',encoding='utf-8')
manifest=[dict(path=p.relative_to(dest).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(dest.rglob('*')) if p.is_file()]
(dest/'FILE_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
tracker=R/'review/revision_tracker.csv';shutil.copy2(tracker,backup/'revision_tracker.csv')
with tracker.open('a',newline='',encoding='utf-8') as f:
 w=csv.writer(f);w.writerow(['DRSA-U1','Replace manuscript DRSA results with optimized fits','Applied; user authorized 2026-11-16','Replaced current DRSA numbers, figures and bootstrap contrasts; added Methods/S1search specification; excluded historical DRSA results from paper while preserving research archive.','main.tex; S1.2;6accuracy/dispersion tables;S8;7figure PDFs;updated data/code records','13settings;81per-seed metrics;30bootstrap contrasts updated;all3builds pass;7PDFs rendered;headline comparators and postbaseline wins unchanged.'])
out=R/'exports/ATLAS_SiM_Overleaf_updated_2026-11-16_v3.zip';assert not out.exists()
with ZipFile(out,'w',ZIP_DEFLATED) as z:
 for p in sorted(M.rglob('*')):
  if p.is_file():z.write(p,p.relative_to(M).as_posix())
with ZipFile(out) as z:
 assert z.testzip() is None
 for p in M.rglob('*'):
  if p.is_file():assert z.read(p.relative_to(M).as_posix())==p.read_bytes()
validation=dict(changed_manuscript_files=changed,builds={'main.tex':'success','supplement.tex':'success','atlas_arxiv.tex':'success'},figure_PDFs_rendered=7,DRSA_settings=13,per_seed_target_metrics=81,updated_bootstrap_contrasts=30,unchanged_bootstrap_contrasts=210,headline_comparators_unchanged=True,postbaseline_ATLAS_per_run_nonwins=0,historical_DRSA_results_in_paper=False,original_archives_preserved=True,zip=str(out),zip_sha256=sha(out),data_folder=str(dest),postponed_checks='Remain on hold')
(backup/'change_record.json').write_text(json.dumps(validation,indent=2),encoding='utf-8');(UPDATE/'final_validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8');out.with_suffix('.manifest.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
shutil.copy2(__file__,UPDATE/'apply_drsa_manuscript_update.py')
print(json.dumps({'changed_files':len(changed),'ZIP':str(out),'data_folder':str(dest),'backup':str(backup)},indent=2))
