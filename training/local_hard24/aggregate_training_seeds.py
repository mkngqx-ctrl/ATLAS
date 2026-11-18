"""Aggregate saved hard24 runs only. No training or model inference is imported."""
from pathlib import Path
import copy, hashlib, json, zipfile
import numpy as np
import pandas as pd

VERSION = '1.0'
ROLLOUT_SEEDS = [41001, 52001, 63001, 74001, 85001]
GROUP = ['target', 'landmark', 'model', 'method']
METRICS = ['MAE', 'post24_MAE', 'RMSE', 'seconds', 'min_ESS']
FIXED = ['CPH', 'Oracle sampler']
SAMPLERS = dict(iid='IID', stratified='Stratified', sobol='Sobol',
                antithetic='Antithetic', importance='Importance',
                weighted_branching='WeightedBranching', conditional_mc='ConditionalMC')

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def seed_of(settings):
    seed = settings.get('training_seed', settings.get('seed'))
    if seed is None: raise ValueError('Missing training seed in settings')
    return int(seed)

def display_name(model, method):
    if model.startswith('Glazier-'):
        return 'ATLAS-' + model[len('Glazier-'):] + '-' + SAMPLERS.get(method, method)
    return model + (' / ' + method if method != 'direct' else '')

def resolve_run(path, expected_seed, cache_dir=None):
    """Accept an extracted run, its wrapper directory, or an existing results ZIP."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f'Seed {expected_seed}: path not found: {path}. '
                                'Set RUNS to the existing local folder or results ZIP.')
    if path.is_file():
        if path.suffix.lower() != '.zip': raise ValueError('Expected a results directory or ZIP')
        cache = Path(cache_dir or Path(__file__).resolve().parent.parent / 'archive_cache').resolve()
        cache = cache / digest(path)[:20]
        if not (cache / '.extracted.json').exists():
            cache.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(path) as archive:
                for item in archive.infolist():
                    target = (cache / item.filename).resolve()
                    if not target.is_relative_to(cache): raise ValueError('Unsafe ZIP member path')
                archive.extractall(cache)
            (cache / '.extracted.json').write_text(json.dumps({'archive': str(path)}))
        path = cache
    if (path / 'training_settings.json').is_file(): candidates = [path]
    else: candidates = [p.parent for p in path.rglob('training_settings.json')]
    candidates = [p for p in candidates if (p / 'evaluation/replicate_metrics.csv').is_file()
                  and seed_of(read_json(p / 'training_settings.json')) == int(expected_seed)]
    if len(candidates) != 1:
        raise ValueError(f'Expected exactly one completed seed-{expected_seed} result root under '
                         f'{path}; found {candidates}. Set the path to the exact run.')
    return candidates[0]

def normalized_options(settings):
    options = copy.deepcopy(settings.get('history_options', settings.get('options', {})))
    for name in ('DynamicDeepHit', 'SurvTransformer', 'SurvTransformer-Discrete'):
        if name in options:
            options[name].setdefault('validation_index_seed', seed_of(settings) + 103)
    return options

def summarize_checkpoint(frame):
    rows = []
    for key, group in frame.groupby(['training_seed', *GROUP], sort=True):
        row = dict(zip(['training_seed', *GROUP], key))
        row['rollout_count'] = len(group)
        for metric in METRICS:
            if metric not in group: continue
            values = pd.to_numeric(group[metric], errors='raise')
            row[metric + '_mean'] = values.mean()
            row[metric + '_rollout_SD'] = values.std(ddof=1)
        row['display_name'] = display_name(row['model'], row['method'])
        rows.append(row)
    return pd.DataFrame(rows)

def aggregate_runs(runs, destination, make_plots=True, cache_dir=None):
    """runs maps training seed to saved result folder/ZIP. Never invokes training."""
    if len(runs) < 2: raise ValueError('Supply at least two distinct training seeds')
    runs = {int(seed): path for seed, path in runs.items()}
    resolved = {seed: resolve_run(path, seed, cache_dir) for seed, path in sorted(runs.items())}
    destination = Path(destination).expanduser().resolve()
    for root in resolved.values():
        if destination == root or root in destination.parents or destination in root.parents:
            raise ValueError('Aggregation destination must be separate from every source run')
    records = []; frames = []; expected_signature = None
    for seed, root in resolved.items():
        complete = read_json(root / 'evaluation/COMPLETE.json')
        if complete.get('complete') is not True: raise ValueError(f'Incomplete run: {root}')
        settings = read_json(root / 'training_settings.json')
        evaluation = read_json(root / 'evaluation/evaluation_settings.json')
        if seed_of(settings) != seed: raise ValueError(f'Training seed mismatch: {root}')
        if evaluation['seeds'] != ROLLOUT_SEEDS:
            raise ValueError(f'Rollout seeds differ for training seed {seed}')
        if evaluation['M'] != 10000 or evaluation['landmarks'] != [0, 20, 23, 24]:
            raise ValueError('This helper expects the full hard24 experiment with M=10,000')
        if settings['cohort_seed'] != 24001 or settings['n'] != 50000:
            raise ValueError('Wrong hard24 cohort settings')
        cohort_hash = digest(root / 'checkpoints/cohort.npz')
        if cohort_hash != evaluation['checkpoint_hashes']['cohort.npz']:
            raise ValueError(f'Cohort checksum mismatch: {root}')
        for name, expected in evaluation['checkpoint_hashes'].items():
            if digest(root / 'checkpoints' / name) != expected:
                raise ValueError(f'Checkpoint checksum mismatch: {root / "checkpoints" / name}')
        config = read_json(root / 'checkpoints/run_config.json')
        for model, fit in config['training'].items():
            if model != 'CPH' and fit.get('seed') != seed:
                raise ValueError(f'{model} has a different training seed in {root}')
        if records:
            first = records[0]
            if cohort_hash != first['cohort_sha256']: raise ValueError('Cohorts/splits differ')
            for key in ('M', 'n_eval', 'seeds', 'landmarks', 'batch_size', 'rule', 'importance_death_tilt'):
                if evaluation.get(key) != first['evaluation'].get(key):
                    raise ValueError(f'Evaluation setting differs: {key}, seed {seed}')
            for key in ('n', 'epochs', 'patience', 'cohort_seed', 'scenario', 'rule'):
                if settings.get(key) != first['training'].get(key):
                    raise ValueError(f'Training/data setting differs: {key}, seed {seed}')
            if normalized_options(settings) != normalized_options(first['training']):
                raise ValueError('Model options or validation-prefix subset differ')
            # Seed 42 predates the audited seed-control patch. Later reruns must use the same patched source.
            previous_reruns = [r for r in records if r['training_seed'] != 42]
            if seed != 42 and previous_reruns and settings['source_hashes'] != previous_reruns[0]['training']['source_hashes']:
                raise ValueError('Rerun source hashes differ; investigate before combining')
            for L in evaluation['landmarks']:
                with np.load(Path(first['root']) / f'evaluation/L{L}/reference.npz') as a, \
                     np.load(root / f'evaluation/L{L}/reference.npz') as b:
                    for key in ('patient_ids', 'times', 'truth'):
                        np.testing.assert_array_equal(a[key], b[key], err_msg=f'Seed {seed}, L={L}, {key}')
        frame = pd.read_csv(root / 'evaluation/replicate_metrics.csv').rename(columns={'seed': 'rollout_seed'})
        if complete.get('rows') != len(frame): raise ValueError('COMPLETE row count differs from metrics')
        frame['training_seed'] = seed
        if 'target' not in frame: frame['target'] = 'Survival'
        required = GROUP + ['rollout_seed', 'MAE', 'post24_MAE', 'RMSE', 'n', 'M']
        if any(k not in frame for k in required): raise ValueError('Required metric columns missing')
        if frame.duplicated(GROUP + ['rollout_seed']).any(): raise ValueError('Duplicate metric rows')
        if not np.isfinite(frame[['MAE', 'post24_MAE', 'RMSE']].to_numpy()).all():
            raise ValueError('Nonfinite accuracy metric')
        signature = set(map(tuple, frame[GROUP + ['rollout_seed', 'n', 'M']].to_numpy()))
        if expected_signature is None: expected_signature = signature
        elif signature != expected_signature: raise ValueError('Missing/mismatched model, sampler or patient counts')
        for key, group in frame.groupby(GROUP):
            expected = set(ROLLOUT_SEEDS) if key[2].startswith('Glazier-') or key[2] == 'Oracle sampler' else {-1}
            if set(group.rollout_seed) != expected: raise ValueError('Missing rollout repetitions: ' + str(key))
        frame['source_run'] = str(root)
        frame['display_name'] = [display_name(m, s) for m, s in zip(frame.model, frame.method)]
        frames.append(frame)
        records.append(dict(training_seed=seed, root=str(root), cohort_sha256=cohort_hash,
                            training=settings, evaluation=evaluation, training_records=config['training'],
                            metrics_sha256=digest(root / 'evaluation/replicate_metrics.csv')))
    all_rows = pd.concat(frames, ignore_index=True)
    fixed = all_rows[all_rows.model.isin(FIXED)].copy()
    neural = all_rows[~all_rows.model.isin(FIXED)].copy()
    per = summarize_checkpoint(neural)
    summary_rows = []
    for key, group in per.groupby(GROUP, sort=True):
        if set(group.training_seed) != set(resolved): raise ValueError('Missing training checkpoint')
        row = dict(zip(GROUP, key));row['training_checkpoints'] = len(group)
        for metric in METRICS:
            if metric + '_mean' not in group: continue
            values = group[metric + '_mean']; within = group[metric + '_rollout_SD'].dropna()
            row[metric + '_mean'] = values.mean()
            row[metric + '_between_training_seed_SD'] = values.std(ddof=1)
            row[metric + '_pooled_within_checkpoint_rollout_SD'] = np.sqrt((within**2).mean()) if len(within) else np.nan
            row[metric + '_checkpoint_means_min'] = values.min()
            row[metric + '_checkpoint_means_max'] = values.max()
        row['display_name'] = display_name(row['model'], row['method']);summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    fixed_per = summarize_checkpoint(fixed)
    baseline_matches = True
    for _, group in fixed_per.groupby(GROUP):
        baseline_matches &= bool(np.allclose(group.MAE_mean, group.MAE_mean.iloc[0], atol=1e-12, rtol=0))
    # Re-running an identical aggregation is safe; different inputs require a new output folder.
    marker = dict(version=VERSION, runs=[dict(seed=r['training_seed'], root=r['root'],
                  metrics_sha256=r['metrics_sha256'], cohort_sha256=r['cohort_sha256']) for r in records])
    plan = destination / 'aggregation_plan.json'
    if destination.exists() and any(destination.iterdir()):
        if not plan.exists() or read_json(plan) != marker:
            raise ValueError('Destination contains another aggregation. Choose a fresh folder.')
    destination.mkdir(parents=True, exist_ok=True)
    plan.write_text(json.dumps(marker, indent=2))
    all_rows.to_csv(destination / 'all_metrics_with_seed_types.csv', index=False)
    fixed.to_csv(destination / 'training_independent_baselines.csv', index=False)
    fixed_per.rename(columns={'training_seed': 'source_run_training_seed'}).to_csv(
        destination / 'training_independent_baseline_summaries.csv', index=False)
    per.to_csv(destination / 'per_checkpoint_metrics.csv', index=False)
    summary.to_csv(destination / 'training_seed_summary_all_metrics.csv', index=False)
    short = summary[GROUP + ['training_checkpoints', 'MAE_mean', 'MAE_between_training_seed_SD',
        'MAE_pooled_within_checkpoint_rollout_SD', 'MAE_checkpoint_means_min', 'MAE_checkpoint_means_max', 'display_name']].rename(columns={
            'MAE_between_training_seed_SD': 'between_checkpoint_mean_SD',
            'MAE_pooled_within_checkpoint_rollout_SD': 'pooled_within_checkpoint_rollout_SD',
            'MAE_checkpoint_means_min': 'checkpoint_means_min', 'MAE_checkpoint_means_max': 'checkpoint_means_max'})
    count = {2: 'two', 3: 'three', 5: 'five'}.get(len(resolved), str(len(resolved)))
    mae_file = destination / f'{count}_training_seed_MAE.csv';short.to_csv(mae_file, index=False)
    (destination / 'source_settings.json').write_text(json.dumps(records, indent=2))
    training_rows = [dict(training_seed=r['training_seed'], model=name, **fit)
                     for r in records for name, fit in r['training_records'].items()]
    pd.json_normalize(training_rows).to_csv(destination / 'training_metadata.csv', index=False)
    notes = f'''Training seeds: {sorted(resolved)}. No training or model inference was performed.
For each configuration, average its five rollout metrics within each checkpoint, then average the checkpoint means equally.
Between-checkpoint SD is the sample SD of {len(resolved)} checkpoint means. It retains finite-rollout error.
Within-checkpoint SD summarizes five fixed-checkpoint rollout repetitions. Pooled SD is sqrt(mean of their variances).
Direct predictions have one value per checkpoint; within-checkpoint SD is undefined and is left blank.
{5 * len(resolved)} rollout measurements for {len(resolved)} training seeds are {len(resolved)} fitted checkpoints, not {5 * len(resolved)} fits.
Matched rollout seeds can correlate sampling errors across checkpoints. These SDs are descriptive, not confidence intervals.
CPH and oracle records are separate. Their source-run labels do not imply independently trained neural models.
CPH/oracle mean MAEs agree across source runs: {baseline_matches}.
All original metric columns and provenance are retained in all_metrics_with_seed_types.csv.
MAE, post24_MAE and RMSE are averaged as recorded metrics; RMSE is not recomputed from pooled squared errors.
seconds records evaluation time, not training time; cached timings and hardware differences limit timing comparisons.
min_ESS is a sampler diagnostic, not a guarantee of independent paths. Its SD is not a survival uncertainty estimate.
Training metadata are copied from run_config.json; unavailable legacy stopping epochs cannot be recovered here.
Seed-42 source hashes may differ due to the previously audited seed-control patch; later rerun source hashes must match.
No predicted curves or checkpoints are overwritten. Figures summarize metrics, not ensemble survival curves.
'''
    (destination / 'INTERPRETATION.txt').write_text(notes)
    if make_plots: plot_results(per, summary, fixed_per, destination)
    print(f'Aggregated training seeds {sorted(resolved)} -> {destination}')
    return dict(destination=destination, mae_file=mae_file, all_metrics_file=destination / 'training_seed_summary_all_metrics.csv',
                figures=destination / 'ATLAS_figures', rows=len(all_rows))

def plot_results(per, summary, fixed_per, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots = destination / 'ATLAS_figures';plots.mkdir(exist_ok=True)
    seeds = sorted(per.training_seed.unique()); markers = ['o', 's', '^', 'D', 'v']
    first_seed = fixed_per.training_seed.min()
    for (target, landmark), group in summary.groupby(['target', 'landmark']):
        baseline = fixed_per[(fixed_per.target == target) & (fixed_per.landmark == landmark) &
                             (fixed_per.training_seed == first_seed)].copy()
        ordered = pd.concat([group, baseline], ignore_index=True).sort_values('MAE_mean')
        y = np.arange(len(ordered));height = max(5, .32 * len(ordered) + 2.0)
        fig, axes = plt.subplots(1, 3, figsize=(17, height), sharey=True, layout='constrained')
        for ax, metric, title in zip(axes, ['MAE', 'post24_MAE', 'RMSE'], ['Overall MAE', 'MAE at steps 24-30', 'RMSE']):
            means = ordered[metric + '_mean'].to_numpy()
            fixed_mask = ordered.model.isin(FIXED).to_numpy()
            errors = ordered.get(metric + '_between_training_seed_SD', pd.Series(np.nan, index=ordered.index)).fillna(0).to_numpy()
            ax.barh(y, means, xerr=errors, color=['#888888' if fixed else '#8064A2' if model.startswith('Glazier-') else '#168B82'
                         for fixed, model in zip(fixed_mask, ordered.model)], alpha=.55, capsize=2)
            for j, seed in enumerate(seeds):
                for i, row in enumerate(ordered.itertuples()):
                    if row.model in FIXED: continue
                    cell = per[(per.training_seed == seed) & (per.target == target) & (per.landmark == landmark) &
                               (per.model == row.model) & (per.method == row.method)]
                    ax.scatter(cell[metric + '_mean'], [i], marker=markers[j % len(markers)], s=18,
                               color=f'C{j}', zorder=3, label=f'Training seed {seed}' if i == np.flatnonzero(~fixed_mask)[0] else None)
            ax.set_title(title);ax.set_xlabel('Error; lower is better');ax.grid(axis='x', alpha=.2)
        axes[0].set_yticks(y, ordered.display_name);axes[0].invert_yaxis();axes[-1].legend(fontsize=8)
        fig.suptitle(f'Hard24: {target}, L={landmark}; training seeds ' + ', '.join(map(str, seeds)) +
                     '\nBars: mean; error bars: SD of checkpoint means; markers: individual checkpoints.\n'
                     f'CPH/oracle shown from seed-{first_seed} source only, without training-variation error bars.', fontsize=10)
        name = f'comparison_L{landmark}_{target}'
        fig.savefig(plots / (name + '.png'), dpi=180)
        fig.savefig(plots / (name + '.svg'))
        plt.close(fig)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True, help='Seed=folder or Seed=results.zip')
    parser.add_argument('--destination', required=True)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    mapping = dict(item.split('=', 1) for item in args.runs)
    aggregate_runs(mapping, args.destination, make_plots=not args.no_plots)
