"""Local hard24 orchestration. Importing this module never starts training."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROLLOUT_SEEDS = [41001, 52001, 63001, 74001, 85001]
DEFAULT_TRAINING_SEEDS = [50, 51, 52, 53, 54]
LANDMARKS = [0, 20, 23, 24]


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def child_environment(seed):
    env = dict(os.environ)
    env['PYTHONHASHSEED'] = str(seed)
    env['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUNBUFFERED'] = '1'
    env['MPLBACKEND'] = 'Agg'
    support = str(HERE / 'python_dependencies')
    env['PYTHONPATH'] = support + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    return env


def check_environment(python=sys.executable):
    """Probe the actual child interpreter and the bundled Cox dependencies."""
    code = '''import datetime
if not hasattr(datetime, 'UTC'): datetime.UTC = datetime.timezone.utc
import sys,json,torch,numpy,pandas,scipy,matplotlib,sklearn,lifelines
print(json.dumps(dict(python=sys.executable,torch=str(torch.__version__),numpy=numpy.__version__,
pandas=pandas.__version__,scipy=scipy.__version__,lifelines=lifelines.__version__,
cuda=torch.cuda.is_available(),gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
gpu_memory_GiB=round(torch.cuda.get_device_properties(0).total_memory/1024**3,2) if torch.cuda.is_available() else None)))'''
    try:
        result = subprocess.run([str(python), '-c', code], env=child_environment(50),
                                check=True, capture_output=True, text=True, encoding='utf-8')
    except subprocess.CalledProcessError as exc:
        raise RuntimeError('The GPU-environment dependency check failed:\n' + exc.stderr) from exc
    info = json.loads(result.stdout.strip().splitlines()[-1])
    if not info['cuda']:
        raise RuntimeError('CUDA is unavailable in this Python. Select the Python (GPU_Env) notebook kernel.')
    return info


def make_plan(previous_run, output_root, seeds=DEFAULT_TRAINING_SEEDS, python=sys.executable):
    seeds = list(seeds)
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError('Supply exactly five distinct training seeds.')
    if any(type(seed) is not int or seed == 42 or not 0 <= seed < 2**32-103 for seed in seeds):
        raise ValueError('Use five integer seeds different from the original seed 42.')
    previous = Path(previous_run).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if previous == output or previous in output.parents or output in previous.parents:
        raise ValueError('New outputs must be separate from the original result directory.')
    required = ['training_settings.json', 'evaluation/evaluation_settings.json',
                'evaluation/COMPLETE.json', 'checkpoints/cohort.npz', 'checkpoints/DynamicDeepHit.pt']
    for name in required:
        if not (previous / name).is_file():
            raise FileNotFoundError(f'Missing {previous / name}. PREVIOUS_RUN must be the completed hard24 v4 seed-42 root.')
    settings = load_json(previous / 'training_settings.json')
    if settings.get('training_seed', settings.get('seed')) != 42:
        raise ValueError('PREVIOUS_RUN must be the original completed training-seed-42 hard24 v4 run.')
    if load_json(previous / 'evaluation/COMPLETE.json').get('complete') is not True:
        raise ValueError('The original evaluation is incomplete.')
    return dict(version=1, experiment='hard24', python=str(Path(python).resolve()),
                previous_run=str(previous), output_root=str(output), training_seeds=seeds,
                rollout_seeds=ROLLOUT_SEEDS, M=10000, landmarks=LANDMARKS,
                runs={str(seed): str(output / f'hard24_training_seed_{seed}') for seed in seeds},
                aggregation=str(output / ('hard24_training_seeds_' + '_'.join(map(str, seeds)))),
                runner_sha256=digest(HERE / 'second_seed.py'),
                worker_sha256=digest(HERE / 'local_worker.py'),
                runtime_sha256=digest(HERE / 'local_runtime.py'),
                source_manifest_sha256=digest(HERE / 'source_manifest.json'))


def command_for(plan, seed, prepare_only=False):
    command = [plan['python'], '-u', str(HERE / 'local_worker.py'), '--experiment', 'hard24',
               '--previous', plan['previous_run'], '--out', plan['runs'][str(seed)],
               '--training-seed', str(seed), '--device', 'cuda']
    if prepare_only:
        command.append('--prepare-only')
    return command


def stream_process(command, seed, log=None):
    """Stream progress and terminate the worker if the notebook is interrupted."""
    handle = None
    if log is not None:
        handle = Path(log).open('a', encoding='utf-8')
    process = None
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace', bufsize=1,
                                   env=child_environment(seed), cwd=str(HERE))
        for line in process.stdout:
            print(line, end='', flush=True)
            if handle:
                handle.write(line)
                handle.flush()
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        if process is not None and process.stdout:
            process.stdout.close()
        if handle:
            handle.close()


def validate_completed(root, seed, previous_run):
    """Reject partial/wrong-seed runs before skipping them or aggregating."""
    import numpy as np
    import pandas as pd
    from aggregate_training_seeds import normalized_options

    root = Path(root)
    previous = Path(previous_run)
    settings = load_json(root / 'training_settings.json')
    original = load_json(previous / 'training_settings.json')
    ev = load_json(root / 'evaluation/evaluation_settings.json')
    old_ev = load_json(previous / 'evaluation/evaluation_settings.json')
    completed = load_json(root / 'evaluation/COMPLETE.json')
    if completed.get('complete') is not True:
        raise ValueError(f'Incomplete evaluation: {root}')
    if settings.get('training_seed', settings.get('seed')) != seed:
        raise ValueError(f'Wrong training seed at {root}')
    if settings['n'] != 50000 or settings['epochs'] != 200 or settings['patience'] != 15:
        raise ValueError('Expected unchanged full-run training settings.')
    if settings['cohort_seed'] != 24001 or normalized_options(settings) != normalized_options(original):
        raise ValueError('Cohort or model/validation settings changed.')
    wanted_sources = load_json(HERE / 'source_manifest.json')['hard24']['revised_hashes']
    if settings['source_hashes'] != wanted_sources:
        raise ValueError('Completed run uses different model/evaluation source files.')
    for key in ['M', 'n_eval', 'seeds', 'landmarks', 'batch_size', 'rule', 'importance_death_tilt']:
        if ev.get(key) != old_ev.get(key):
            raise ValueError(f'Evaluation setting changed: {key}')
    for name, wanted in ev['checkpoint_hashes'].items():
        if digest(root / 'checkpoints' / name) != wanted:
            raise ValueError(f'Checkpoint checksum mismatch: {root / "checkpoints" / name}')
    if digest(root / 'checkpoints/cohort.npz') != digest(previous / 'checkpoints/cohort.npz'):
        raise ValueError('Saved cohort or splits differ.')
    expected_models = set(load_json(previous / 'checkpoints/run_config.json')['training'])
    fits = load_json(root / 'checkpoints/run_config.json')['training']
    if set(fits) != expected_models or any(fit.get('seed') != seed for name, fit in fits.items() if name != 'CPH'):
        raise ValueError('Missing model or wrong training seed in checkpoint metadata.')
    independence = load_json(root / 'checkpoint_independence_check.json')
    expected_neural_files = {name + '.pt' for name in expected_models if name != 'CPH'}
    if set(independence) != expected_neural_files:
        raise ValueError('Missing neural checkpoint independence records.')
    for name, record in independence.items():
        if record.get('different_learned_parameters') is not True or record['new_sha256'] != digest(root / 'checkpoints' / name):
            raise ValueError('Independence record does not match saved checkpoint.')
    frame = pd.read_csv(root / 'evaluation/replicate_metrics.csv')
    base = pd.read_csv(previous / 'evaluation/replicate_metrics.csv')
    columns = ['model', 'method', 'seed', 'landmark', 'n', 'M']
    if len(frame) != 472 or completed.get('rows') != 472 or frame.duplicated(columns[:4]).any():
        raise ValueError('Incomplete or duplicated evaluation rows; expected 472.')
    if set(map(tuple, frame[columns].to_numpy())) != set(map(tuple, base[columns].to_numpy())):
        raise ValueError('Model/sampler/rollout/patient counts do not match the original evaluation.')
    if not np.isfinite(frame[['MAE', 'post24_MAE', 'RMSE']].to_numpy()).all():
        raise ValueError('Nonfinite accuracy metric.')
    for row in frame.itertuples():
        if not (root / f'evaluation/L{row.landmark}/{row.model}_{row.method}_{row.seed}.npz').is_file():
            raise ValueError('A saved prediction file is missing.')
    for L in LANDMARKS:
        with np.load(root / f'evaluation/L{L}/reference.npz') as a, np.load(previous / f'evaluation/L{L}/reference.npz') as b:
            for name in ['patient_ids', 'times', 'truth']:
                np.testing.assert_array_equal(a[name], b[name])
    return dict(training_seed=seed, complete=True, metric_rows=len(frame), output=str(root))


def prepare_batch(plan):
    """No model training: verify settings and stage only cohort/reference/oracle data."""
    out = Path(plan['output_root'])
    marker = out / 'five_seed_plan.json'
    if marker.exists() and load_json(marker) != plan:
        raise ValueError('The output directory belongs to a different batch configuration. Choose a new output root.')
    out.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    for seed in plan['training_seeds']:
        stream_process(command_for(plan, seed, prepare_only=True), seed)
    print('Preflight passed for all five seeds. No neural models have been trained by this step.', flush=True)
    return plan


def run_batch(plan):
    """Five sequential GPU workers; only resume this batch's same-seed outputs."""
    prepare_batch(plan)
    status = []
    for number, seed in enumerate(plan['training_seeds'], 1):
        root = Path(plan['runs'][str(seed)])
        if (root / 'evaluation/COMPLETE.json').is_file():
            result = validate_completed(root, seed, plan['previous_run'])
            print(f'Seed {seed}: completed same-seed run verified; skipping training/evaluation.', flush=True)
            if not (root / 'ATLAS_figures/comparison_L0.png').is_file():
                stream_process([plan['python'], str(HERE / 'atlas_figures.py'), '--experiment', 'hard24',
                                '--evaluation', str(root / 'evaluation'), '--destination', str(root / 'ATLAS_figures')], seed)
        else:
            print(f'===== Independent training seed {seed}: run {number}/5 =====', flush=True)
            stream_process(command_for(plan, seed), seed, root / 'full_run.log')
            result = validate_completed(root, seed, plan['previous_run'])
        status.append(result)
        (Path(plan['output_root']) / 'batch_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    return status


def aggregate_batch(plan, make_plots=True):
    from aggregate_training_seeds import aggregate_runs
    for seed in plan['training_seeds']:
        validate_completed(plan['runs'][str(seed)], seed, plan['previous_run'])
    return aggregate_runs({seed: plan['runs'][str(seed)] for seed in plan['training_seeds']},
                          plan['aggregation'], make_plots=make_plots)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs=5, default=DEFAULT_TRAINING_SEEDS)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--aggregate-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(check_environment(), indent=2))
    plan = make_plan(args.previous, args.out, args.seeds)
    if args.prepare_only:
        prepare_batch(plan)
    elif args.aggregate_only:
        aggregate_batch(plan)
    else:
        run_batch(plan)
        aggregate_batch(plan)
