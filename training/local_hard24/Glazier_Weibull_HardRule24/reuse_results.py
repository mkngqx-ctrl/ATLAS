"""Validate and import saved hard24 predictions without repeating old rollouts."""
from pathlib import Path
import hashlib
import json
import pickle
import shutil
import numpy as np
import torch
from extension_workflow import SPECS, LEGACY_LABELS
from autoregressive_samplers import METHODS

ROOT = Path(__file__).resolve().parent

# Audited display-text-only renames. All other prediction-source changes fail
# closed. Keeping both hashes prevents treating an arbitrary changed model as Lite.
LABEL_ONLY_REVISIONS = {
    'source/dynamic_deephit_model_sm.py': (
        '1b27fce1938d9b891fe85f59fe31581678fd76469f77c6e9076187eb64583ec7',
        'ee821d29ae0df15a31f8ff8f662881760acc219f472b6e096f6f817e539cc24e'),
    'source/survtransformer_model_sm.py': (
        'cc17ce30f20d6676b9d0a7bee959b1585e27556b2b71738ef429eca6c4129bc2',
        '2c6503bce4757ff24574bc196a2c47ca714bd8825b7f06059e87391a6bf7ce0c'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_root(path):
    path = Path(path).resolve()
    if path.name in ('checkpoints', 'evaluation'):
        path = path.parent
    if not (path/'checkpoints/run_config.json').is_file() or not (path/'evaluation/evaluation_settings.json').is_file():
        raise ValueError('Reuse requires a result folder containing checkpoints and evaluation')
    return path


def plan_reuse(e, source, out, settings, reference_function):
    """Read-only preflight: verify settings, weights, patients, grids and truth.

    Returns an explicit copy plan. No old files, new files or checkpoints are
    written until every source prediction has passed validation.
    """
    source, out = run_root(source), Path(out).resolve()
    old_eval, old_ckpt = source/'evaluation', source/'checkpoints'
    if out == old_eval or out.is_relative_to(source):
        raise ValueError('Use an output directory outside the previous results folder')
    old = json.loads((old_eval/'evaluation_settings.json').read_text())
    cfg = json.loads((old_ckpt/'run_config.json').read_text())
    for key in ('M','n_eval','seeds','landmarks','batch_size','rule','importance_death_tilt'):
        if old.get(key) != settings.get(key):
            raise ValueError(f'Cannot reuse predictions: {key} differs from the saved evaluation')
    if cfg['kind'] != e.kind or cfg['scenario'] != e.scenario.name or cfg['cohort_seed'] != e.seed:
        raise ValueError('Previous scenario or cohort seed differs')
    expected_hash = old['checkpoint_hashes']['cohort.npz']
    if sha(old_ckpt/'cohort.npz') != expected_hash or sha(e.out/'cohort.npz') != expected_hash:
        raise ValueError('The evaluation must use the identical saved cohort and patient split')
    modules = ['source/sfsa_data_generation.py','source/sfsa_semi_markov.py',
               'source/sfsa_semi_markov_extras.py','source/reference_recursion.py',
               'autoregressive_samplers.py']
    old_sources = {k.replace('\\','/'): v for k,v in old['source_hashes'].items()}
    renamed = cfg.get('history_comparator_version', 0) < 2
    model_pairs = []
    for old_label in cfg['training']:
        label = LEGACY_LABELS.get(old_label, old_label) if renamed else old_label
        if label not in SPECS or label not in e.models:
            continue
        model, meta, _ = e.models[label]
        modules.append('source/' + SPECS[label][0] + '_sm.py')
        filename = old_label + ('.pkl' if label == 'CPH' else '.pt')
        if sha(old_ckpt/filename) != old['checkpoint_hashes'].get(filename):
            raise ValueError('Saved checkpoint no longer matches its evaluation: ' + filename)
        if label == 'CPH':
            with (old_ckpt/filename).open('rb') as handle:
                previous_model = pickle.load(handle)  # user-supplied trusted local run
            if pickle.dumps(model) != pickle.dumps(previous_model):
                raise ValueError('CPH model changed; its old predictions cannot be reused')
        else:
            saved = torch.load(old_ckpt/filename, map_location='cpu', weights_only=True)
            current = model.state_dict()
            if saved['class_name'] != type(model).__name__ or set(current) != set(saved['state_dict']):
                raise ValueError('Model architecture differs: ' + label)
            if any(not torch.equal(v.detach().cpu(), saved['state_dict'][k]) for k,v in current.items()):
                raise ValueError('Model weights differ: ' + label)
            clean = lambda m: {k:v for k,v in m.items() if k != 'device'}
            if clean(meta) != clean(saved['meta']):
                raise ValueError('Prediction metadata differs: ' + label)
        model_pairs.append((old_label, label))
    for name in modules:
        pair = (old_sources.get(name), sha(ROOT/name))
        if pair[0] != pair[1] and LABEL_ONLY_REVISIONS.get(name) != pair:
            raise ValueError('Prediction/reference source changed: ' + name)

    cohort = e.cohort
    ids0 = np.random.default_rng(e.seed+2).permutation(cohort['test_index'])[:settings['n_eval']]
    copies = []
    predictions = 0
    def add(source_file, destination_file):
        digest = sha(source_file)
        if destination_file.exists() and sha(destination_file) != digest:
            raise ValueError('Destination contains a different cached file: ' + str(destination_file))
        copies.append(dict(source=str(source_file), destination=str(destination_file), sha256=digest))

    for L in settings['landmarks']:
        ids = ids0 if L == 0 else ids0[cohort['states'][ids0,L-1] < 3]
        if not len(ids):
            continue
        old_folder, new_folder = old_eval/f'L{L}', out/f'L{L}'
        reference_path = old_folder/'reference.npz'
        with np.load(reference_path, allow_pickle=False) as z:
            np.testing.assert_array_equal(z['patient_ids'], ids, err_msg='Saved evaluation patients differ')
            np.testing.assert_array_equal(z['times'], np.arange(L+1,31), err_msg='Time grid differs')
            truth = z['truth']
        fresh_truth = reference_function(e, ids, L)
        np.testing.assert_allclose(truth, fresh_truth, atol=1e-12, rtol=1e-12,
                                   err_msg='Saved deterministic references differ')
        add(reference_path, new_folder/'reference.npz')
        tasks = []
        for old_label,label in model_pairs:
            if label.startswith('Glazier'):
                tasks += [(old_label,label,method,seed) for method in METHODS for seed in settings['seeds']]
            else:
                tasks.append((old_label,label,'direct',-1))
        tasks += [('Oracle sampler','Oracle sampler','iid',seed) for seed in settings['seeds']]
        for old_label,label,method,seed in tasks:
            name = f'{old_label}_{method}_{seed}.npz'
            file = old_folder/name
            with np.load(file, allow_pickle=False) as z:
                prediction = z['prediction']
                seconds, ess = float(z['seconds']), float(z['min_ESS'])
            if prediction.shape != truth.shape or not np.isfinite(prediction).all():
                raise ValueError('Invalid saved prediction: ' + name)
            if prediction.min() < -1e-6 or prediction.max() > 1+1e-6 or (np.diff(prediction,axis=1)>1e-6).any():
                raise ValueError('Invalid survival probabilities: ' + name)
            if not np.isfinite(seconds) or seconds < 0:
                raise ValueError('Invalid saved runtime: ' + name)
            add(file, new_folder/f'{label}_{method}_{seed}.npz')
            predictions += 1
    return dict(source_run=str(source), saved_predictions=predictions, reference_files=len(copies)-predictions,
                cohort_sha256=expected_hash, source_settings_sha256=sha(old_eval/'evaluation_settings.json'),
                label_mapping=dict(model_pairs), files=copies)


def apply_reuse(plan, out):
    for item in plan['files']:
        destination = Path(item['destination'])
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            # Source files may have changed between preflight and copy.
            if sha(item['source']) != item['sha256']:
                raise ValueError('Source prediction changed during import')
            temporary = destination.with_suffix('.importing')
            shutil.copy2(item['source'], temporary)
            if sha(temporary) != item['sha256']:
                raise IOError('Imported prediction checksum mismatch')
            temporary.replace(destination)
    Path(out,'reused_predictions.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    print(f"Reused {plan['saved_predictions']} saved predictions; no old rollouts repeated.", flush=True)
