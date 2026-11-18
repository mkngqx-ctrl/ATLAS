from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BENCHMARKS=ROOT/'outputs/runs'
DEPENDENCIES=Path(__file__).parent
def relocated(path):
    name=Path(str(path).replace(chr(92),'/')).name
    exp='Hard24' if 'hard24' in name.lower() else 'CR3'
    seed=name.split('_')[-1] if 'training_seed_' in name else '42'
    return BENCHMARKS/exp/seed
