"""Single-run Windows entry point; normal process exit and error propagation."""
import argparse
from pathlib import Path


def main():
    from second_seed import run
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=['hard24'], required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--training-seed', type=int, required=True)
    parser.add_argument('--device', choices=['cuda'], default='cuda')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    run(args.experiment, args.previous, args.out, args.training_seed,
        args.device, args.prepare_only)


if __name__ == '__main__':
    main()
