# Reference calculations

These files preserve the bytes of the archived or hash-matched reference implementations. Only their filenames have changed here.

- `single_covariate_reference.py`: original `source/reference_recursion.py`, matched to the single-covariate run's recorded SHA-256. Use the Weibull scenario and semi-Markov (`semi`) state space for the paper's Section 6 experiment. Its other legacy scenarios are not the new Section 7 experiment.
- `competing_risks_generator_reference.py`: original `source/sfsa_cr.py` from the competing-risks archive's source snapshot. Includes that experiment's generator and exact reference calculations.

Both require Python and NumPy. Neither file is needed to compile the paper. This is a manuscript package, not a complete model-training environment. Retain the original result archives to reproduce predictions or inspect checkpoints. File hashes are in the repository-root `MANIFEST.sha256`; original recorded hashes and settings are in `../results_20260919/run_settings.json`.
