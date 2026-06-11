# LATTE

LATTE is a research codebase for learning latent-token dynamics models for
time-evolving scientific fields. The typical workflow is:

1. Train a VQ-VAE on simulation frames.
2. Export VQ tokens for each trajectory.
3. Train a transformer over latent tokens.
4. Roll out the transformer and decode predicted tokens to pixelspace with VQ-VAE.

This public repository contains code and configuration templates only. It does
not include datasets, checkpoints, generated tokens, or rollouts.

## Installation

Install PyTorch for your hardware first, following the official PyTorch
instructions for your platform. Then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

For Conda:

```bash
conda create -n latte python=3.11 pip
conda activate latte
python -m pip install torch
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

## Configure Your Data

LATTE uses JSON dataset registries. Public examples live in
`configs/examples/` and are meant to be copied and edited for your filesystem.
The registry loader expands environment variables in selected dataset entries,
so paths can use `${LATTE_DATA_ROOT}` and `${LATTE_WORK_ROOT}`.

```bash
export LATTE_REPO_ROOT="$PWD"
export LATTE_DATA_ROOT="/path/to/your/data"
export LATTE_WORK_ROOT="/path/to/your/latte-runs"
```

The VQ-VAE example assumes an HDF5 file with field arrays shaped like
`(N_traj, T, H, W)` and a min/max normalization JSON. The transformer example
assumes token `.npz` files exported from LATTE.

## Workflow

Train a VQ-VAE:

```bash
python scripts/train_vqvae.py \
  --datasets_json configs/examples/datasets_vqvae.example.json \
  --train_datasets ExampleHDF5Fields \
  --val_datasets ExampleHDF5Fields \
  --out_dir "${LATTE_WORK_ROOT}/vqvae_example" \
  --batch_size 8 \
  --num_workers 4 \
  --devices 1 \
  --precision 32 \
  --max_epochs 50
```

Export tokens from a trained VQ-VAE:

```bash
python scripts/export_vq_tokens.py \
  --datasets_json configs/examples/datasets_vqvae.example.json \
  --dataset ExampleHDF5Fields \
  --split train \
  --ckpt "${LATTE_WORK_ROOT}/vqvae_example/checkpoints/last.ckpt" \
  --out_dir "${LATTE_WORK_ROOT}/tokens/example"
```

Train a transformer over exported tokens:

```bash
python scripts/transformer.py \
  --datasets_json configs/examples/datasets_transformer.example.json \
  --dataset_name ExampleTokenDataset \
  --out_dir "${LATTE_WORK_ROOT}/transformer_example" \
  --batch_size 4 \
  --num_workers 4 \
  --max_steps 10000 \
  --use_sdpa
```

Rollout scripts are dataset-format specific. For HDF5 trajectories where the
first frame is read directly from the raw file, see:

```bash
python scripts/rollout_from_raw_firstframe.py --help
```

SLURM templates are in `sbatch/`. They are intentionally generic and use
environment variables so users can adapt them to their own cluster.

## Repository Hygiene

Generated artifacts should not be committed. Keep checkpoints, tokens,
rollouts, plots, logs, and local data outside Git or under ignored directories
such as `runs/`, `outputs/`, `tokens/`, `rollouts/`, and `checkpoints/`.

## License
