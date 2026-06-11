#!/bin/bash
#SBATCH --job-name=latte-transformer
#SBATCH --output=logs/latte-transformer-%j.out
#SBATCH --error=logs/latte-transformer-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00

set -euo pipefail

: "${LATTE_REPO_ROOT:=$PWD}"
: "${LATTE_CONDA_ENV:=latte}"
: "${LATTE_DATASETS_JSON:=configs/examples/datasets_transformer.example.json}"
: "${LATTE_DATASET_NAME:=ExampleTokenDataset}"
: "${LATTE_OUT_DIR:=runs/transformer_example}"

cd "${LATTE_REPO_ROOT}"
mkdir -p logs "${LATTE_OUT_DIR}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${LATTE_CONDA_ENV}"
fi

export PYTHONPATH="${LATTE_REPO_ROOT}:${PYTHONPATH:-}"

python scripts/transformer.py \
  --datasets_json "${LATTE_DATASETS_JSON}" \
  --dataset_name "${LATTE_DATASET_NAME}" \
  --out_dir "${LATTE_OUT_DIR}" \
  --batch_size 4 \
  --num_workers 4 \
  --max_steps 10000 \
  --eval_every 500 \
  --save_every 1000 \
  --lr 1e-4 \
  --warmup_steps 500 \
  --n_layer_enc 4 \
  --n_layer_dec 8 \
  --n_head 8 \
  --n_embd 512 \
  --use_sdpa
