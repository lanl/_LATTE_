#!/bin/bash
#SBATCH --job-name=latte-rollout
#SBATCH --output=logs/latte-rollout-%j.out
#SBATCH --error=logs/latte-rollout-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00

set -euo pipefail

: "${LATTE_REPO_ROOT:=$PWD}"
: "${LATTE_CONDA_ENV:=latte}"
: "${LATTE_TEST_LIST:?Set LATTE_TEST_LIST to a text file with one HDF5 path per line.}"
: "${LATTE_GPT_CKPT:?Set LATTE_GPT_CKPT to a transformer checkpoint.}"
: "${LATTE_VQ_CKPT:?Set LATTE_VQ_CKPT to a VQ-VAE checkpoint.}"
: "${LATTE_MINMAX_JSON:?Set LATTE_MINMAX_JSON to the normalization JSON.}"
: "${LATTE_FIELD:=t0_fields/av_density}"
: "${LATTE_OUT_DIR:=runs/rollouts_example}"
: "${LATTE_STEPS:=100}"

cd "${LATTE_REPO_ROOT}"
mkdir -p logs "${LATTE_OUT_DIR}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${LATTE_CONDA_ENV}"
fi

export PYTHONPATH="${LATTE_REPO_ROOT}:${PYTHONPATH:-}"

python scripts/rollout_from_raw_firstframe.py \
  --test_list_txt "${LATTE_TEST_LIST}" \
  --field "${LATTE_FIELD}" \
  --gpt_ckpt "${LATTE_GPT_CKPT}" \
  --vq_ckpt "${LATTE_VQ_CKPT}" \
  --minmax_json "${LATTE_MINMAX_JSON}" \
  --out_dir "${LATTE_OUT_DIR}" \
  --steps "${LATTE_STEPS}" \
  --resume
