#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=12:00:00
#SBATCH --job-name=generate_traces
#SBATCH --output=./logs/slurm-%A.out
#SBATCH --error=./logs/slurm-%A.err

set -euo pipefail

# ----- Edit these constants -----
TEACHER_MODEL="Qwen/Qwen3-8B"
STUDENT_MODEL="Qwen/Qwen3-1.7B"
SEED=42
ALGORITHM="entropy_aware_sd"  # standard_target, standard_draft, standard_sd, entropy_aware_sd
MIXTURE_TYPE="none"  # none, geometric, or convex
ENTROPY_TRANSFORM="linear"  # linear, sqrt, sqrt2, sq, or constant

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
DATA_FILE="${PROJECT_DIR}/data/math_boxed/train.parquet"

TEMPERATURE=1.0
TOP_K=-1
TOP_P=1.0
GAMMA=5
NUM_RUNS=8
BATCH_SIZE=4096
TENSOR_PARALLEL_SIZE=1
MAX_PROMPT_LENGTH=1024
MAX_NEW_TOKENS=4096

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

DP_SIZE="${SLURM_NTASKS}"
MODEL_TAG="${TEACHER_MODEL##*/}+${STUDENT_MODEL##*/}"

OUTPUT_DIR="${PROJECT_DIR}/generated_traces/${MODEL_TAG}/${ALGORITHM}/mix_${MIXTURE_TYPE}"
if [ "${ALGORITHM}" = "entropy_aware_sd" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}/transform_${ENTROPY_TRANSFORM}"
fi
OUTPUT_DIR="${OUTPUT_DIR}/seed_${SEED}"

mkdir -p "${OUTPUT_DIR}"

export PATCH_FILE="${PROJECT_DIR}/vllm_patches/v0.15.1-entropy-aware-decoding.patch"
export VLLM_SITE_ROOT="/usr/local/lib/python3.12/dist-packages/vllm"
srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
    bash -c 'patch -d "$(dirname "${VLLM_SITE_ROOT}")" -p1 < "${PATCH_FILE}"'

ARGS=(
    --draft-model "${STUDENT_MODEL}"
    --target-model "${TEACHER_MODEL}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --dp-size "${DP_SIZE}"
    --data-file "${DATA_FILE}"
    --output-dir "${OUTPUT_DIR}"
    --num-runs "${NUM_RUNS}"
    --algorithm "${ALGORITHM}"
    --seed "${SEED}"
    --temperature "${TEMPERATURE}"
    --top-k "${TOP_K}"
    --top-p "${TOP_P}"
    --gamma "${GAMMA}"
    --max-prompt-length "${MAX_PROMPT_LENGTH}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --batch-size "${BATCH_SIZE}"
)

if [ "${MIXTURE_TYPE}" != "none" ]; then
    ARGS+=(--entropy-aware-mixing "${MIXTURE_TYPE}")
    ARGS+=(--entropy-transform "${ENTROPY_TRANSFORM}")
fi

srun python3 "${PROJECT_DIR}/generate.py" "${ARGS[@]}"
python3 "${PROJECT_DIR}/generate.py" "${ARGS[@]}" --finalize-only
echo "Verified rollouts and training parquet written to: ${OUTPUT_DIR}"
