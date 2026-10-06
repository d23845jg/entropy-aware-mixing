#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --job-name=verify
#SBATCH --output=./logs/slurm-%A.out
#SBATCH --error=./logs/slurm-%A.err

set -euo pipefail

# ----- Edit these constants -----
SEED=0

export VLLM_WORKER_MULTIPROC_METHOD=spawn

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_DIR}"

MODEL_PATH="$(realpath -- "${1:?Usage: sbatch verify.sh MODEL_PATH}")"

[[ "${SEED}" =~ ^[0-9]+$ ]] || { echo "SEED must be a non-negative integer" >&2; exit 1; }

DATASETS=(
    "gsm8k 64 8192 ${PROJECT_DIR}/data/gsm8k_boxed/test.parquet"
    "amc23 64 8192 ${PROJECT_DIR}/data/amc23_boxed/test.parquet"
    "aime24 64 8192 ${PROJECT_DIR}/data/aime24_boxed/test.parquet"
    "aime25 64 8192 ${PROJECT_DIR}/data/aime25_boxed/test.parquet"
    "math_500 64 8192 ${PROJECT_DIR}/data/math_500_boxed/test.parquet"
    "minerva_math 64 8192 ${PROJECT_DIR}/data/minerva_math_boxed/test.parquet"
)

VERIFY_ARGS=(
    "${MODEL_PATH}"
    --dp-size "${SLURM_NTASKS}"
    --seed "${SEED}"
    --show-progress
)
for dataset_spec in "${DATASETS[@]}"; do
    read -r DATASET K MAX_TOKENS DATASET_PATH <<<"${dataset_spec}"
    echo "Queueing ${DATASET}: k=${K}, max_tokens=${MAX_TOKENS}, path=${DATASET_PATH}"
    VERIFY_ARGS+=(
        --dataset-spec
        "${DATASET}:${K}:${MAX_TOKENS}:${DATASET_PATH}"
    )
done

srun python3 "${PROJECT_DIR}/verify.py" "${VERIFY_ARGS[@]}"
python3 "${PROJECT_DIR}/verify.py" "${VERIFY_ARGS[@]}" --aggregate-only


# LM_EVAL_TASKS="hellaswag,mmlu_flan_cot_zeroshot,ifeval"
# LM_EVAL_OUTPUT_DIR="${MODEL_PATH}/evals_${SEED}/lm_eval"
# mkdir -p "${LM_EVAL_OUTPUT_DIR}"

# srun --nodes=1 --ntasks=1 --gpus-per-task=1 \
#     python3 -m lm_eval run \
#     --model vllm \
#     --model_args "pretrained=${MODEL_PATH},tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.75,seed=${SEED}" \
#     --gen_kwargs "n=8,temperature=1.0,top_p=0.8,do_sample=True,max_gen_toks=8192" \
#     --tasks "${LM_EVAL_TASKS}" \
#     --apply_chat_template \
#     --batch_size auto \
#     --seed "${SEED}" \
#     --output_path "${LM_EVAL_OUTPUT_DIR}/results.json"
