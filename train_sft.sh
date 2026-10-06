#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --time=01:00:00
#SBATCH --job-name=sft-train
#SBATCH --output=./logs/slurm-%A.out
#SBATCH --error=./logs/slurm-%A.err

set -euo pipefail

# ----- Edit these constants -----
TEACHER_MODEL="Qwen/Qwen3-8B"
STUDENT_MODEL="Qwen/Qwen3-1.7B"
SEED=44
ALGORITHM="entropy_aware_sd" # standard_target, standard_draft, standard_sd, entropy_aware_sd
MIXTURE_TYPE="geometric" # none, geometric, or convex
ENTROPY_TRANSFORM="linear" # linear, sqrt, sqrt2, sq, or constant
MAX_LENGTH=5120

cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

RUN_SUBDIR="${ALGORITHM}/mix_${MIXTURE_TYPE}"
if [ -n "${ENTROPY_TRANSFORM}" ]; then
    RUN_SUBDIR="${RUN_SUBDIR}/transform_${ENTROPY_TRANSFORM}"
fi
RUN_SUBDIR="${RUN_SUBDIR}/seed_${SEED}"

TRAIN_FILES="${SLURM_SUBMIT_DIR}/generated_traces/${TEACHER_MODEL##*/}+${STUDENT_MODEL##*/}/${RUN_SUBDIR}/train.parquet"
OUTPUT_DIR="${SLURM_SUBMIT_DIR}/sft_results/${TEACHER_MODEL##*/}+${STUDENT_MODEL##*/}/${RUN_SUBDIR}"

[ -f "${TRAIN_FILES}" ] || {
    printf '%s\n' "Training file not found: ${TRAIN_FILES}" >&2
    exit 1
}
mkdir -p "${OUTPUT_DIR}"

HYDRA_ARGS=(
    -m verl.trainer.sft_trainer
    data.train_files="${TRAIN_FILES}"
    data.messages_key=prompt
    data.ignore_input_ids_mismatch=true
    data.enable_thinking_default=false
    data.max_length="${MAX_LENGTH}"
    data.max_token_len_per_gpu=$((8 * MAX_LENGTH))
    data.truncation=error
    data.train_batch_size=128
    data.micro_batch_size_per_gpu=32
    optim.lr=5.0e-6
    optim.lr_scheduler_type=cosine
    optim.lr_warmup_steps_ratio=0.03
    optim.clip_grad=1.0
    engine.strategy=fsdp
    engine.seed="${SEED}"
    model.path="${STUDENT_MODEL}"
    checkpoint.save_contents='["hf_model"]'
    trainer.total_epochs=2
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.n_gpus_per_node=4
    trainer.seed="${SEED}"
    trainer.default_local_dir="${OUTPUT_DIR}"
    trainer.project_name=entropy_aware_mixing
    trainer.experiment_name="${TEACHER_MODEL##*/}+${STUDENT_MODEL##*/}-${RUN_SUBDIR//\//-}-SFT"
    trainer.logger='["console","wandb"]'
)

HYDRA_FULL_ERROR=1 torchrun --standalone --nnodes=1 --nproc_per_node=4 "${HYDRA_ARGS[@]}"
