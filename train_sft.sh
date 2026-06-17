#!/usr/bin/env bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --container-writable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --time=01:00:00
#SBATCH --job-name=eaft-train
#SBATCH --output=./logs/%A/slurm-%A.out
#SBATCH --error=./logs/%A/slurm-%A.err
#SBATCH --environment=entropy-aware-mixing

set -euo pipefail

# ----- Edit these constants -----
TEACHER_MODEL="Qwen/Qwen3-8B"
STUDENT_MODEL="Qwen/Qwen3-0.6B"
ALGORITHM="standard_target_eaft"

EAFT_ALPHA=1.0
EAFT_K=32

cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

TRAIN_FILES="${SLURM_SUBMIT_DIR}/generated_traces/${TEACHER_MODEL##*/}+${STUDENT_MODEL##*/}/${ALGORITHM}/train.parquet"
OUTPUT_DIR="${SLURM_SUBMIT_DIR}/sft_results/${TEACHER_MODEL##*/}+${STUDENT_MODEL##*/}/${ALGORITHM}"

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
    data.max_length=4609
    data.max_token_len_per_gpu=$((8 * 4609))
    data.truncation=error
    data.train_batch_size=128
    data.micro_batch_size_per_gpu=32
    optim.lr=5.0e-6
    optim.lr_scheduler_type=cosine
    optim.lr_warmup_steps_ratio=0.03
    optim.clip_grad=1.0
    engine.strategy=fsdp
    model.path="${STUDENT_MODEL}"
    checkpoint.save_contents='["hf_model"]'
    use_eaft_loss=true
    eaft_alpha="${EAFT_ALPHA}"
    eaft_k="${EAFT_K}"
    trainer.total_epochs=2
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.n_gpus_per_node=4
    trainer.default_local_dir="${OUTPUT_DIR}"
    trainer.project_name=entropy_aware_mixing
    trainer.experiment_name="${TEACHER_MODEL##*/}+${STUDENT_MODEL##*/}-${ALGORITHM}-SFT"
    trainer.logger='["console","wandb"]'
)

HYDRA_FULL_ERROR=1 torchrun --standalone --nnodes=1 --nproc_per_node=4 "${HYDRA_ARGS[@]}"
