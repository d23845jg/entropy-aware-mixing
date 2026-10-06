#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --time=12:00:00
#SBATCH --job-name=opd-train
#SBATCH --output=./logs/slurm-%A.out
#SBATCH --error=./logs/slurm-%A.err

set -euo pipefail

# ----- Edit these constants -----
TEACHER_MODEL="Qwen/Qwen3-8B"
STUDENT_MODEL="Qwen/Qwen3-1.7B-Base"
DATA_FILE="./data/math_boxed/train.parquet"
SEED=44
ALGORITHM="fkl"  # opd or fkl
MIXING_MODE="entropy"  # none, fixed, or entropy
MIXTURE_ALPHA=1.0  # -1.0 is arithmetic; 1.0 is geometric
MIXTURE_LAMBDA=0.1  # teacher weight used by fixed mode
ENTROPY_TRANSFORM="linear"  # linear, sqrt, sqrt2, or sq

DISTILLATION_TOPK=64
DISTILLATION_TEACHER_TOPK=64

cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

case "${ALGORITHM}" in
    opd)
        DISTILLATION_LOSS_MODE="k1"
        USE_POLICY_GRADIENT=true
        ;;
    fkl)
        DISTILLATION_LOSS_MODE="forward_kl_topk"
        USE_POLICY_GRADIENT=false
        ;;
    *)
        printf '%s\n' "Unsupported main OPD algorithm: ${ALGORITHM}" >&2
        exit 1
        ;;
esac

case "${MIXING_MODE}" in
    none|fixed|entropy) ;;
    *)
        printf '%s\n' "Unsupported MIXING_MODE: ${MIXING_MODE}" >&2
        exit 1
        ;;
esac

case "${ENTROPY_TRANSFORM}" in
    linear|sqrt|sqrt2|sq) ;;
    *)
        printf '%s\n' "Unsupported ENTROPY_TRANSFORM: ${ENTROPY_TRANSFORM}" >&2
        exit 1
        ;;
esac

OUTPUT_DIR="${SLURM_SUBMIT_DIR}/opd_results/${TEACHER_MODEL##*/}+${STUDENT_MODEL##*/}/${ALGORITHM}/mix_${MIXING_MODE}"
if [ "${MIXING_MODE}" != "none" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}/alpha_${MIXTURE_ALPHA}"
fi
if [ "${MIXING_MODE}" = "fixed" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}/lambda_${MIXTURE_LAMBDA}"
elif [ "${MIXING_MODE}" = "entropy" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}/transform_${ENTROPY_TRANSFORM}"
fi
OUTPUT_DIR="${OUTPUT_DIR}/seed_${SEED}"
mkdir -p "${OUTPUT_DIR}"

# verl requires a non-empty validation dataset even when validation is disabled.
HYDRA_ARGS=(
    data.train_files="${DATA_FILE}"
    data.val_files="${DATA_FILE}"
    data.val_max_samples=1
    data.max_prompt_length=1024
    data.max_response_length=4096
    data.train_batch_size=128
    data.seed="${SEED}"
    data.filter_overlong_prompts=true
    +data.apply_chat_template_kwargs.enable_thinking=false
    actor_rollout_ref.model.path="${STUDENT_MODEL}"
    actor_rollout_ref.model.enable_gradient_checkpointing=true
    actor_rollout_ref.actor.checkpoint.save_contents='["hf_model"]'
    actor_rollout_ref.model.use_remove_padding=true
    actor_rollout_ref.actor.optim.lr=3.0e-6
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
    actor_rollout_ref.actor.fsdp_config.param_offload=false
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false
    actor_rollout_ref.actor.ppo_mini_batch_size=128
    actor_rollout_ref.actor.data_loader_seed="${SEED}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=25600
    actor_rollout_ref.actor.use_dynamic_bsz=true
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=51200
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.enforce_eager=false
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2
    actor_rollout_ref.rollout.max_model_len=5121
    actor_rollout_ref.rollout.max_num_batched_tokens=5121
    actor_rollout_ref.rollout.max_num_seqs=8
    actor_rollout_ref.rollout.n=1
    algorithm.adv_estimator=grpo
    distillation.enabled=true
    distillation.teacher_model.n_gpus_per_node=4
    distillation.teacher_model.nnodes=1
    distillation.teacher_model.enable_resource_pool=false
    distillation.teacher_model.model_path="${TEACHER_MODEL}"
    distillation.teacher_model.inference.tensor_model_parallel_size=1
    distillation.teacher_model.inference.name=vllm
    distillation.teacher_model.inference.gpu_memory_utilization=0.3
    distillation.teacher_model.inference.enforce_eager=false
    distillation.teacher_model.inference.max_model_len=5121
    distillation.teacher_model.inference.max_num_batched_tokens=5121
    distillation.teacher_model.inference.max_num_seqs=8
    distillation.distillation_loss.loss_mode="${DISTILLATION_LOSS_MODE}"
    distillation.distillation_loss.topk="${DISTILLATION_TOPK}"
    distillation.distillation_loss.teacher_topk="${DISTILLATION_TEACHER_TOPK}"
    distillation.distillation_loss.use_task_rewards=false
    distillation.distillation_loss.use_policy_gradient="${USE_POLICY_GRADIENT}"
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
    distillation.distillation_loss.mixing_mode="${MIXING_MODE}"
    distillation.distillation_loss.mixture_alpha="${MIXTURE_ALPHA}"
    distillation.distillation_loss.mixture_lambda="${MIXTURE_LAMBDA}"
    distillation.distillation_loss.entropy_transform="${ENTROPY_TRANSFORM}"
    distillation.distillation_loss.entropy_top_k=32
    trainer.logger='["console","wandb"]'
    trainer.project_name=entropy_aware_mixing
    trainer.experiment_name="${TEACHER_MODEL##*/}+${STUDENT_MODEL##*/}-loss_${ALGORITHM}-mix_${MIXING_MODE}-alpha_${MIXTURE_ALPHA}-lambda_${MIXTURE_LAMBDA}-transform_${ENTROPY_TRANSFORM}"
    trainer.default_local_dir="${OUTPUT_DIR}"
    trainer.nnodes=1
    trainer.n_gpus_per_node=4
    trainer.total_epochs=3
    trainer.save_freq=174
    trainer.test_freq=-1
    trainer.log_val_generations=-1
    trainer.val_before_train=false
    trainer.use_legacy_worker_impl=disable
)

HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo "${HYDRA_ARGS[@]}"
