#!/usr/bin/env bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --container-writable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --time=12:00:00
#SBATCH --job-name=eopd-train
#SBATCH --output=./logs/%A/slurm-%A.out
#SBATCH --error=./logs/%A/slurm-%A.err
#SBATCH --environment=entropy-aware-mixing

set -euo pipefail

# ----- Edit these constants -----
TEACHER_MODEL="Qwen/Qwen3-8B"
STUDENT_MODEL="Qwen/Qwen3-1.7B-Base"
DATA_FILE="./data/math_boxed/train.parquet"
TEST_FILES='[./data/amc23_boxed/test.parquet,./data/aime24_boxed/test.parquet,./data/aime25_boxed/test.parquet,./data/math_500_boxed/test.parquet,./data/minerva_math_boxed/test.parquet,./data/gsm8k_500_boxed/test.parquet]'
SEED=44

TOP_P=1.0
DISTILLATION_TOPK=64
DISTILLATION_TEACHER_TOPK=128
EOPD_FORWARD_KL_COEF=1.0
EOPD_FORWARD_KL_ENTROPY_THRESHOLD=0.8

cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

ALGORITHM="eopd"
OUTPUT_DIR="${SLURM_SUBMIT_DIR}/opd_results/${TEACHER_MODEL##*/}+${STUDENT_MODEL##*/}/${ALGORITHM}"

mkdir -p "${OUTPUT_DIR}"

HYDRA_ARGS=(
    data.train_files="${DATA_FILE}"
    data.val_files="${TEST_FILES}"
    data.max_prompt_length=1024
    data.max_response_length=4096
    data.train_batch_size=128
    data.seed="${SEED}"
    data.filter_overlong_prompts=true
    actor_rollout_ref.model.path="${STUDENT_MODEL}"
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
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((5 * (1024 + 4096)))
    actor_rollout_ref.actor.use_dynamic_bsz=true
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((10 * (1024 + 4096)))
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.enforce_eager=false
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2
    actor_rollout_ref.rollout.max_model_len=$((1024 + 4096 + 1))
    actor_rollout_ref.rollout.max_num_batched_tokens=$((1024 + 4096 + 1))
    actor_rollout_ref.rollout.max_num_seqs=8
    actor_rollout_ref.rollout.top_p="${TOP_P}"
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
    distillation.teacher_model.inference.max_model_len=$((1024 + 4096 + 1))
    distillation.teacher_model.inference.max_num_batched_tokens=$((1024 + 4096 + 1))
    distillation.teacher_model.inference.max_num_seqs=8
    distillation.distillation_loss.loss_mode=k1
    distillation.distillation_loss.topk="${DISTILLATION_TOPK}"
    distillation.distillation_loss.teacher_topk="${DISTILLATION_TEACHER_TOPK}"
    distillation.distillation_loss.use_task_rewards=false
    distillation.distillation_loss.use_policy_gradient=true
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
    distillation.distillation_loss.eopd_forward_kl_coef="${EOPD_FORWARD_KL_COEF}"
    distillation.distillation_loss.eopd_forward_kl_entropy_threshold="${EOPD_FORWARD_KL_ENTROPY_THRESHOLD}"
    trainer.logger='["console","wandb"]'
    trainer.project_name=entropy_aware_mixing
    trainer.experiment_name="${TEACHER_MODEL##*/}+${STUDENT_MODEL##*/}-loss_${ALGORITHM}"
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
