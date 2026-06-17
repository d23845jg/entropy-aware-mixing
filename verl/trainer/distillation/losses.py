# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
from tensordict import TensorDict

from verl.base_config import BaseConfig
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig, DistillationConfig, DistillationLossConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding

DistillationLossFn = Callable[
    [
        ActorConfig,  # actor_config
        DistillationConfig,  # distillation_config
        dict,  # model_output
        TensorDict,  # micro batch input
    ],
    tuple[torch.Tensor, dict[str, Any]],
]


def is_distillation_enabled(config: Optional[DistillationConfig]) -> bool:
    """Check if distillation is enabled based on the provided configuration."""
    if config is None:
        return False
    return config.enabled


@dataclass
class DistillationLossSettings(BaseConfig):
    """
    Settings for a distillation loss function to be registered.

    Args:
        names (str | list[str]): Name(s) to register the distillation loss function under.
        use_topk (bool): Whether the loss function uses top-k log probabilities.
        use_estimator (bool): Whether the loss function uses single-sample KL estimators.
    """

    names: str | list[str] = field(default_factory=list)
    use_topk: bool = False
    use_estimator: bool = False

    _mutable_fields = {"names"}

    def __post_init__(self):
        self.names = [self.names] if isinstance(self.names, str) else self.names
        if sum([self.use_topk, self.use_estimator]) != 1:
            raise ValueError(
                f"Expected only one of use_estimator, use_topk, but got {self.use_estimator=}, {self.use_topk=}."
            )


DISTILLATION_LOSS_REGISTRY: dict[str, DistillationLossFn] = {}
DISTILLATION_SETTINGS_REGISTRY: dict[str, DistillationLossSettings] = {}


def register_distillation_loss(
    loss_settings: DistillationLossSettings,
) -> Callable[[DistillationLossFn], DistillationLossFn]:
    """Register a distillation loss function with the given name."""

    def decorator(func: DistillationLossFn) -> DistillationLossFn:
        for name in loss_settings.names:
            if name in DISTILLATION_LOSS_REGISTRY:
                raise ValueError(
                    f"Distillation loss function with name '{name}' is already registered."
                )
            DISTILLATION_LOSS_REGISTRY[name] = func
            DISTILLATION_SETTINGS_REGISTRY[name] = loss_settings
        return func

    return decorator


def get_distillation_loss_fn(loss_name: str) -> DistillationLossFn:
    """Get the distillation loss function with a given name."""
    if loss_name not in DISTILLATION_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_LOSS_REGISTRY.keys())}"
        )
    return DISTILLATION_LOSS_REGISTRY[loss_name]


def get_distillation_loss_settings(loss_name: str) -> DistillationLossSettings:
    """Get the distillation loss settings with a given name."""
    if loss_name not in DISTILLATION_SETTINGS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_SETTINGS_REGISTRY.keys())}"
        )
    return DISTILLATION_SETTINGS_REGISTRY[loss_name]


def normalize_log_probs(log_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Renormalize log-probabilities along the last dimension."""
    log_probs = log_probs.float()
    normalized_log_probs = log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)
    return normalized_log_probs, normalized_log_probs.exp()


def compute_distillation_loss_range(
    distillation_losses: torch.Tensor, response_mask: torch.Tensor
) -> dict[str, Metric]:
    """Compute min and max distillation loss over valid response tokens."""
    distillation_losses_response = distillation_losses[response_mask.bool()]
    return {
        "distillation/loss_min": Metric(
            AggregationType.MIN, distillation_losses_response.min()
        ),
        "distillation/loss_max": Metric(
            AggregationType.MAX, distillation_losses_response.max()
        ),
    }


def _compute_support_mass_metrics(
    student_mass: torch.Tensor,
    teacher_mass: torch.Tensor,
    response_mask_bool: torch.Tensor,
    metric_prefix: str = "",
) -> dict[str, Any]:
    """Aggregate teacher-support overlap metrics over valid response tokens."""
    assert student_mass.shape == teacher_mass.shape == response_mask_bool.shape
    valid_student_mass = student_mass[response_mask_bool]
    valid_teacher_mass = teacher_mass[response_mask_bool]
    metric_name = f"distillation/{metric_prefix}"
    eps = torch.finfo(valid_teacher_mass.dtype).eps
    valid_mass_ratio = valid_student_mass / valid_teacher_mass.clamp_min(eps)
    metrics: dict[str, Any] = {
        f"{metric_name}student_mass": valid_student_mass.mean().item(),
        f"{metric_name}teacher_mass": valid_teacher_mass.mean().item(),
        f"{metric_name}student_teacher_mass_ratio": valid_mass_ratio.mean().item(),
    }

    return metrics


def _compute_scalar_metrics(
    metric_name: str,
    values: torch.Tensor,
    response_mask_bool: torch.Tensor,
) -> dict[str, Any]:
    """Aggregate a token-level scalar metric."""
    assert values.shape == response_mask_bool.shape
    valid_values = values[response_mask_bool]
    return {metric_name: valid_values.mean().item()}


def compute_topk_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    data: TensorDict,
    student_logits: torch.Tensor,
    data_format: str,
) -> torch.Tensor:
    """Compute the topk loss in logit processor.

    Returns:
    - distillation_losses: (bsz, seqlen/cp_size)
    - student_mass: (bsz, seqlen/cp_size)
    - teacher_mass: (bsz, seqlen/cp_size)
    """
    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    match config.strategy:
        case "fsdp":
            import verl.trainer.distillation.fsdp.losses as fsdp_losses

            distillation_loss_fn = fsdp_losses.compute_forward_kl_topk
        case "megatron":
            if (
                loss_config.teacher_topk is not None
                and loss_config.teacher_topk != loss_config.topk
            ):
                raise NotImplementedError(
                    "Decoupled teacher_topk and topk support is currently implemented only for the FSDP path."
                )
            import verl.trainer.distillation.megatron.losses as megatron_losses

            distillation_loss_fn = megatron_losses.compute_forward_kl_topk
        case _:
            raise NotImplementedError(f"Unsupported strategy: {config.strategy=}")

    outputs = distillation_loss_fn(
        student_logits=student_logits,
        teacher_topk_log_probs=data["teacher_logprobs"],
        teacher_topk_ids=data["teacher_ids"],
        config=distillation_config,
        data_format=data_format,
    )

    expected_shape = student_logits.shape[:2]
    for k, v in outputs.items():
        assert v.shape == expected_shape, (
            f"Expected shape {expected_shape}, but got {v.shape} for {k=}."
        )

    return outputs


def _compute_forward_kl_tensors_and_metrics(
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load forward-KL tensors from model output and compute shared metrics."""
    if "distillation_losses" not in model_output:
        raise ValueError(
            "Forward-KL requires model_output['distillation_losses'] from the top-k logits processor."
        )
    if "student_mass" not in model_output or "teacher_mass" not in model_output:
        raise ValueError(
            "Forward-KL requires top-k mass statistics from the logits processor."
        )

    distillation_losses = no_padding_2_padding(
        model_output["distillation_losses"], data
    )
    student_mass = no_padding_2_padding(model_output["student_mass"], data)
    teacher_mass = no_padding_2_padding(model_output["teacher_mass"], data)
    response_mask_bool = data["response_mask"].bool()
    assert (
        distillation_losses.shape
        == student_mass.shape
        == teacher_mass.shape
        == response_mask_bool.shape
    )

    forward_kl = no_padding_2_padding(
        model_output.get("forward_kl", model_output["distillation_losses"]), data
    ).clamp_min(0.0)
    metrics = _compute_scalar_metrics(
        "distillation/forward_kl",
        forward_kl,
        response_mask_bool,
    )
    if "topk_token_overlap" in model_output:
        topk_token_overlap = no_padding_2_padding(
            model_output["topk_token_overlap"], data
        )
        metrics.update(
            _compute_scalar_metrics(
                "distillation/topk_token_overlap",
                topk_token_overlap,
                response_mask_bool,
            )
        )
    has_update_support = (
        "update_student_mass" in model_output and "update_teacher_mass" in model_output
    )
    metrics.update(
        _compute_support_mass_metrics(
            student_mass,
            teacher_mass,
            response_mask_bool,
        )
    )
    if has_update_support:
        # When teacher_topk > topk, update mass is most meaningful on the
        # smaller support that actually receives the top-k update.
        update_student_mass = no_padding_2_padding(
            model_output["update_student_mass"], data
        )
        update_teacher_mass = no_padding_2_padding(
            model_output["update_teacher_mass"], data
        )
        metrics.update(
            _compute_support_mass_metrics(
                update_student_mass,
                update_teacher_mass,
                response_mask_bool,
                metric_prefix="update_",
            )
        )
    # Top-k teacher distributions may be unnormalized, so the truncated KL can
    # be slightly negative; keep the previous optimization behavior.
    distillation_losses = distillation_losses.clamp_min(0.0)
    return distillation_losses, metrics


def distillation_ppo_loss(
    config: ActorConfig,
    distillation_config: Optional[DistillationConfig],
    model_output: dict = None,
    data: TensorDict = None,
    dp_group=None,
    student_logits: torch.Tensor = None,
    data_format: str = "thd",
):
    """Loss function used both for logit processor and final policy loss.
    - student_logits is not None, compute the topk loss in logit processor.
    - student_logits is None, compute final policy loss.

    [split sequence across sp/cp groups]
                   |
    [model forward and output logits: (bsz, seqlen/cp_size, vocab_size/tp_size)]
                   |
    [logits processor compute topk loss: (bsz, seqlen/cp_size)]
                   |
    [all gather topk loss across sp/cp groups: (bsz, seqlen)]
                   |
    [combine topk loss with policy loss]

    Args:
        config: Actor configuration.
        distillation_config: Distillation configuration.
        model_output: Model output, including log_probs, entropy.
        data: Micro input batch, contains
          - teacher_logprobs: (bsz, seqlen, topk)
          - teacher_ids: (bsz, seqlen, topk)
        student_logits: (bsz, seqlen/cp_size, vocab_size/tp_size).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - student_logits is not None, return the topk loss tensor (bsz, seqlen/cp_size).
    - student_logits is None, return the final policy loss scalar and metrics.
    """

    # Called as logits processor
    if student_logits is not None:
        return compute_topk_loss(
            config,
            distillation_config,
            data,
            student_logits,
            data_format,
        )

    # Called as final policy loss
    distillation_loss_config = distillation_config.distillation_loss
    distill_loss, distill_metrics = distillation_loss(
        config, distillation_config, model_output, data
    )
    policy_loss, policy_metrics = ppo_loss(config, model_output, data, dp_group)
    if not distillation_loss_config.use_task_rewards:
        policy_loss = 0.0

    # Combine distillation with policy loss
    policy_metrics.update(distill_metrics)
    distillation_loss_coef = (
        distillation_loss_config.distillation_loss_coef
        if distillation_loss_config.use_task_rewards
        else 1.0
    )
    policy_loss += distill_loss * distillation_loss_coef
    policy_metrics["distillation/loss"] = Metric(
        value=distill_loss, aggregation=AggregationType.SUM
    )

    return policy_loss, policy_metrics


def distillation_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics.

    Returns:
    - distillation_loss: Aggregated distillation loss scalar.
    - distillation_metrics: Dictionary of metrics.
    """
    assert distillation_config is not None
    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_loss_fn = get_distillation_loss_fn(loss_config.loss_mode)
    distillation_losses, distillation_metrics = distillation_loss_fn(
        config=config,
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
    )
    response_mask = data["response_mask"]
    loss_agg_mode = config.loss_agg_mode

    distillation_metrics.update(
        compute_distillation_loss_range(
            distillation_losses=distillation_losses, response_mask=response_mask
        )
    )
    if loss_config.loss_max_clamp is not None:
        # clamping min is for k1 loss which can be negative
        distillation_losses = distillation_losses.clamp(
            min=-loss_config.loss_max_clamp, max=loss_config.loss_max_clamp
        )

    if loss_config.use_policy_gradient:
        # Use negative distillation loss as reward, as done by https://thinkingmachines.ai/blog/on-policy-distillation/.
        policy_loss_fn = get_policy_loss_fn(loss_config.policy_loss_mode)
        for k, v in config.global_batch_info.items():
            loss_config.global_batch_info[k] = v
        log_prob = no_padding_2_padding(model_output["log_probs"], data)
        old_log_prob = data["old_log_probs"]
        rollout_is_weights = data.get("rollout_is_weights", None)
        distillation_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=-distillation_losses.detach(),
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=loss_config,
            rollout_is_weights=rollout_is_weights,
        )
        pg_metrics = {
            f"distillation/{k[len('actor/') :]}": v for k, v in pg_metrics.items()
        }
        distillation_metrics.update(pg_metrics)
    else:
        # Directly backpropagate distillation loss as a supervised loss, as in https://arxiv.org/abs/2306.13649.
        distillation_loss = agg_loss(
            loss_mat=distillation_losses,
            loss_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            **config.global_batch_info,
        )

    return distillation_loss, distillation_metrics


@register_distillation_loss(
    DistillationLossSettings(names="forward_kl_topk", use_topk=True)
)  # type: ignore[arg-type]
def compute_forward_kl_topk(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute forward KL distillation loss and related metrics using top-k log probabilities.

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    return _compute_forward_kl_tensors_and_metrics(
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
    )


@register_distillation_loss(
    DistillationLossSettings(
        names=["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"], use_estimator=True
    )
)  # type: ignore[arg-type]
def compute_distillation_loss_reverse_kl_estimator(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics using single-sample KL estimators.

    Uses the kl_penalty function from core_algos which supports various KL divergence
    estimators: "kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3".
    When entropy-aware mixing is enabled, this sampled-logprob estimator path only
    supports the linear "kl"/"k1" objective. Nonlinear estimators would require
    a normalized mixed distribution over token support, which is not available here.

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = data.get("teacher_sampled_logprobs", None)
    if teacher_log_probs is None:
        teacher_log_probs = data["teacher_logprobs"]
    teacher_log_probs = no_padding_2_padding(teacher_log_probs, data)
    if teacher_log_probs.ndim > student_log_probs.ndim:
        teacher_log_probs = teacher_log_probs.squeeze(-1)
    response_mask_bool = data["response_mask"].bool()
    assert (
        teacher_log_probs.shape == student_log_probs.shape == response_mask_bool.shape
    )

    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    metrics = {}
    if "student_mass" in model_output and "teacher_mass" in model_output:
        student_mass = no_padding_2_padding(model_output["student_mass"], data)
        teacher_mass = no_padding_2_padding(model_output["teacher_mass"], data)
        metrics.update(
            _compute_support_mass_metrics(
                student_mass,
                teacher_mass,
                response_mask_bool,
            )
        )
    if "forward_kl" in model_output:
        forward_kl = no_padding_2_padding(
            model_output["forward_kl"], data
        ).clamp_min(0.0)
        metrics.update(
            _compute_scalar_metrics(
                "distillation/forward_kl",
                forward_kl,
                response_mask_bool,
            )
        )
    if "topk_token_overlap" in model_output:
        topk_token_overlap = no_padding_2_padding(
            model_output["topk_token_overlap"], data
        )
        metrics.update(
            _compute_scalar_metrics(
                "distillation/topk_token_overlap",
                topk_token_overlap,
                response_mask_bool,
            )
        )

    distillation_losses = kl_penalty(
        logprob=student_log_probs,
        ref_logprob=teacher_log_probs,
        kl_penalty=loss_config.loss_mode,
    )
    # Since k1 can be negative, log the mean absolute loss.
    metrics["distillation/abs_loss"] = Metric(
        AggregationType.MEAN, distillation_losses[response_mask_bool].abs().mean()
    )
    return distillation_losses, metrics
