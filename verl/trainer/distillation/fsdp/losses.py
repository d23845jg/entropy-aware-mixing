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

import math
from typing import Optional

import torch
import torch.nn.functional as F

from verl.trainer.distillation.losses import (
    _transform_entropy_lambda,
    normalize_log_probs,
)
from verl.utils.ulysses import (
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
)
from verl.workers.config import DistillationConfig, DistillationLossConfig


def kl_divergence(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between two distributions given their log probabilities."""
    log_p = log_p.float()
    log_q = log_q.float()
    p = log_p.exp()
    kld = p * (log_p - log_q)
    return kld.sum(dim=-1)


def _prepare_teacher_topk_support(
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    student_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(
        0
    )  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)

    assert (
        teacher_topk_log_probs.shape[:2]
        == teacher_topk_ids.shape[:2]
        == student_logits.shape[:2]
    )
    return teacher_topk_log_probs, teacher_topk_ids


def _gather_student_topk_log_probs(
    student_logits: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_topk_log_probs = torch.gather(
        student_log_probs, dim=-1, index=teacher_topk_ids
    )
    student_mass = student_topk_log_probs.detach().exp().sum(dim=-1)
    return student_topk_log_probs, student_mass


def _compute_topk_forward_kl(
    student_topk_log_probs: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    loss_config: DistillationLossConfig,
) -> torch.Tensor:
    if loss_config.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(
            loss_config.log_prob_min_clamp
        )
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(
            loss_config.log_prob_min_clamp
        )
    return kl_divergence(log_q=student_topk_log_probs, log_p=teacher_topk_log_probs)


def _compute_alpha_mixture_log_probs(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    mixing_lambda: torch.Tensor | float,
    mixture_alpha: float,
) -> torch.Tensor:
    """Compute the normalized alpha-mixture with a detached student target."""
    if not math.isfinite(mixture_alpha):
        raise ValueError(f"mixture_alpha must be finite, got {mixture_alpha}.")
    if isinstance(mixing_lambda, (float, int)) and not 0.0 <= mixing_lambda <= 1.0:
        raise ValueError("mixing_lambda must be in [0, 1].")

    student_log_probs = student_log_probs.detach().float()
    teacher_log_probs = teacher_log_probs.detach().float()
    mixing_lambda = torch.as_tensor(
        mixing_lambda,
        dtype=torch.float32,
        device=student_log_probs.device,
    ).detach()
    while mixing_lambda.ndim < student_log_probs.ndim:
        mixing_lambda = mixing_lambda.unsqueeze(-1)

    if abs(mixture_alpha - 1.0) < 1e-6:
        mixed_log_scores = (
            mixing_lambda * teacher_log_probs
            + (1.0 - mixing_lambda) * student_log_probs
        )
    else:
        exponent = (1.0 - mixture_alpha) / 2.0
        teacher_term = torch.log(mixing_lambda) + exponent * teacher_log_probs
        student_term = torch.log1p(-mixing_lambda) + exponent * student_log_probs
        mixed_log_scores = torch.logaddexp(teacher_term, student_term) / exponent
        mixed_log_scores = torch.where(
            mixing_lambda == 0.0,
            student_log_probs,
            torch.where(
                mixing_lambda == 1.0,
                teacher_log_probs,
                mixed_log_scores,
            ),
        )

    return mixed_log_scores - torch.logsumexp(
        mixed_log_scores, dim=-1, keepdim=True
    )


@torch.no_grad()
def _compute_topk_token_overlap(
    student_logits: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    k: int,
) -> torch.Tensor:
    k = min(k, teacher_topk_ids.shape[-1], student_logits.shape[-1])
    if k <= 0:
        return torch.zeros_like(student_logits[..., 0], dtype=torch.float32)

    student_topk_ids = torch.topk(
        student_logits.detach(), k, dim=-1, sorted=False
    ).indices
    teacher_topk_ids = teacher_topk_ids[..., :k].detach()
    return (student_topk_ids.unsqueeze(-1) == teacher_topk_ids.unsqueeze(-2)).any(
        dim=-1
    ).float().mean(dim=-1)


def compute_forward_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
    entropy_aware_lambda: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    del data_format

    teacher_topk_log_probs, teacher_topk_ids = _prepare_teacher_topk_support(
        student_logits=student_logits,
        teacher_topk_log_probs=teacher_topk_log_probs,
        teacher_topk_ids=teacher_topk_ids,
    )

    student_topk_log_probs, student_mass = _gather_student_topk_log_probs(
        student_logits=student_logits, teacher_topk_ids=teacher_topk_ids
    )
    loss_config: DistillationLossConfig = config.distillation_loss
    teacher_mass = teacher_topk_log_probs.detach().exp().sum(dim=-1)
    support_size = teacher_topk_log_probs.shape[-1]
    update_topk = (
        support_size
        if loss_config.topk is None
        else min(loss_config.topk, support_size)
    )
    metrics = {"student_mass": student_mass, "teacher_mass": teacher_mass}
    metrics_only = (
        not loss_config.loss_settings.use_topk and loss_config.eopd_forward_kl_coef <= 0
    )
    metrics["topk_token_overlap"] = _compute_topk_token_overlap(
        student_logits=student_logits,
        teacher_topk_ids=teacher_topk_ids,
        k=update_topk,
    )

    teacher_topk_log_probs_for_loss = teacher_topk_log_probs[..., :update_topk]
    student_topk_log_probs_for_loss = student_topk_log_probs[..., :update_topk]
    if not metrics_only:
        metrics["update_student_mass"] = (
            student_topk_log_probs_for_loss.detach().exp().sum(dim=-1)
        )
        metrics["update_teacher_mass"] = (
            teacher_topk_log_probs_for_loss.detach().exp().sum(dim=-1)
        )

    metrics["forward_kl"] = _compute_topk_forward_kl(
        student_topk_log_probs=student_topk_log_probs_for_loss,
        teacher_topk_log_probs=teacher_topk_log_probs_for_loss,
        loss_config=loss_config,
    ).detach()

    if loss_config.eopd_forward_kl_teacher_entropy_source == "topk_renorm":
        teacher_topk_log_probs_for_loss, _ = normalize_log_probs(
            teacher_topk_log_probs_for_loss
        )

    if loss_config.loss_mode == "taid_topk":
        t = loss_config.taid_t
        if t is None:
            t = loss_config.taid_t_start
        mixed_log_scores = (
            (1.0 - t) * student_topk_log_probs_for_loss.detach()
            + t * teacher_topk_log_probs_for_loss
        )
        taid_log_probs, _ = normalize_log_probs(mixed_log_scores)
        distillation_losses = kl_divergence(
            log_q=student_topk_log_probs_for_loss,
            log_p=taid_log_probs,
        )
    elif loss_config.mixing_mode != "none":
        if loss_config.mixing_mode == "entropy":
            if entropy_aware_lambda is None:
                raise ValueError(
                    "Entropy-aware alpha-mixture distillation requires a per-token lambda."
                )
            mixing_lambda = _transform_entropy_lambda(
                entropy_aware_lambda, loss_config.entropy_transform
            )
        else:
            mixing_lambda = torch.full_like(
                student_mass,
                loss_config.mixture_lambda,
                dtype=torch.float32,
            )
        metrics["mixing_lambda"] = mixing_lambda.detach()
        mixed_topk_log_probs = _compute_alpha_mixture_log_probs(
            student_log_probs=student_topk_log_probs_for_loss,
            teacher_log_probs=teacher_topk_log_probs_for_loss,
            mixing_lambda=mixing_lambda,
            mixture_alpha=loss_config.mixture_alpha,
        )
        if loss_config.log_prob_min_clamp is not None:
            student_topk_log_probs_for_loss = student_topk_log_probs_for_loss.clamp_min(
                loss_config.log_prob_min_clamp
            )
        distillation_losses = kl_divergence(
            log_q=student_topk_log_probs_for_loss,
            log_p=mixed_topk_log_probs,
        )
    else:
        if entropy_aware_lambda is not None:
            raise ValueError(
                "Received an entropy-aware lambda while mixing_mode='none'."
            )
        distillation_losses = _compute_topk_forward_kl(
            student_topk_log_probs=student_topk_log_probs_for_loss,
            teacher_topk_log_probs=teacher_topk_log_probs_for_loss,
            loss_config=loss_config,
        )

    if metrics_only:
        metrics["distillation_losses"] = torch.zeros_like(student_mass)
        return metrics

    metrics["distillation_losses"] = distillation_losses
    return metrics
