#!/usr/bin/env python3
"""Teacher-anchored Matryoshka distillation helpers for AGILLM4.3.

This is the asymmetric variant needed for a *new* ~49M popout student trained
from an already-pretrained ~1.1B parent.  It intentionally does not pretend to
be the exact joint-training setup from Godey & Artzi (2026): the parent is
frozen, so there is no shared backward pass and no guarantee of shared KV cache.

The useful Matryoshka ideas that survive this asymmetry are:
  * same-token online distillation from the largest model;
  * alpha_d=0.3 as a conservative default;
  * explicit cross-model agreement/KL logging for speculative decoding;
  * a standalone student checkpoint.

Two guards are added because AGILLM's teacher is already trained and the student
is fresh:
  * teacher-better token gating: distill only where the teacher gives the gold
    next token lower NLL than the student;
  * global catch-up gating: fade alpha as held-out student CE beats teacher CE.

The module is deliberately trainer-agnostic.  It can be imported by the current
plain-PyTorch 49M popout trainer without changing checkpoint format.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class DistillConfig:
    """Configuration for frozen-parent -> new-student distillation."""

    alpha: float = 0.30
    temperature: float = 1.0
    teacher_every: int = 4
    topk: int = 32
    teacher_better_only: bool = True
    teacher_advantage_nats: float = 0.0
    catchup_margin_ce: float = 0.50
    min_alpha: float = 0.0
    ignore_index: int = -100

    def validate(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        if self.teacher_every < 1:
            raise ValueError("teacher_every must be >= 1")
        if self.topk < 1:
            raise ValueError("topk must be >= 1")
        if self.catchup_margin_ce <= 0.0:
            raise ValueError("catchup_margin_ce must be > 0")
        if not 0.0 <= self.min_alpha <= self.alpha:
            raise ValueError("min_alpha must be in [0, alpha]")


class DistillController:
    """Tracks the global held-out quality guard for the frozen teacher.

    While the student is worse than the teacher, the paper-style alpha is used.
    Once the student becomes better, alpha is linearly faded to ``min_alpha``
    over ``catchup_margin_ce`` nats of CE advantage.  This avoids turning a
    mature-but-imperfect parent into a permanent ceiling for a new 49M model.
    """

    def __init__(self, cfg: DistillConfig):
        cfg.validate()
        self.cfg = cfg
        self.teacher_ce: Optional[float] = None
        self.student_ce: Optional[float] = None

    def observe_eval(self, *, teacher_ce: float, student_ce: float) -> float:
        if teacher_ce <= 0 or student_ce <= 0:
            raise ValueError("cross-entropies must be positive")
        self.teacher_ce = float(teacher_ce)
        self.student_ce = float(student_ce)
        return self.effective_alpha

    @property
    def effective_alpha(self) -> float:
        base = self.cfg.alpha
        if self.teacher_ce is None or self.student_ce is None:
            return base
        advantage = self.teacher_ce - self.student_ce
        if advantage <= 0.0:
            return base
        frac = min(1.0, advantage / self.cfg.catchup_margin_ce)
        return base + frac * (self.cfg.min_alpha - base)


def should_query_teacher(step: int, cfg: DistillConfig) -> bool:
    """Deterministic compute throttle for the expensive 1.1B teacher forward."""
    cfg.validate()
    return int(step) % cfg.teacher_every == 0


def _flatten_logits_targets(
    logits: torch.Tensor, targets: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim < 2:
        raise ValueError("logits must have a vocabulary dimension")
    if tuple(logits.shape[:-1]) != tuple(targets.shape):
        raise ValueError(
            f"target shape {tuple(targets.shape)} must equal logits prefix "
            f"{tuple(logits.shape[:-1])}"
        )
    return logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)


def token_cross_entropy(
    student_logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-valid-token CE and the flattened validity mask."""
    s, y = _flatten_logits_targets(student_logits, targets)
    valid = y.ne(ignore_index)
    safe_y = y.masked_fill(~valid, 0)
    ce = F.cross_entropy(s.float(), safe_y, reduction="none")
    return ce, valid


def _gold_nll(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int) -> torch.Tensor:
    flat, y = _flatten_logits_targets(logits, targets)
    valid = y.ne(ignore_index)
    safe_y = y.masked_fill(~valid, 0)
    logz = torch.logsumexp(flat.float(), dim=-1)
    gold = flat.float().gather(-1, safe_y[:, None]).squeeze(-1)
    nll = logz - gold
    return nll.masked_fill(~valid, 0.0)


def topk_residual_distill_ce(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    topk: int = 32,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Per-token soft CE using teacher top-k plus one residual-probability bucket.

    The residual bucket retains the teacher's probability mass outside the top-k,
    making this much less lossy than simply renormalizing the top-k.  Full teacher
    logits only need to exist for the current chunk and can be freed immediately.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"student/teacher logits must match, got {student_logits.shape} and "
            f"{teacher_logits.shape}"
        )
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")
    vocab = student_logits.shape[-1]
    k = min(int(topk), int(vocab))

    s = student_logits.reshape(-1, vocab).float() / temperature
    t = teacher_logits.reshape(-1, vocab).float().detach() / temperature

    # Exact full-vocabulary form. Besides avoiding a pointless top-k call, this
    # makes the k==V case a useful correctness reference for tests.
    if k == vocab:
        t_logp = F.log_softmax(t, dim=-1)
        s_logp = F.log_softmax(s, dim=-1)
        return -(t_logp.exp() * s_logp).sum(dim=-1) * (temperature * temperature)

    # Teacher is frozen even if the caller forgot inference_mode/no_grad.
    t_logz = torch.logsumexp(t, dim=-1, keepdim=True)
    s_logz = torch.logsumexp(s, dim=-1, keepdim=True)

    t_top, idx = torch.topk(t, k=k, dim=-1)
    s_top = torch.gather(s, -1, idx)

    p_top = torch.exp(t_top - t_logz)
    log_q_top = s_top - s_logz
    q_top = torch.exp(log_q_top)

    # Aggregate the long tail into one bucket instead of discarding it.
    eps = torch.finfo(torch.float32).eps
    p_other = (1.0 - p_top.sum(dim=-1)).clamp(min=0.0, max=1.0)
    q_other = (1.0 - q_top.sum(dim=-1)).clamp(min=eps, max=1.0)

    ce = -(p_top * log_q_top).sum(dim=-1) - p_other * torch.log(q_other)
    # Standard temperature-distillation gradient scaling. At T=1 this is a no-op.
    return ce * (temperature * temperature)


def teacher_better_mask(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = -100,
    advantage_nats: float = 0.0,
) -> torch.Tensor:
    """Select tokens where the frozen teacher currently knows more than student."""
    with torch.no_grad():
        s_nll = _gold_nll(student_logits.detach(), targets, ignore_index)
        t_nll = _gold_nll(teacher_logits.detach(), targets, ignore_index)
        _, y = _flatten_logits_targets(student_logits, targets)
        valid = y.ne(ignore_index)
        return valid & (t_nll + float(advantage_nats) < s_nll)


def agreement_metrics(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
    *,
    ignore_index: int = -100,
) -> Dict[str, float]:
    """Cheap alignment metrics; top-1 agreement is speculative-decoding relevant."""
    with torch.no_grad():
        s = student_logits.reshape(-1, student_logits.shape[-1]).float()
        t = teacher_logits.reshape(-1, teacher_logits.shape[-1]).float()
        if s.shape != t.shape:
            raise ValueError("student/teacher logits must match")
        if targets is not None:
            y = targets.reshape(-1)
            valid = y.ne(ignore_index)
        else:
            valid = torch.ones(s.shape[0], dtype=torch.bool, device=s.device)
        if not bool(valid.any()):
            return {"top1_agreement": 0.0, "teacher_student_kl": 0.0}
        sv = s[valid]
        tv = t[valid]
        agree = sv.argmax(-1).eq(tv.argmax(-1)).float().mean()
        t_logp = F.log_softmax(tv, dim=-1)
        s_logp = F.log_softmax(sv, dim=-1)
        t_prob = t_logp.exp()
        kl = (t_prob * (t_logp - s_logp)).sum(-1).mean()
        return {
            "top1_agreement": float(agree.item()),
            "teacher_student_kl": float(kl.item()),
        }


def teacher_anchored_loss(
    student_logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    cfg: DistillConfig,
    teacher_logits: Optional[torch.Tensor] = None,
    alpha: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute CE-only or guarded Matryoshka-style CE + soft distillation.

    On teacher steps the objective follows the paper's convex combination:
        (1-alpha) * L_ce + alpha * L_distill
    with additional token/global guards for the pretrained-teacher asymmetry.
    """
    cfg.validate()
    ce_tok, valid = token_cross_entropy(
        student_logits, targets, ignore_index=cfg.ignore_index
    )
    if not bool(valid.any()):
        raise ValueError("batch contains no valid target tokens")
    ce = ce_tok[valid].mean()

    stats: Dict[str, float] = {
        "loss_ce": float(ce.detach().item()),
        "alpha": 0.0,
        "teacher_used": 0.0,
        "teacher_better_fraction": 0.0,
        "valid_tokens": float(valid.sum().item()),
        "distill_tokens": 0.0,
    }
    if teacher_logits is None:
        return ce, stats

    if teacher_logits.shape != student_logits.shape:
        raise ValueError("teacher logits must have exactly the student logit shape")
    a = cfg.alpha if alpha is None else float(alpha)
    if not 0.0 <= a <= 1.0:
        raise ValueError("effective alpha must be in [0, 1]")
    if a == 0.0:
        return ce, stats

    distill_tok = topk_residual_distill_ce(
        student_logits,
        teacher_logits,
        topk=cfg.topk,
        temperature=cfg.temperature,
    )
    if cfg.teacher_better_only:
        dmask = teacher_better_mask(
            student_logits,
            teacher_logits,
            targets,
            ignore_index=cfg.ignore_index,
            advantage_nats=cfg.teacher_advantage_nats,
        )
    else:
        dmask = valid

    # If this teacher has nothing useful to say in this batch, do not shrink the
    # ordinary data CE by (1-alpha). That would be a very silly way to learn less.
    if not bool(dmask.any()):
        stats.update(agreement_metrics(student_logits, teacher_logits, targets,
                                       ignore_index=cfg.ignore_index))
        return ce, stats

    distill = distill_tok[dmask].mean()
    loss = (1.0 - a) * ce + a * distill
    stats.update(
        {
            "loss_distill": float(distill.detach().item()),
            "loss_total": float(loss.detach().item()),
            "alpha": a,
            "teacher_used": 1.0,
            "teacher_better_fraction": float(
                dmask.float().sum().item() / valid.float().sum().item()
            ),
            "distill_tokens": float(dmask.sum().item()),
        }
    )
    stats.update(agreement_metrics(student_logits, teacher_logits, targets,
                                   ignore_index=cfg.ignore_index))
    return loss, stats


__all__ = [
    "DistillConfig",
    "DistillController",
    "agreement_metrics",
    "should_query_teacher",
    "teacher_anchored_loss",
    "teacher_better_mask",
    "token_cross_entropy",
    "topk_residual_distill_ce",
]
