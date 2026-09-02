# Teacher-anchored Matryoshka for the AGILLM 49.1M popout

Date: 2026-09-02

## Decision

Train the new 49.1M popout as a standalone language model, but use the already-pretrained ~1.1B AGILLM parent as a **frozen online teacher** on a throttled subset of batches. This is a modified Matryoshka recipe, not a claim that the existing 1.1B architecture magically became a jointly nested Matryoshka suite after ~200B training tokens.

The original paper jointly trains every exit from the start. AGILLM has the opposite chronology: the ~1.1B parent is mature and the 49.1M student is new. Therefore:

1. Never backpropagate through or update the 1.1B teacher.
2. Keep ordinary next-token CE on real data as the student's primary objective.
3. On teacher steps, add paper-style soft-target distillation with alpha_d = 0.30.
4. Distill only on tokens where the teacher currently gives the gold next token lower NLL than the student. This prevents negative transfer from known weak regions of the parent.
5. Fade the global distillation coefficient as held-out student CE overtakes teacher CE, rather than forcing the 49M model to inherit the teacher's ceiling.
6. Log teacher/student top-1 agreement and KL because agreement is the quantity the paper connects directly to speculative-decoding acceptance.

## What is copied from Matryoshka

- Online largest-model -> smaller-model logit distillation.
- Convex CE/distillation objective.
- alpha_d = 0.30 as the initial conservative setting.
- Cross-model alignment as an explicit training target/metric.
- The 49M model remains independently deployable.

## What is deliberately different

- The teacher is frozen and pretrained; the student is new.
- No inter-model junction is inserted into the existing 1.1B model. That junction matters when width/depth blocks are jointly stacked and trained end-to-end; it is not required for a detached student trained from a frozen teacher.
- No claim of zero-cost distillation. A separate 1.1B forward pass costs compute, so the default runs it every 4th student step.
- No claim of shared-KV-cache speculative decoding. Exact KV reuse requires shared, identical draft/verifier layers. Distillation improves agreement, but independently updated 49M weights are not identical to the 1.1B verifier weights.

## Default production knobs

```text
alpha_d                  0.30
teacher_every            4        # teacher on 25% of steps
temperature              1.0      # paper-compatible default
topk                     32       # top-k + residual bucket soft CE
teacher_better_only      true
teacher_advantage_nats   0.0
catchup_margin_ce        0.50
min_alpha                0.0
```

The top-k + residual-bucket loss keeps the teacher's omitted probability mass instead of naively renormalizing top-k logits. This is intended to cut working-memory pressure around the large-vocabulary output head.

## Trainer integration

The current 49M trainer already computes student AR logits. On every `teacher_every` step:

```python
from tools.teacher_anchored_matryoshka import (
    DistillConfig, DistillController, should_query_teacher, teacher_anchored_loss,
)

cfg_d = DistillConfig()
controller = DistillController(cfg_d)

# once per held-out eval interval
alpha = controller.observe_eval(teacher_ce=teacher_eval_ce,
                                student_ce=student_eval_ce)

# each training step
teacher_logits = None
if should_query_teacher(step, cfg_d):
    with torch.inference_mode():
        teacher_h = teacher_core(input_ids, None)
        teacher_logits = teacher_ar(teacher_h)

loss, distill_stats = teacher_anchored_loss(
    student_logits,
    targets,
    cfg=cfg_d,
    teacher_logits=teacher_logits,
    alpha=controller.effective_alpha,
)
loss.backward()
```

For the live 49M trainer, apply the teacher head in sequence chunks if full `[B,L,V]` teacher logits create avoidable VRAM pressure. If chunking, aggregate CE and distillation numerators/counts across chunks rather than naively averaging per-chunk means; the returned `valid_tokens` and `distill_tokens` stats make the weighting explicit.

## Promotion rule

Do not promote because "distillation is on". Promote only if held-out CE improves versus the CE-only 49M baseline and generation quality does not regress. Record top-1 teacher agreement as a secondary inference metric, not as the quality gate.

A minimum A/B should compare, at matched student tokens and wall-clock budget:

- CE-only 49M.
- Teacher-anchored 49M with `teacher_every=4`, `alpha=0.30`.
- If teacher compute dominates, `teacher_every=8` rather than removing the quality guards.

The mature parent should be treated as a source of useful priors rather than a permanent ceiling on the student.
