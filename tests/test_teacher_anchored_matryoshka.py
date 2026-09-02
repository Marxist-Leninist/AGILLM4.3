import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

MOD_PATH = Path(__file__).resolve().parents[1] / "tools" / "teacher_anchored_matryoshka.py"
spec = importlib.util.spec_from_file_location("teacher_anchored_matryoshka", MOD_PATH)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def test_topk_residual_is_exact_when_k_equals_vocab():
    torch.manual_seed(0)
    s = torch.randn(2, 3, 7, requires_grad=True)
    t = torch.randn(2, 3, 7)
    got = m.topk_residual_distill_ce(s, t, topk=7, temperature=1.0)
    p = F.softmax(t, dim=-1)
    want = -(p * F.log_softmax(s, dim=-1)).sum(-1).reshape(-1)
    assert torch.allclose(got, want, atol=2e-6, rtol=2e-6)


def test_teacher_better_gate_selects_only_helpful_tokens():
    # Teacher is perfect on token 0, student perfect on token 1.
    y = torch.tensor([[0, 1]])
    s = torch.tensor([[[0.0, 0.0], [0.0, 6.0]]])
    t = torch.tensor([[[6.0, 0.0], [0.0, 0.0]]])
    mask = m.teacher_better_mask(s, t, y)
    assert mask.tolist() == [True, False]


def test_no_teacher_is_plain_ce():
    torch.manual_seed(1)
    s = torch.randn(2, 4, 11, requires_grad=True)
    y = torch.randint(0, 11, (2, 4))
    cfg = m.DistillConfig()
    got, stats = m.teacher_anchored_loss(s, y, cfg=cfg)
    want = F.cross_entropy(s.reshape(-1, 11), y.reshape(-1))
    assert torch.allclose(got, want)
    assert stats["teacher_used"] == 0.0


def test_global_catchup_guard_fades_alpha():
    cfg = m.DistillConfig(alpha=0.3, min_alpha=0.0, catchup_margin_ce=0.5)
    c = m.DistillController(cfg)
    assert c.observe_eval(teacher_ce=10.4, student_ce=10.8) == 0.3
    mid = c.observe_eval(teacher_ce=10.4, student_ce=10.15)
    assert abs(mid - 0.15) < 1e-8
    assert c.observe_eval(teacher_ce=10.4, student_ce=9.8) == 0.0


def test_teacher_schedule():
    cfg = m.DistillConfig(teacher_every=4)
    assert [i for i in range(9) if m.should_query_teacher(i, cfg)] == [0, 4, 8]
