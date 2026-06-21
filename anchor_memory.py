#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AnchorMemoryConfig:
    d_model: int
    heads: int
    anchor_stride: int = 256
    max_anchors: int = 2048
    dropout: float = 0.0


class AnchorCompressor(nn.Module):
    """Compress local token spans into trainable anchor vectors."""

    def __init__(self, d_model: int, anchor_stride: int):
        super().__init__()
        self.anchor_stride = anchor_stride
        self.score = nn.Linear(d_model, 1)
        self.mix = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, dim = x.shape
        pad = (-seq) % self.anchor_stride
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        chunks = x.view(bsz, -1, self.anchor_stride, dim)
        weights = self.score(chunks).softmax(dim=2)
        pooled = (chunks * weights).sum(dim=2)
        return pooled + self.mix(pooled)


class AnchorMemoryLayer(nn.Module):
    """Local-token stream reads from a bounded bank of learned anchors."""

    def __init__(self, cfg: AnchorMemoryConfig):
        super().__init__()
        self.cfg = cfg
        self.compress = AnchorCompressor(cfg.d_model, cfg.anchor_stride)
        self.q_ln = nn.LayerNorm(cfg.d_model)
        self.mem_ln = nn.LayerNorm(cfg.d_model)
        self.read = nn.MultiheadAttention(
            cfg.d_model,
            cfg.heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.gate = nn.Sequential(nn.Linear(2 * cfg.d_model, cfg.d_model), nn.Sigmoid())
        self.out_ln = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor | None = None,
        *,
        detach_memory: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        new_anchors = self.compress(x)
        if detach_memory:
            new_anchors = new_anchors.detach()
        if memory is None:
            bank = new_anchors
        else:
            bank = torch.cat([memory, new_anchors], dim=1)
        if bank.size(1) > self.cfg.max_anchors:
            bank = bank[:, -self.cfg.max_anchors :]

        recalled, _ = self.read(self.q_ln(x), self.mem_ln(bank), self.mem_ln(bank), need_weights=False)
        gate = self.gate(torch.cat([x, recalled], dim=-1))
        mixed = x + gate * recalled
        return self.out_ln(mixed), bank


def smoke_test() -> None:
    cfg = AnchorMemoryConfig(d_model=128, heads=8, anchor_stride=32, max_anchors=64)
    layer = AnchorMemoryLayer(cfg)
    x = torch.randn(2, 256, 128)
    y, memory = layer(x)
    assert y.shape == x.shape
    assert memory.shape == (2, 8, 128)
    y2, memory2 = layer(x, memory)
    assert y2.shape == x.shape
    assert memory2.shape == (2, 16, 128)
    print("anchor_memory smoke OK", y.shape, memory2.shape)


if __name__ == "__main__":
    smoke_test()
