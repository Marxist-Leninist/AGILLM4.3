#!/usr/bin/env python3
"""
AGILLM-4.3 DiffusionBlocks - INDEPENDENT (one-block-resident) training.

Implements the memory / parallelism property from the DiffusionBlocks paper
(arXiv:2506.14202), "Comparison with activation checkpointing":

  * Activation checkpointing reduces ONLY activation memory.
  * DiffusionBlocks reduces ALL memory components (params + grads + optimizer
    + activations) by ~B, because each of the B blocks is trained INDEPENDENTLY
    - only ONE block's params/grads/optimizer/activations are resident at a time.
  * Each block is embarrassingly parallel: B invocations on B machines, zero
    communication. They are then composed into ONE unified inference model.

Why this is valid for AGILLM-4.3: in agillm41._dblock_step every block reads the
EDM-noised embedding DIRECTLY (h = ci*zt, zt = emb + sigma*noise) and runs only
its OWN contiguous layer group - blocks are PARALLEL sigma-band denoisers that
share emb / ln / head, NOT a sequential stack. So block_b's forward
(emb -> layers[b] -> ln) is mathematically identical whether layers[b] live in a
full L-layer Encoder or in a standalone Encoder with L/B layers. That identity is
exactly what makes independent training + compose() exact.

This module REUSES the live trainer's real classes/objective (agillm41.Encoder,
ARHead, _block_sigmas, _edm_pre, _edm_w, fused_ce, _run_block). It does NOT import
or modify the training loop, and never needs to run on the training GPU
(CPU-only is fine; the math is identical).

Shared-stem policy: emb + final ln (+ AR head if untied) are a small SHARED stem.
Independent block training FREEZES the stem (loaded from a snapshot), so each
block-trainer only holds grads/optimizer for its L/B layers, and compose() is
EXACT (every block saw the identical frozen stem).

CLI:
  python agillm43_diffusionblocks_independent.py selftest
  python agillm43_diffusionblocks_independent.py mem-report [--d --layers --heads --rank --vocab]
  python agillm43_diffusionblocks_independent.py make-stem --d.. --layers.. --B.. --out stem.pt
  python agillm43_diffusionblocks_independent.py train-block --stem stem.pt --B 4 --block 0 --out b0.pt [--steps..] [--grad-checkpoint]
  python agillm43_diffusionblocks_independent.py compose --blocks b0.pt b1.pt b2.pt b3.pt --out full.pt
"""
import os, sys, math, json, argparse, time
import torch
import torch.nn as nn

import agillm41 as A
import nB300_agillm4 as M

# ---- faithful reuse of the live trainer's building blocks ----
Encoder      = A.Encoder
ARHead       = A.ARHead
block_sigmas = A._block_sigmas
edm_pre      = A._edm_pre
edm_w        = A._edm_w
fused_ce     = A.fused_ce
run_block    = A._run_block

# The live AGILLM-4.3 (~1.22B): d_model=1280, 28 layers, 20 heads, low-rank 160.
LIVE_CFG = {"d": 1280, "layers": 28, "heads": 20, "rank": 160}


# ----------------------------- helpers --------------------------------------
def even_split(L, B):
    if int(L) % int(B) != 0:
        raise ValueError(
            f"even-split demo: L={L} not divisible by B={B}. The live "
            f"_dblock_block_layers handles a remainder; that is out of scope here.")
    return int(L) // int(B)


def sub_cfg(cfg, lb):
    c = dict(cfg); c["layers"] = int(lb); return c


def build_block_model(cfg, B, device="cpu", tie_weights=True, attn_backend="sdpa"):
    """One DiffusionBlocks block as a standalone model: emb + L/B layers + ln + AR head."""
    lb = even_split(int(cfg["layers"]), B)
    core = Encoder(sub_cfg(cfg, lb), tie_weights=tie_weights, attn_backend=attn_backend).to(device)
    ew = core.emb.weight if tie_weights else None
    ar = ARHead(int(cfg["d"]), tie_weights=tie_weights, embedding_weight=ew).to(device)
    return core, ar, lb


def stem_param_list(core, ar, tie_weights):
    ps = list(core.emb.parameters()) + list(core.ln.parameters())
    if not tie_weights:
        ps += list(ar.parameters())
    return ps


def set_freeze(params, frozen=True):
    for p in params:
        p.requires_grad = (not frozen)


def unique_trainable(*modules):
    seen, out = set(), []
    for m in modules:
        for p in m.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p)); out.append(p)
    return out


def _causal(T, device):
    m = M.causal_mask(T, structured=False)
    if torch.is_tensor(m):
        m = m.to(device)
    return m


def _checkpoint_this_layer(enabled, layer_pos, layer_count, stride=1, skip_tail=0):
    if not enabled:
        return False
    stride = int(stride or 1)
    if stride <= 0:
        return False
    skip_tail = max(0, int(skip_tail or 0))
    if skip_tail and int(layer_pos) >= max(0, int(layer_count) - skip_tail):
        return False
    return stride == 1 or (int(layer_pos) % stride) == 0


def _denoise_hidden(emb_mod, ln_mod, layers, ids, sig, noise, attn_args=None,
                    grad_checkpoint=False, checkpoint_stride=1, checkpoint_skip_tail=0):
    """Deterministic EDM-preconditioned forward through `layers` (mirrors _dblock_step).
    `layers` is an iterable of Block modules. `noise` is supplied for reproducibility."""
    device = ids.device
    T = ids.size(1)
    cs, co, ci = edm_pre(sig)
    causal = _causal(T, device)
    emb = emb_mod(ids)
    zt = emb + sig[:, None, None] * noise
    h = ci * zt
    layer_list = list(layers)
    layer_count = len(layer_list)
    for lpos, blk in enumerate(layer_list):
        use_ckpt = _checkpoint_this_layer(grad_checkpoint, lpos, layer_count, checkpoint_stride, checkpoint_skip_tail)
        h = run_block(blk, h, causal, use_ckpt, None, "off")
    return ln_mod(cs * zt + co * h)


def dblock_ar_loss(core, ar, ids, lo, hi, wmax=5.0, grad_checkpoint=False,
                   checkpoint_stride=1, checkpoint_skip_tail=0):
    """AR branch of agillm41._dblock_step, restricted to this block's sigma band."""
    device = ids.device
    u = torch.rand(ids.size(0), device=device)
    sig = torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))   # log-uniform in band
    w = edm_w(sig, wmax)
    noise = torch.randn(core.emb(ids).shape, device=device)
    Dn = _denoise_hidden(core.emb, core.ln, core.blocks, ids, sig, noise,
                         grad_checkpoint=grad_checkpoint, checkpoint_stride=checkpoint_stride,
                         checkpoint_skip_tail=checkpoint_skip_tail)
    raw = fused_ce(Dn[:, :-1], ar.proj.weight, ids[:, 1:])
    return w * raw, float(raw.detach())


def opt_state_bytes(opt):
    tot = 0
    for st in opt.state.values():
        for v in st.values():
            if torch.is_tensor(v):
                tot += v.numel() * v.element_size()
    return tot


def trainable_bytes(params):
    return sum(p.numel() * p.element_size() for p in params if p.requires_grad)


# ----------------------------- stem I/O -------------------------------------
def make_stem(cfg, B, out_path, device="cpu", tie_weights=True, attn_backend="sdpa", seed=0):
    """Create + save a shared stem (emb + ln [+ AR head if untied]).
    In real use you would instead snapshot the stem from an existing checkpoint."""
    torch.manual_seed(seed)
    core, ar, lb = build_block_model(cfg, B, device, tie_weights, attn_backend)
    payload = {
        "kind": "diffusionblock_stem",
        "cfg": dict(cfg), "tie_weights": bool(tie_weights),
        "emb": core.emb.state_dict(), "ln": core.ln.state_dict(),
        "ar": ar.state_dict(),
    }
    if out_path:
        torch.save(payload, out_path)
    return payload


def load_stem_into(core, ar, stem, tie_weights):
    if isinstance(stem, str):
        stem = torch.load(stem, map_location="cpu")
    core.emb.load_state_dict(stem["emb"])
    core.ln.load_state_dict(stem["ln"])
    if not tie_weights:
        ar.load_state_dict(stem["ar"])


# ----------------------------- train one block ------------------------------
def random_batch(batch, seqlen, vocab, device):
    return torch.randint(0, int(vocab), (int(batch), int(seqlen)), device=device)


def train_block(cfg, B, block_index, *, steps=60, batch=4, seqlen=32, lr=3e-4,
                tie_weights=True, device="cpu", attn_backend="sdpa",
                stem=None, freeze_stem=True, out_path=None, log_every=0,
                data_fn=None, seed=0, grad_checkpoint=False, checkpoint_stride=1,
                checkpoint_skip_tail=0):
    """Train exactly ONE block independently. Embarrassingly parallel: run B of
    these (one per block_index) on B machines, no communication."""
    torch.manual_seed(1000 + seed + block_index)
    sig = block_sigmas(B)
    lo, hi = sorted([sig[block_index], sig[block_index + 1]])
    core, ar, lb = build_block_model(cfg, B, device, tie_weights, attn_backend)
    if stem is not None:
        load_stem_into(core, ar, stem, tie_weights)
    if freeze_stem:
        set_freeze(stem_param_list(core, ar, tie_weights), True)
    core.train()
    train_params = unique_trainable(core, ar)
    opt = torch.optim.AdamW(train_params, lr=lr)
    if data_fn is None:
        data_fn = lambda: random_batch(batch, seqlen, A.VOCAB, device)

    losses = []
    for step in range(int(steps)):
        ids = data_fn()
        opt.zero_grad(set_to_none=True)
        loss, raw = dblock_ar_loss(core, ar, ids, lo, hi, grad_checkpoint=grad_checkpoint,
                                   checkpoint_stride=checkpoint_stride,
                                   checkpoint_skip_tail=checkpoint_skip_tail)
        loss.backward()
        nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()
        losses.append(raw)
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"[block {block_index}/{B}] step {step:4d}  sigma[{lo:.3f},{hi:.3f}]  ar_ce={raw:.4f}", flush=True)

    info = {
        "block_index": int(block_index), "B": int(B), "layers_per_block": int(lb),
        "sigma_lo": float(lo), "sigma_hi": float(hi),
        "loss_first": float(losses[0]) if losses else None,
        "loss_last": float(losses[-1]) if losses else None,
        "trainable_param_bytes": int(trainable_bytes(train_params)),
        "optimizer_state_bytes": int(opt_state_bytes(opt)),
        "n_trainable_params": int(sum(p.numel() for p in train_params)),
        "grad_checkpoint": bool(grad_checkpoint),
        "checkpoint_stride": int(checkpoint_stride or 0),
        "checkpoint_skip_tail": int(checkpoint_skip_tail or 0),
    }
    if out_path:
        torch.save({
            "kind": "diffusionblock_independent",
            "meta": {"block_index": int(block_index), "B": int(B),
                     "layers_per_block": int(lb), "base_cfg": dict(cfg),
                     "tie_weights": bool(tie_weights), "attn_backend": attn_backend,
                     "sigma_lo": float(lo), "sigma_hi": float(hi)},
            "blocks": core.blocks.state_dict(),
            "emb": core.emb.state_dict(), "ln": core.ln.state_dict(),
            "ar": ar.state_dict(),
        }, out_path)
        info["out_path"] = out_path
    return info, (core, ar)


# ----------------------------- compose --------------------------------------
def compose(block_ckpts, out_path=None, device="cpu"):
    """Reassemble B independently-trained blocks into ONE unified L-layer Encoder."""
    payloads = [torch.load(p, map_location=device) if isinstance(p, str) else p for p in block_ckpts]
    metas = [d["meta"] for d in payloads]
    B = metas[0]["B"]; cfg = dict(metas[0]["base_cfg"]); tie = metas[0]["tie_weights"]
    ab = metas[0].get("attn_backend", "sdpa"); lb = even_split(int(cfg["layers"]), B)
    idxs = sorted(m["block_index"] for m in metas)
    if idxs != list(range(B)):
        raise ValueError(f"compose needs exactly blocks 0..{B-1}, got {idxs}")
    for m in metas:
        if m["B"] != B or dict(m["base_cfg"]) != cfg:
            raise ValueError("compose: mismatched B / base_cfg across block checkpoints")

    full = Encoder(cfg, tie_weights=tie, attn_backend=ab).to(device)
    ew = full.emb.weight if tie else None
    ar = ARHead(int(cfg["d"]), tie_weights=tie, embedding_weight=ew).to(device)

    by_bi = {m["block_index"]: d for m, d in zip(metas, payloads)}
    # shared stem: identical across blocks (all trained frozen to the same stem) -> take block 0
    b0 = by_bi[0]
    full.emb.load_state_dict(b0["emb"]); full.ln.load_state_dict(b0["ln"])
    if not tie:
        ar.load_state_dict(b0["ar"])
    # slot each block's L/B layers into the full stack
    for bi in range(B):
        blk_sd = by_bi[bi]["blocks"]
        for li in range(lb):
            pref = f"{li}."
            sub = {k[len(pref):]: v for k, v in blk_sd.items() if k.startswith(pref)}
            full.blocks[bi * lb + li].load_state_dict(sub)

    if out_path:
        torch.save({"kind": "diffusionblock_composed", "cfg": cfg, "tie_weights": tie,
                    "attn_backend": ab, "B": B, "core": full.state_dict(),
                    "ar": ar.state_dict()}, out_path)
    return full, ar, cfg, B, lb


# ----------------------------- memory report --------------------------------
def _measure_per_layer_params(cfg):
    """Build a 1-layer Encoder with a tiny vocab (vocab-independent layer count)."""
    saved = A.VOCAB
    try:
        A.VOCAB = 8
        enc = Encoder(sub_cfg(cfg, 1), tie_weights=False, attn_backend="sdpa")
        p_layer = sum(p.numel() for p in enc.blocks.parameters())
        p_ln = sum(p.numel() for p in enc.ln.parameters())
        del enc
    finally:
        A.VOCAB = saved
    return int(p_layer), int(p_ln)


def mem_report(cfg=None, vocab=None, Bs=(1, 2, 4, 7, 14, 28), bytes_per=4):
    cfg = dict(cfg or LIVE_CFG)
    vocab = int(vocab or A.VOCAB)
    d, L = int(cfg["d"]), int(cfg["layers"])
    p_layer, p_ln = _measure_per_layer_params(cfg)
    p_emb = vocab * d                       # tied: AR head shares this
    p_stem = p_emb + p_ln                   # shared, frozen during independent training
    p_full = p_stem + L * p_layer
    GB = lambda n: n * bytes_per / 1e9

    print(f"# DiffusionBlocks memory model  cfg={cfg}  vocab={vocab}  (fp32, {bytes_per}B/elem)")
    print(f"#   per-layer params = {p_layer:,}   shared stem (emb+ln, tied head) = {p_stem:,}")
    print(f"#   FULL model params = {p_full:,}  ({GB(p_full):.2f} GB params)")
    print(f"#   Monolith TRAIN mem ~ 4*P_full (param+grad+2*Adam) = {GB(4*p_full):.2f} GB  (+ activations A*L)")
    print()
    hdr = ("B", "L/B", "trainable P", "4*train (param+grad+adam)", "frozen stem", "resident vs full 4P", "speedup")
    print("{:>3} {:>4} {:>14} {:>26} {:>13} {:>20} {:>8}".format(*hdr))
    for B in Bs:
        if L % B:
            continue
        lb = L // B
        p_train = lb * p_layer
        train_mem = 4 * p_train               # param + grad + 2 Adam moments (trainable only)
        stem_mem = p_stem                     # frozen: params only, no grad/opt
        resident_4p = train_mem + stem_mem    # vs monolith 4*p_full
        factor = (4 * p_full) / resident_4p
        print("{:>3} {:>4} {:>14,} {:>23.2f} GB {:>10.2f} GB {:>16.2f} GB {:>7.2f}x".format(
            B, lb, p_train, GB(train_mem), GB(stem_mem), GB(resident_4p), factor))
    print()
    print("# 'speedup' = monolith 4*P_full / one-block-resident (4*trainable + frozen stem).")
    print("# The 4P (param+grad+Adam) term scales ~1/B; the frozen shared stem is the floor")
    print("# (itself offloadable/shardable). Activations scale 1/B too and combine with")
    print("# --grad_checkpoint (already in agillm41) for the paper's 4/3-time, 1/B-memory point.")


# ----------------------------- self test ------------------------------------
def selftest():
    torch.manual_seed(0)
    dev = "cpu"
    cfg = {"d": 64, "layers": 8, "heads": 4, "rank": 16}
    B = 4
    saved_vocab = A.VOCAB
    ok = True
    try:
        A.VOCAB = 256                      # tiny vocab -> fast/small selftest
        print(f"[selftest] cfg={cfg} B={B} vocab={A.VOCAB}")

        # 1) shared stem
        stem = make_stem(cfg, B, None, device=dev, tie_weights=True, seed=7)

        # 2) train each block INDEPENDENTLY (frozen stem)
        blocks, infos = [], []
        for bi in range(B):
            info, _ = train_block(cfg, B, bi, steps=40, batch=4, seqlen=24, lr=5e-3,
                                  tie_weights=True, device=dev, stem=stem,
                                  freeze_stem=True, out_path=f"/tmp/_dbi_b{bi}.pt", seed=3)
            infos.append(info); blocks.append(f"/tmp/_dbi_b{bi}.pt")
            dec = info["loss_first"] - info["loss_last"]
            print(f"  block {bi}: ar_ce {info['loss_first']:.4f} -> {info['loss_last']:.4f} "
                  f"(drop {dec:+.4f}), trainable={info['n_trainable_params']:,} "
                  f"opt_state={info['optimizer_state_bytes']/1e6:.2f}MB")
            ok &= (info["loss_last"] < info["loss_first"])     # independent training reduces its band loss

        # 3) memory: one block's trainable+opt vs a MONOLITHIC full model
        full_core = Encoder(cfg, tie_weights=True, attn_backend="sdpa").to(dev)
        full_ar = ARHead(cfg["d"], tie_weights=True, embedding_weight=full_core.emb.weight).to(dev)
        full_params = unique_trainable(full_core, full_ar)
        full_opt = torch.optim.AdamW(full_params, lr=1e-3)
        # one adam step so optimizer state exists
        ids = random_batch(4, 24, A.VOCAB, dev)
        sigs = block_sigmas(B)
        l, _ = dblock_ar_loss(full_core, full_ar, ids, sigs[0], sigs[-1]); l.backward(); full_opt.step()
        full_opt_mb = opt_state_bytes(full_opt) / 1e6
        one_opt_mb = infos[0]["optimizer_state_bytes"] / 1e6
        ratio = full_opt_mb / max(one_opt_mb, 1e-9)
        print(f"[selftest] optimizer-state  full={full_opt_mb:.2f}MB  one-block={one_opt_mb:.2f}MB  ratio={ratio:.2f}x (target ~{B}x on layer params)")
        # full opt state includes the (large, tied) emb head too; on the LAYER portion the ratio ~ B.
        ok &= (one_opt_mb < full_opt_mb)

        # 4) compose -> unified model
        full, ar, ccfg, cB, lb = compose(blocks, out_path="/tmp/_dbi_full.pt", device=dev)
        full.eval()
        print(f"[selftest] composed full Encoder: layers={ccfg['layers']} from B={cB} x lb={lb}")

        # 5) EXACT round-trip: composed full's block-slice forward == standalone block forward
        max_diff = 0.0
        for bi in range(B):
            torch.manual_seed(500 + bi)
            ids = random_batch(2, 20, A.VOCAB, dev)
            sgl = block_sigmas(B); lo, hi = sgl[bi], sgl[bi + 1]
            sig = torch.full((ids.size(0),), float((lo * hi) ** 0.5), device=dev)
            noise = torch.randn(full.emb(ids).shape, device=dev)
            # standalone block (reload its ckpt)
            d = torch.load(blocks[bi], map_location=dev)
            sc = Encoder(sub_cfg(cfg, lb), tie_weights=True, attn_backend="sdpa").to(dev)
            sc.emb.load_state_dict(d["emb"]); sc.ln.load_state_dict(d["ln"]); sc.blocks.load_state_dict(d["blocks"])
            sc.eval()
            Dn_standalone = _denoise_hidden(sc.emb, sc.ln, sc.blocks, ids, sig, noise)
            # composed full, using only this block's layer slice
            sl = [full.blocks[bi * lb + j] for j in range(lb)]
            Dn_composed = _denoise_hidden(full.emb, full.ln, sl, ids, sig, noise)
            diff = float((Dn_standalone - Dn_composed).abs().max())
            max_diff = max(max_diff, diff)
        print(f"[selftest] compose round-trip max|Δhidden| = {max_diff:.2e}  (want < 1e-4)")
        ok &= (max_diff < 1e-4)

        print()
        print("=" * 70)
        print(f"[selftest] {'PASS' if ok else 'FAIL'}")
        print("=" * 70)
    finally:
        A.VOCAB = saved_vocab
    return 0 if ok else 1


# ----------------------------- max-model calculator -------------------------
# bytes per TRAINABLE param = param + grad + optimizer-state:
#   fp32     : 4(p) + 4(g) + 8(Adam m,v fp32)       = 16
#   adam8bit : 2(p bf16) + 2(g bf16) + 2(m,v 8-bit) = 6   (what the live trainer uses)
#   bf16     : 2(p) + 2(g) + 4(Adam m,v)            = 8
_OPT_BYTES = {"fp32": 16, "adam8bit": 6, "bf16": 8}


def max_model(budget_gb=24.0, cfg=None, vocab=None, opt="adam8bit",
              per_layer_params=None, act_reserve_gb=2.5, stem_bytes=4,
              offload_stem=False, Bs=(1, 4, 14, 28, 64, 128),
              tok_per_s=0.0, target_tokens=0.0):
    """Largest model trainable when only ONE block is resident at a time.
    VRAM caps the BLOCK; the total = stem + B*block, and B is bounded by
    TIME (serial on 1 GPU) or #MACHINES (parallel), NOT by memory."""
    cfg = dict(cfg or LIVE_CFG)
    vocab = int(vocab or A.VOCAB)
    d = int(cfg["d"])
    if per_layer_params is None:
        per_layer_params, _ = _measure_per_layer_params(cfg)
    p_layer = int(per_layer_params)
    p_stem = vocab * d                                   # tied emb+head = shared floor
    opt_bytes = _OPT_BYTES.get(opt, 16)
    stem_gb = 0.0 if offload_stem else p_stem * stem_bytes / 1e9
    avail = budget_gb - stem_gb - act_reserve_gb
    if avail <= 0 or p_layer <= 0:
        print(f"# budget {budget_gb}GB too small for stem({stem_gb:.2f})+act({act_reserve_gb})")
        return
    per_layer_gb = opt_bytes * p_layer / 1e9
    maxL = max(0, int(avail / per_layer_gb))
    timed = bool(tok_per_s and target_tokens)
    print(f"# max-model  budget={budget_gb}GB  d={d}  vocab={vocab}  opt={opt}({opt_bytes}B/param)")
    print(f"#   per-layer={p_layer/1e6:.1f}M  stem={'offloaded' if offload_stem else f'{stem_gb:.2f}GB'}  "
          f"act_reserve={act_reserve_gb}GB  usable={avail:.1f}GB")
    print(f"#   -> ONE resident block holds up to {maxL} layers (~{maxL*p_layer/1e9:.2f}B params)")
    print(f"#   total model = stem + B x block ; B bounded by TIME/#machines, NOT memory.")
    print("  {:>4} {:>14} {:>14}   {}".format("B", "total layers", "TOTAL params",
          "wall-clock (parallel / serial)" if timed else ""))
    for B in Bs:
        totL = B * maxL
        totP = (p_stem + totL * p_layer) / 1e9
        wc = ""
        if timed:
            t_block_h = target_tokens / tok_per_s / 3600.0   # hrs to give ONE block target_tokens
            wc = f"{t_block_h:.1f}h / {t_block_h*B:.1f}h"
        print("  {:>4} {:>14,} {:>12.1f}B   {}".format(B, totL, totP, wc))
    if not timed:
        print("#   (add --tok-per-s and --target-tokens for wall-clock: parallel=1 block-time, serial=B block-times)")


# ----------------------------- CLI ------------------------------------------
def _cfg_from_args(a):
    return {"d": a.d, "layers": a.layers, "heads": a.heads, "rank": a.rank}


def main():
    ap = argparse.ArgumentParser(description="AGILLM-4.3 DiffusionBlocks independent training")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("selftest")

    m = sub.add_parser("mem-report")
    for name, dv in (("d", 1280), ("layers", 28), ("heads", 20), ("rank", 160)):
        m.add_argument(f"--{name}", type=int, default=dv)
    m.add_argument("--vocab", type=int, default=0)

    mk = sub.add_parser("make-stem")
    for name, dv in (("d", 1280), ("layers", 28), ("heads", 20), ("rank", 160)):
        mk.add_argument(f"--{name}", type=int, default=dv)
    mk.add_argument("--B", type=int, required=True)
    mk.add_argument("--out", required=True)
    mk.add_argument("--no-tie", action="store_true")

    tb = sub.add_parser("train-block")
    for name, dv in (("d", 1280), ("layers", 28), ("heads", 20), ("rank", 160)):
        tb.add_argument(f"--{name}", type=int, default=dv)
    tb.add_argument("--B", type=int, required=True)
    tb.add_argument("--block", type=int, required=True)
    tb.add_argument("--stem", default=None)
    tb.add_argument("--out", required=True)
    tb.add_argument("--steps", type=int, default=200)
    tb.add_argument("--batch", type=int, default=4)
    tb.add_argument("--seqlen", type=int, default=64)
    tb.add_argument("--lr", type=float, default=3e-4)
    tb.add_argument("--no-tie", action="store_true")
    tb.add_argument("--no-freeze-stem", action="store_true")
    tb.add_argument("--device", default="cpu")
    tb.add_argument("--log-every", type=int, default=20)
    tb.add_argument("--grad-checkpoint", "--grad_checkpoint", action="store_true", dest="grad_checkpoint",
                    help="Checkpoint selected independent block layers during backward to reduce activation VRAM.")
    tb.add_argument("--checkpoint-stride", "--dblock_checkpoint_stride", type=int, default=1, dest="checkpoint_stride",
                    help="With --grad-checkpoint, checkpoint one layer every N block layers; 1=all, 2=alternate, 0=off.")
    tb.add_argument("--checkpoint-skip-tail", "--dblock_checkpoint_skip_tail", type=int, default=0, dest="checkpoint_skip_tail",
                    help="With --grad-checkpoint, do not checkpoint this many final layers in the independent block.")

    cp = sub.add_parser("compose")
    cp.add_argument("--blocks", nargs="+", required=True)
    cp.add_argument("--out", required=True)
    cp.add_argument("--device", default="cpu")

    mm = sub.add_parser("max-model")
    for name, dv in (("d", 1280), ("heads", 20), ("rank", 160), ("layers", 1)):
        mm.add_argument(f"--{name}", type=int, default=dv)
    mm.add_argument("--budget-gb", type=float, default=24.0)
    mm.add_argument("--vocab", type=int, default=0)
    mm.add_argument("--opt", default="adam8bit", choices=list(_OPT_BYTES))
    mm.add_argument("--per-layer-m", type=float, default=0.0,
                    help="override per-layer params in millions (e.g. 37.7 = real MoE layer)")
    mm.add_argument("--act-reserve-gb", type=float, default=2.5)
    mm.add_argument("--stem-bytes", type=int, default=4)
    mm.add_argument("--offload-stem", action="store_true")
    mm.add_argument("--tok-per-s", type=float, default=0.0)
    mm.add_argument("--target-tokens", type=float, default=0.0)

    a = ap.parse_args()
    if a.cmd == "selftest":
        sys.exit(selftest())
    if a.cmd == "mem-report":
        mem_report(_cfg_from_args(a), vocab=(a.vocab or None)); return
    if a.cmd == "max-model":
        cfg = {"d": a.d, "layers": max(1, a.layers), "heads": a.heads, "rank": a.rank}
        pl = (a.per_layer_m * 1e6) if a.per_layer_m > 0 else None
        max_model(budget_gb=a.budget_gb, cfg=cfg, vocab=(a.vocab or None), opt=a.opt,
                  per_layer_params=pl, act_reserve_gb=a.act_reserve_gb, stem_bytes=a.stem_bytes,
                  offload_stem=a.offload_stem, tok_per_s=a.tok_per_s, target_tokens=a.target_tokens)
        return
    if a.cmd == "make-stem":
        info = make_stem(_cfg_from_args(a), a.B, a.out, tie_weights=not a.no_tie)
        print(f"[make-stem] saved stem -> {a.out}"); return
    if a.cmd == "train-block":
        info, _ = train_block(_cfg_from_args(a), a.B, a.block, steps=a.steps, batch=a.batch,
                              seqlen=a.seqlen, lr=a.lr, tie_weights=not a.no_tie,
                              device=a.device, stem=a.stem, freeze_stem=not a.no_freeze_stem,
                              out_path=a.out, log_every=a.log_every,
                              grad_checkpoint=a.grad_checkpoint, checkpoint_stride=a.checkpoint_stride,
                              checkpoint_skip_tail=a.checkpoint_skip_tail)
        print(json.dumps(info, indent=2)); return
    if a.cmd == "compose":
        full, ar, cfg, B, lb = compose(a.blocks, out_path=a.out, device=a.device)
        print(f"[compose] {B} blocks x {lb} layers -> full {cfg['layers']}-layer model saved {a.out}"); return


if __name__ == "__main__":
    main()
