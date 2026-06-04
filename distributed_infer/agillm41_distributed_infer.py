#!/usr/bin/env python3
"""Distributed inference harness for the real AGILLM4.1 transformer blocks.

Phase 1 is exact full-sequence AR inference over pipeline stages. Each stage
owns a contiguous transformer/DiffusionBlock layer range and runs the actual
AGILLM4.1 Block implementation, including MoE FFNs when enabled by the
checkpoint config. The coordinator keeps embeddings, final norm, and AR head.
"""
from __future__ import annotations

import argparse
import gc
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import ssl
import struct
import sys
import time
import uuid
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def load_agillm41(path: str | Path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("agillm41_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import AGILLM4.1 runtime from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("agillm41_runtime", module)
    sys.modules.setdefault("agillm35_runtime", module)
    spec.loader.exec_module(module)
    return module


def torch_io():
    import torch
    return torch


def resolve_device(name: str):
    torch = torch_io()
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def load_ckpt(runtime: Any, ckpt_path: str | Path) -> dict[str, Any]:
    torch = torch_io()
    path = Path(ckpt_path)
    resolved = path if path.is_file() else (runtime._resolve_ckpt(path) or path)
    sd = torch.load(resolved, map_location="cpu", weights_only=False)
    if sd.get("delta"):
        cfg = runtime.PRESETS["large"].copy()
        sd["cfg"] = cfg
        sd["tie_weights"] = False
        sd["core"] = sd["weights"]["core"]
        sd["ar"] = sd["weights"]["ar"]
        sd["sat"] = sd["weights"].get("sat", {})
        if "nat" in sd["weights"]:
            sd["nat"] = sd["weights"]["nat"]
    if "tokenizer_json" in sd:
        try:
            from tokenizers import Tokenizer as _Tokenizer
            runtime.tok.backend_tokenizer = _Tokenizer.from_str(sd["tokenizer_json"])
        except Exception:
            pass
    return sd


def dblock_ranges(layers: int, blocks: int) -> list[tuple[int, int]]:
    blocks = max(1, int(blocks))
    span = max(1, layers // blocks)
    out = []
    for i in range(blocks):
        start = i * span
        end = (i + 1) * span if i < blocks - 1 else layers
        if start < layers:
            out.append((start, min(end, layers)))
    return out


def make_dense_mask(mode: str, n: int, device: Any, sat_block: int):
    torch = torch_io()
    if mode == "ar":
        return torch.triu(torch.full((1, 1, n, n), float("-inf"), device=device), 1)
    if mode == "sat":
        idx = torch.arange(n, device=device)
        grp = idx.unsqueeze(0) // int(sat_block)
        allow = (grp.T == grp) | (grp.T > grp)
        return torch.where(allow, 0.0, float("-inf")).unsqueeze(0).unsqueeze(0)
    if mode == "nat":
        return None
    raise ValueError(f"bad mode {mode!r}")


def make_cached_mask(mode: str, q_len: int, total_seq_len: int, device: Any, sat_block: int):
    torch = torch_io()
    if mode == "ar":
        if q_len == 1:
            return None
        k_len = int(total_seq_len)
        q_start = k_len - int(q_len)
        q_pos = torch.arange(q_start, k_len, device=device).view(q_len, 1)
        k_pos = torch.arange(k_len, device=device).view(1, k_len)
        blocked = k_pos > q_pos
        return torch.where(blocked, float("-inf"), 0.0).view(1, 1, q_len, k_len)
    if mode == "nat":
        return None
    return make_dense_mask(mode, int(total_seq_len), device, sat_block)[..., -int(q_len):, :]


class StageModule:
    def __init__(
        self,
        runtime: Any,
        sd: dict[str, Any],
        start_layer: int,
        end_layer: int,
        device: str,
        attn_backend: str,
    ):
        torch = torch_io()
        nn = torch.nn
        cfg = sd["cfg"]
        self.runtime = runtime
        self.start_layer = int(start_layer)
        self.end_layer = int(end_layer)
        self.device = torch.device(device)
        self.cache: dict[str, list[Any]] = {}
        self.cache_last_used: dict[str, float] = {}
        self.max_cache_sessions = 64
        self.module = nn.Module()
        self.module.blocks = nn.ModuleList(
            [
                runtime.Block(
                    int(cfg["d"]),
                    int(cfg["heads"]),
                    int(cfg["rank"]),
                    attn_backend=attn_backend,
                    moe_ffn=bool(cfg.get("moe_ffn", runtime.DEFAULT_MOE_FFN)),
                    moe_experts=int(cfg.get("moe_experts", runtime.DEFAULT_MOE_EXPERTS)),
                    moe_top_k=int(cfg.get("moe_top_k", runtime.DEFAULT_MOE_TOP_K)),
                    moe_mlp_mult=int(cfg.get("moe_mlp_mult", runtime.DEFAULT_MOE_MLP_MULT)),
                )
                for _ in range(self.end_layer - self.start_layer)
            ]
        )
        core_sd = runtime._strip_orig_mod_prefix(sd["core"])
        local_sd = {}
        for local_i, global_i in enumerate(range(self.start_layer, self.end_layer)):
            src_prefix = f"blocks.{global_i}."
            dst_prefix = f"blocks.{local_i}."
            for key, value in core_sd.items():
                if isinstance(key, str) and key.startswith(src_prefix):
                    local_sd[dst_prefix + key[len(src_prefix):]] = value
        local_sd = runtime._prepare_core_state_dict_for_load(self.module, local_sd)
        self.module.load_state_dict(local_sd, strict=True)
        del local_sd, core_sd
        gc.collect()
        self.module.to(self.device)
        self.module.eval()

    def run(self, hidden: Any, mode: str, sat_block: int) -> tuple[Any, float]:
        torch = torch_io()
        start = time.time()
        x = hidden.to(self.device)
        mask = make_dense_mask(mode, int(x.size(1)), self.device, sat_block)
        with torch.no_grad():
            for block in self.module.blocks:
                x = block(x, mask)
        return x.detach().cpu(), time.time() - start

    def _prune_cache(self) -> None:
        excess = len(self.cache) - self.max_cache_sessions
        if excess <= 0:
            return
        for session_id, _ in sorted(self.cache_last_used.items(), key=lambda kv: kv[1])[:excess]:
            self.cache.pop(session_id, None)
            self.cache_last_used.pop(session_id, None)

    def clear_cache(self, session_id: str) -> None:
        self.cache.pop(session_id, None)
        self.cache_last_used.pop(session_id, None)

    def run_cached(
        self,
        hidden: Any,
        mode: str,
        sat_block: int,
        session_id: str,
        total_seq_len: int,
        reset_cache: bool = False,
    ) -> tuple[Any, float]:
        torch = torch_io()
        start = time.time()
        if reset_cache:
            self.clear_cache(session_id)
        x = hidden.to(self.device)
        q_len = int(x.size(1))
        mask = make_cached_mask(mode, q_len, int(total_seq_len), self.device, sat_block)
        kvs = self.cache.get(session_id)
        if kvs is not None and len(kvs) != len(self.module.blocks):
            kvs = None
        new_kvs = []
        with torch.no_grad():
            for idx, block in enumerate(self.module.blocks):
                kv = None if kvs is None else kvs[idx]
                x, new_kv = block(x, mask, kv=kv, use_cache=True, total_seq_len=int(total_seq_len))
                if isinstance(new_kv, tuple):
                    new_kv = tuple(t.detach() for t in new_kv)
                new_kvs.append(new_kv)
        self.cache[session_id] = new_kvs
        self.cache_last_used[session_id] = time.time()
        self._prune_cache()
        return x.detach().cpu(), time.time() - start


WIRE_MAGIC = b"AGI35INF1"


def _torch_dtype_name(dtype: Any) -> str:
    text = str(dtype)
    return text.split(".", 1)[1] if text.startswith("torch.") else text


def _torch_dtype_from_name(name: str) -> Any:
    torch = torch_io()
    table = {
        "float64": torch.float64,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int64": torch.int64,
        "int32": torch.int32,
        "int16": torch.int16,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }
    if name not in table:
        raise ValueError(f"unsupported tensor dtype over wire: {name}")
    return table[name]


def tensor_payload(data: dict[str, Any]) -> bytes:
    hidden = data["hidden"].detach().cpu().contiguous()
    header = {
        "shape": list(hidden.shape),
        "dtype": _torch_dtype_name(hidden.dtype),
        "meta": {k: v for k, v in data.items() if k != "hidden"},
    }
    if header["dtype"] == "bfloat16":
        raw = hidden.view(torch_io().uint16).numpy().tobytes(order="C")
    else:
        raw = hidden.numpy().tobytes(order="C")
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_bytes) > 1_000_000:
        raise ValueError("tensor payload header is too large")
    return WIRE_MAGIC + struct.pack(">I", len(header_bytes)) + header_bytes + raw


def tensor_from_payload(data: bytes) -> dict[str, Any]:
    torch = torch_io()
    if len(data) < len(WIRE_MAGIC) + 4 or not data.startswith(WIRE_MAGIC):
        raise ValueError("bad AGILLM41/AGILLM35 inference wire payload")
    header_len = struct.unpack(">I", data[len(WIRE_MAGIC):len(WIRE_MAGIC) + 4])[0]
    header_start = len(WIRE_MAGIC) + 4
    header_end = header_start + header_len
    if header_len <= 0 or header_len > 1_000_000 or header_end > len(data):
        raise ValueError("bad AGILLM41/AGILLM35 inference wire header")
    header = json.loads(data[header_start:header_end].decode("utf-8"))
    raw = data[header_end:]
    shape = tuple(int(x) for x in header["shape"])
    dtype_name = str(header["dtype"])
    if dtype_name == "bfloat16":
        base = torch.frombuffer(bytearray(raw), dtype=torch.uint16).clone()
        hidden = base.view(torch.bfloat16).reshape(shape)
    else:
        hidden = torch.frombuffer(bytearray(raw), dtype=_torch_dtype_from_name(dtype_name)).clone().reshape(shape)
    out = dict(header.get("meta", {}))
    out["hidden"] = hidden
    return out


def bearer(headers: Any) -> str:
    auth = headers.get("Authorization", "")
    return auth.split(" ", 1)[1].strip() if auth.startswith("Bearer ") else ""


class WorkerHandler(BaseHTTPRequestHandler):
    server_version = "AGILLM41DistributedInferWorker/1"

    def send_json(self, code: int, data: Any) -> None:
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def check_auth(self) -> bool:
        token = getattr(self.server, "token", "")  # type: ignore[attr-defined]
        if not token:
            return True
        if bearer(self.headers) == token:
            return True
        self.send_json(401, {"error": "bad bearer token"})
        return False

    def do_GET(self) -> None:
        if self.path == "/health":
            stage = self.server.stage  # type: ignore[attr-defined]
            self.send_json(
                200,
                {
                    "ok": True,
                    "start_layer": stage.start_layer,
                    "end_layer": stage.end_layer,
                    "device": str(stage.device),
                },
            )
            return
        self.send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/run":
            self.send_json(404, {"error": "not found"})
            return
        if not self.check_auth():
            return
        n = int(self.headers.get("Content-Length", "0"))
        if n <= 0 or n > int(getattr(self.server, "max_bytes", 2_000_000_000)):  # type: ignore[attr-defined]
            self.send_json(413, {"error": "payload too large", "bytes": n})
            return
        payload = tensor_from_payload(self.rfile.read(n))
        if bool(payload.get("use_cache", False)):
            hidden, sec = self.server.stage.run_cached(  # type: ignore[attr-defined]
                payload["hidden"],
                str(payload.get("mode", "ar")),
                int(payload.get("sat_block", 8)),
                str(payload.get("session_id", "")),
                int(payload.get("total_seq_len", int(payload["hidden"].size(1)))),
                bool(payload.get("reset_cache", False)),
            )
        else:
            hidden, sec = self.server.stage.run(  # type: ignore[attr-defined]
                payload["hidden"],
                str(payload.get("mode", "ar")),
                int(payload.get("sat_block", 8)),
            )
        body = tensor_payload(
            {
                "hidden": hidden,
                "stage_sec": sec,
                "start_layer": self.server.stage.start_layer,  # type: ignore[attr-defined]
                "end_layer": self.server.stage.end_layer,  # type: ignore[attr-defined]
            }
        )
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (time.strftime("%FT%TZ", time.gmtime()), fmt % args))


def cmd_worker(args: argparse.Namespace) -> None:
    runtime = load_agillm41(args.agillm35_path)
    sd = load_ckpt(runtime, args.ckpt)
    args.device = resolve_device(args.device)
    stage = StageModule(runtime, sd, args.start_layer, args.end_layer, args.device, args.attn_backend)
    del sd
    gc.collect()
    httpd = ThreadingHTTPServer((args.host, args.port), WorkerHandler)
    httpd.stage = stage  # type: ignore[attr-defined]
    httpd.token = args.token  # type: ignore[attr-defined]
    httpd.max_bytes = args.max_payload_bytes  # type: ignore[attr-defined]
    if args.tls_cert and args.tls_key:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(args.tls_cert, args.tls_key)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
    print(
        json.dumps(
            {
                "event": "worker_ready",
                "bind": [args.host, args.port],
                "layers": [args.start_layer, args.end_layer],
                "device": args.device,
            }
        ),
        flush=True,
    )
    httpd.serve_forever()


class LocalStageClient:
    def __init__(self, stage: StageModule, name: str):
        self.stage = stage
        self.name = name

    def run(self, hidden: Any, mode: str, sat_block: int) -> tuple[Any, dict[str, Any]]:
        out, sec = self.stage.run(hidden, mode, sat_block)
        return out, {"name": self.name, "sec": sec, "layers": [self.stage.start_layer, self.stage.end_layer]}

    def run_cached(
        self,
        hidden: Any,
        mode: str,
        sat_block: int,
        session_id: str,
        total_seq_len: int,
        reset_cache: bool,
    ) -> tuple[Any, dict[str, Any]]:
        out, sec = self.stage.run_cached(hidden, mode, sat_block, session_id, total_seq_len, reset_cache)
        return out, {"name": self.name, "sec": sec, "layers": [self.stage.start_layer, self.stage.end_layer], "cached": True}


class RemoteStageClient:
    def __init__(self, url: str, token: str, name: str, insecure: bool):
        self.url = url.rstrip("/")
        self.token = token
        self.name = name
        self.insecure = insecure

    def run(self, hidden: Any, mode: str, sat_block: int) -> tuple[Any, dict[str, Any]]:
        payload = tensor_payload({"hidden": hidden.detach().cpu(), "mode": mode, "sat_block": sat_block})
        headers = {"Content-Type": "application/octet-stream"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = Request(self.url + "/run", data=payload, method="POST", headers=headers)
        context = ssl._create_unverified_context() if self.insecure else None
        start = time.time()
        with urlopen(req, timeout=600, context=context) as r:
            result = tensor_from_payload(r.read())
        wall = time.time() - start
        return result["hidden"], {
            "name": self.name,
            "sec": float(result.get("stage_sec", 0.0)),
            "wall_sec": wall,
            "layers": [result.get("start_layer"), result.get("end_layer")],
        }

    def run_cached(
        self,
        hidden: Any,
        mode: str,
        sat_block: int,
        session_id: str,
        total_seq_len: int,
        reset_cache: bool,
    ) -> tuple[Any, dict[str, Any]]:
        payload = tensor_payload(
            {
                "hidden": hidden.detach().cpu(),
                "mode": mode,
                "sat_block": sat_block,
                "use_cache": True,
                "session_id": session_id,
                "total_seq_len": int(total_seq_len),
                "reset_cache": bool(reset_cache),
            }
        )
        headers = {"Content-Type": "application/octet-stream"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = Request(self.url + "/run", data=payload, method="POST", headers=headers)
        context = ssl._create_unverified_context() if self.insecure else None
        start = time.time()
        with urlopen(req, timeout=600, context=context) as r:
            result = tensor_from_payload(r.read())
        wall = time.time() - start
        return result["hidden"], {
            "name": self.name,
            "sec": float(result.get("stage_sec", 0.0)),
            "wall_sec": wall,
            "layers": [result.get("start_layer"), result.get("end_layer")],
            "cached": True,
        }


def parse_stage_specs(args: argparse.Namespace, runtime: Any, sd: dict[str, Any]) -> list[Any]:
    specs = args.stage or []
    cfg = sd["cfg"]
    if not specs:
        specs = [f"local:0:{int(cfg['layers'])}"]
    out = []
    for idx, spec in enumerate(specs):
        if spec.startswith("local:"):
            _, a, b = spec.split(":", 2)
            stage = StageModule(runtime, sd, int(a), int(b), args.device, args.attn_backend)
            out.append(LocalStageClient(stage, f"local-{a}-{b}"))
            continue
        if "," not in spec:
            raise SystemExit("remote stage syntax: URL,START,END or local:START:END")
        url, a, b = [x.strip() for x in spec.split(",", 2)]
        out.append(RemoteStageClient(url, args.token, f"remote-{idx}-{a}-{b}", args.insecure))
    return out


def restore_heads(runtime: Any, sd: dict[str, Any], device: str):
    torch = torch_io()
    cfg = sd["cfg"]
    tie_weights = bool(sd.get("tie_weights", False))
    emb = torch.nn.Embedding(runtime.VOCAB, int(cfg["d"])).to(device)
    ln = torch.nn.LayerNorm(int(cfg["d"])).to(device)
    core_sd = runtime._strip_orig_mod_prefix(sd["core"])
    emb.weight.data.copy_(core_sd["emb.weight"].to(device))
    ln.load_state_dict({"weight": core_sd["ln.weight"], "bias": core_sd["ln.bias"]})
    ar_h = runtime.ARHead(int(cfg["d"]), tie_weights=tie_weights, embedding_weight=emb.weight if tie_weights else None).to(device)
    ar_h.load_state_dict(sd["ar"])
    emb.eval()
    ln.eval()
    ar_h.eval()
    return emb, ln, ar_h


def run_stage_pipeline(
    stages: list[Any],
    hidden: Any,
    args: argparse.Namespace,
    use_cache: bool = False,
    session_id: str = "",
    total_seq_len: int = 0,
    reset_cache: bool = False,
) -> tuple[Any, list[dict[str, Any]]]:
    stats = []
    for stage in stages:
        if use_cache:
            hidden, stat = stage.run_cached(
                hidden,
                args.mode,
                args.sat_block,
                session_id,
                int(total_seq_len),
                bool(reset_cache),
            )
        else:
            hidden, stat = stage.run(hidden, args.mode, args.sat_block)
        stats.append(stat)
    return hidden, stats


def sample_next(runtime: Any, ar_h: Any, hidden: Any, ids: Any, args: argparse.Namespace) -> Any:
    logits = ar_h(hidden)[:, -1]
    logits = runtime._apply_penalties(
        logits,
        ids.to(logits.device),
        args.penalty_last_n,
        args.repetition_penalty,
        args.presence_penalty,
        args.frequency_penalty,
    )
    return runtime._sample(logits, args.temperature, args.top_k, args.top_p, args.min_p, args.greedy)


def cmd_infer(args: argparse.Namespace) -> None:
    torch = torch_io()
    runtime = load_agillm41(args.agillm35_path)
    sd = load_ckpt(runtime, args.ckpt)
    args.device = resolve_device(args.device)
    if bool(sd["cfg"].get("anchor_memory", False)):
        raise SystemExit("distributed phase-1 does not support anchor_memory yet")
    stages = parse_stage_specs(args, runtime, sd)
    emb, ln, ar_h = restore_heads(runtime, sd, args.device)
    del sd
    gc.collect()
    prompt_tokens = runtime.tok.encode(args.prompt)
    if not prompt_tokens:
        prompt_tokens = [runtime.EOS]
    ids = torch.tensor([prompt_tokens], dtype=torch.long)
    prompt_len = ids.size(1)
    stage_stats: list[dict[str, Any]] = []
    session_id = args.session_id or f"agillm41-{uuid.uuid4().hex}"
    eos_id = getattr(runtime, "EOS", None)
    generated_tokens = 0
    start = time.time()
    with torch.no_grad():
        if args.cache_mode == "kv":
            hidden = emb(ids.to(args.device)).detach().cpu()
            hidden, stats = run_stage_pipeline(
                stages,
                hidden,
                args,
                use_cache=True,
                session_id=session_id,
                total_seq_len=int(ids.size(1)),
                reset_cache=True,
            )
            stage_stats.extend(stats)
            for step in range(int(args.max_new)):
                h = ln(hidden.to(args.device))
                nxt = sample_next(runtime, ar_h, h, ids, args)
                ids = torch.cat([ids, nxt.detach().cpu()], dim=1)
                generated_tokens += 1
                if eos_id is not None and int(nxt.reshape(-1)[0].item()) == int(eos_id):
                    break
                if step + 1 >= int(args.max_new):
                    break
                hidden = emb(nxt.to(args.device)).detach().cpu()
                hidden, stats = run_stage_pipeline(
                    stages,
                    hidden,
                    args,
                    use_cache=True,
                    session_id=session_id,
                    total_seq_len=int(ids.size(1)),
                    reset_cache=False,
                )
                stage_stats.extend(stats)
        else:
            for _ in range(int(args.max_new)):
                hidden = emb(ids.to(args.device)).detach().cpu()
                hidden, stats = run_stage_pipeline(stages, hidden, args, use_cache=False)
                stage_stats.extend(stats)
                h = ln(hidden.to(args.device))
                nxt = sample_next(runtime, ar_h, h, ids, args)
                ids = torch.cat([ids, nxt.detach().cpu()], dim=1)
                generated_tokens += 1
                if eos_id is not None and int(nxt.reshape(-1)[0].item()) == int(eos_id):
                    break
    elapsed = time.time() - start
    all_ids = ids[0].tolist()
    prompt = runtime.tok.decode(all_ids[:prompt_len], skip_special_tokens=True)
    completion = runtime.tok.decode(all_ids[prompt_len:], skip_special_tokens=True)
    by_stage: dict[str, dict[str, Any]] = {}
    for stat in stage_stats:
        item = by_stage.setdefault(stat["name"], {"calls": 0, "sec": 0.0, "wall_sec": 0.0, "layers": stat.get("layers")})
        item["calls"] += 1
        item["sec"] += float(stat.get("sec", 0.0))
        item["wall_sec"] += float(stat.get("wall_sec", stat.get("sec", 0.0)))
    result = {
        "event": "distributed_infer_done",
        "mode": args.mode,
        "cache_mode": args.cache_mode,
        "session_id": session_id if args.cache_mode == "kv" else None,
        "tokens": generated_tokens,
        "elapsed_sec": round(elapsed, 3),
        "tok_per_sec": round(generated_tokens / max(elapsed, 1e-9), 3),
        "stages": by_stage,
    }
    if args.json:
        result["prompt"] = prompt
        result["completion"] = completion
        print(json.dumps(result, indent=2))
    else:
        print(prompt + completion)
        print(json.dumps(result, indent=2))


def cmd_plan(args: argparse.Namespace) -> None:
    runtime = load_agillm41(args.agillm35_path)
    sd = load_ckpt(runtime, args.ckpt)
    layers = int(sd["cfg"]["layers"])
    ranges = dblock_ranges(layers, args.dblock_blocks)
    print(json.dumps({"layers": layers, "dblock_blocks": args.dblock_blocks, "ranges": ranges}, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description="AGILLM4.1 distributed transformer/MoE/DiffusionBlock inference")
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--agillm35-path",
        "--agillm41-path",
        dest="agillm35_path",
        default=os.environ.get("AGILLM41_RUNTIME") or os.environ.get("AGILLM35_RUNTIME", "./agillm41.py"),
    )
    common.add_argument("--ckpt", required=True)
    common.add_argument("--attn-backend", choices=["manual", "sdpa"], default="manual")
    common.add_argument("--device", default="auto")

    p = sub.add_parser("plan", parents=[common])
    p.add_argument("--dblock-blocks", type=int, default=8)
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("worker", parents=[common])
    p.add_argument("--start-layer", type=int, required=True)
    p.add_argument("--end-layer", type=int, required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9100)
    p.add_argument("--token", default=os.environ.get("AGILLM41_INFER_TOKEN") or os.environ.get("AGILLM35_INFER_TOKEN", ""))
    p.add_argument("--max-payload-bytes", type=int, default=2_000_000_000)
    p.add_argument("--tls-cert")
    p.add_argument("--tls-key")
    p.set_defaults(func=cmd_worker)

    p = sub.add_parser("infer", parents=[common])
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new", type=int, default=16)
    p.add_argument("--mode", choices=["ar"], default="ar")
    p.add_argument("--cache-mode", choices=["kv", "full"], default="kv")
    p.add_argument("--session-id", default="")
    p.add_argument("--stage", action="append", help="local:START:END or URL,START,END. Repeat in pipeline order.")
    p.add_argument("--token", default=os.environ.get("AGILLM41_INFER_TOKEN") or os.environ.get("AGILLM35_INFER_TOKEN", ""))
    p.add_argument("--insecure", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--repetition-penalty", type=float, default=1.3)
    p.add_argument("--presence-penalty", type=float, default=0.0)
    p.add_argument("--frequency-penalty", type=float, default=0.3)
    p.add_argument("--penalty-last-n", type=int, default=128)
    p.add_argument("--sat-block", type=int, default=8)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_infer)

    args = ap.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
