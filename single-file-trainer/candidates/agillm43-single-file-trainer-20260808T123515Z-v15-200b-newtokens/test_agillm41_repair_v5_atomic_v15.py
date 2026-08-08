#!/usr/bin/env python3
"""Torch-free contract tests for the AGILLM4.3 v15 production trainer."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
import pathlib
import re
import tempfile
import types


SOURCE = pathlib.Path(__file__).with_name(
    "agillm41_repair_v5_atomic_v15.py"
)
SOURCE_TEXT = SOURCE.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE_TEXT, filename=str(SOURCE))


def _node(name):
    for item in TREE.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if item.name == name:
                return item
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return item
    raise AssertionError(f"missing source node {name}")


def function_source(name):
    return ast.get_source_segment(SOURCE_TEXT, _node(name)) or ""


def load_pure_contract():
    names = {
        "_AGILLM43_PRODUCTION_SCHEMA",
        "_AGILLM43_CONTINUATION_BUDGET_SCHEMA",
        "_AGILLM43_PRODUCTION_DBLOCK_RESUME_SCHEMA",
        "_AGILLM43_DEFAULT_ADDITIONAL_TOKENS",
        "_AGILLM43_PRODUCTION_BLOCK",
        "_AGILLM43_PRODUCTION_BATCH",
        "_AGILLM43_PRODUCTION_TOKENS_PER_COMMIT",
        "_AGILLM43_PRODUCTION_SOURCE",
        "_AGILLM43_QUALITY_TELEMETRY_ONLY",
        "_AGILLM43_REPAIR_BASE_STEP",
        "_AGILLM43_REPAIR_SEED_SHA256",
        "_AGILLM43_REPAIR_BASE_SEEN_TOK",
        "_dblock_require_exact_int",
        "_dblock_require_finite_number",
        "_continuation_budget_seal",
        "_continuation_budget_validate",
        "_continuation_budget_payload",
        "_continuation_budget_commit",
        "_repair_lr_multiplier",
        "_dblock_local_aux_policy_probe",
        "_dblock_local_spike_probe",
        "_dblock_anchor_spike_probe",
        "_agillm43_quality_gate_latest_payload",
    }
    selected = []
    for item in TREE.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if item.name in names:
                selected.append(item)
        elif isinstance(item, ast.Assign):
            assigned = {
                target.id for target in item.targets
                if isinstance(target, ast.Name)
            }
            if assigned & names:
                selected.append(item)
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "copy": copy,
        "hashlib": hashlib,
        "json": json,
        "math": math,
        "os": os,
        "pathlib": pathlib,
        "Path": pathlib.Path,
        "re": re,
    }
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return types.SimpleNamespace(**namespace)


def assert_raises(callable_, exception=ValueError):
    try:
        callable_()
    except exception:
        return
    raise AssertionError(f"expected {exception.__name__}")


def test_release_contract(mod):
    assert mod._AGILLM43_PRODUCTION_SCHEMA == "agillm43.production.v15"
    assert mod._AGILLM43_QUALITY_TELEMETRY_ONLY is True
    assert mod._AGILLM43_DEFAULT_ADDITIONAL_TOKENS == 200_000_000_000
    assert mod._AGILLM43_PRODUCTION_BLOCK == 2_048
    assert mod._AGILLM43_PRODUCTION_BATCH == 24
    assert mod._AGILLM43_PRODUCTION_TOKENS_PER_COMMIT == 49_152


def test_exact_200b_budget_and_final_batch(mod):
    origin = 84_999_045_120
    anchor_sha = mod._AGILLM43_REPAIR_SEED_SHA256
    budget = mod._continuation_budget_payload(
        200_000_000_000, origin, {}, anchor_step=1_729_310,
        anchor_checkpoint_sha256=anchor_sha,
    )
    assert budget["target_new_tokens"] == 200_000_000_000
    assert budget["target_seen_tok"] == 284_999_045_120
    assert budget["full_commits"] == 4_069_010
    assert budget["final_commit_tokens"] == 20_480
    assert budget["target_commits"] == 4_069_011
    assert budget["remaining_new_tokens"] == 200_000_000_000
    assert budget["quality_gate_can_block"] is False

    first = mod._continuation_budget_commit(budget, 49_152)
    assert first["commits_since_anchor"] == 1
    assert first["new_tokens_committed"] == 49_152
    assert mod._continuation_budget_payload(
        200_000_000_000, origin + 49_152,
        {"continuation_budget": first}, anchor_step=0,
    ) == first
    assert_raises(lambda: mod._continuation_budget_commit(budget, 20_480))

    near_final = dict(budget)
    near_final["commits_since_anchor"] = budget["full_commits"]
    near_final["new_tokens_committed"] = (
        budget["full_commits"] * budget["nominal_tokens_per_commit"]
    )
    near_final["remaining_new_tokens"] = budget["final_commit_tokens"]
    near_final = mod._continuation_budget_seal(near_final)
    final = mod._continuation_budget_commit(near_final, 20_480)
    assert final["new_tokens_committed"] == 200_000_000_000
    assert final["remaining_new_tokens"] == 0
    assert final["commits_since_anchor"] == 4_069_011
    assert_raises(lambda: mod._continuation_budget_commit(final, 49_152))


def test_budget_rejects_coerced_clock_types(mod):
    anchor = mod._AGILLM43_REPAIR_SEED_SHA256
    for invalid in (True, "200000000000", 200_000_000_000.0):
        assert_raises(lambda value=invalid: mod._continuation_budget_payload(
            value, 84_999_045_120, {}, anchor_step=1_729_310,
            anchor_checkpoint_sha256=anchor,
        ))
    budget = mod._continuation_budget_payload(
        200_000_000_000, 84_999_045_120, {}, anchor_step=1_729_310,
        anchor_checkpoint_sha256=anchor,
    )
    for invalid in (True, "49152", 49_152.0):
        assert_raises(
            lambda value=invalid: mod._continuation_budget_commit(budget, value)
        )
    corrupted = dict(budget)
    corrupted["anchor_step"] = "1729310"
    corrupted = mod._continuation_budget_seal(corrupted)
    assert_raises(lambda: mod._continuation_budget_validate(corrupted))
    assert_raises(lambda: mod._continuation_budget_payload(
        199_999_999_999, 84_999_045_120, {}, anchor_step=1_729_310,
        anchor_checkpoint_sha256=anchor,
    ))
    wrong_target = dict(budget)
    wrong_target["target_new_tokens"] = 100_000_000_000
    wrong_target = mod._continuation_budget_seal(wrong_target)
    assert_raises(lambda: mod._continuation_budget_validate(wrong_target))


def test_lr_uses_new_token_clock_not_raw_lifetime_counter(mod):
    budget = mod._continuation_budget_payload(
        200_000_000_000, 84_999_045_120, {}, anchor_step=1_729_310,
        anchor_checkpoint_sha256=mod._AGILLM43_REPAIR_SEED_SHA256,
    )
    budget = mod._continuation_budget_commit(budget, 49_152)
    args = types.SimpleNamespace(
        lr_decay="cosine", lr_decay_tokens=200_000_000_000,
        _continuation_budget=budget, lr_min_mult=0.2,
        lr_warmup_tokens=10_000_000, lr_warmup_min_mult=0.1,
    )
    left = mod._repair_lr_multiplier(args, 1, 2)
    right = mod._repair_lr_multiplier(args, 999_999_999_999, 3)
    assert left == right
    assert left[1] == 49_152


def test_finite_quality_is_telemetry_only(mod):
    args = types.SimpleNamespace(
        loss_spike_skip=3.0,
        dblock_local_aux_warn_ce=25.0,
        dblock_local_aux_max_ce=50.0,
    )
    state = {
        "B": 14,
        "spike_ema_by_objective": {"sat": 10.0},
        "spike_block_ema_by_objective": {},
    }
    probe = mod._dblock_local_spike_probe(state, args, "sat", 0, 500.0)
    assert probe["spike"] is False
    assert probe["advisory_spike"] is True
    assert probe["pending"] is not None
    assert probe["aux_policy"]["hard_reject"] is False
    assert probe["aux_policy"]["would_hard_reject"] is True
    zero = mod._dblock_local_spike_probe(
        {"B": 14, "spike_ema_by_objective": {},
         "spike_block_ema_by_objective": {}},
        args, "ar", 0, 0.0,
    )
    assert zero["spike"] is False
    assert zero["pending"]["global_next"] == 0.0
    assert zero["pending"]["block_next"] == 0.0

    anchor_args = types.SimpleNamespace(
        dblock_fullstack_anchor_spike_skip=3.0,
        dblock_fullstack_anchor_max_ce=50.0,
    )
    anchor = mod._dblock_anchor_spike_probe(
        {"fullstack_anchor_spike_ema_by_family": {"ar": 10.0}},
        anchor_args, "ar", 500.0,
    )
    assert anchor["spike"] is False
    assert anchor["advisory_spike"] is True
    assert math.isfinite(anchor["ema_next"])
    zero_anchor = mod._dblock_anchor_spike_probe(
        {"fullstack_anchor_spike_ema_by_family": {"ar": 0.0}},
        anchor_args, "ar", 0.0,
    )
    assert zero_anchor["spike"] is False
    assert zero_anchor["ema_next"] == 0.0


def test_no_finite_quality_control_sinks():
    step_source = function_source("_dblock_step")
    for forbidden in (
        "_dblock_record_local_spike_rejection",
        "_dblock_record_anchor_spike_rejection",
        '"local_loss_spike"',
        '"fullstack_anchor_spike"',
        "spike rejected",
    ):
        assert forbidden not in step_source
    assert "no valid document crop" in step_source
    assert "optional anchor" in step_source
    assert "nonfinite_local_loss" in step_source
    assert "nonfinite_gradient_norm" in step_source
    for name in (
        "_dblock_fullstack_ar_anchor",
        "_dblock_fullstack_sat_anchor",
        "_dblock_fullstack_nat_anchor",
    ):
        source = function_source(name)
        assert "finite and not spike" not in source
        assert "if finite:" in source
        assert ".backward()" in source


def test_exact_child_resume_and_restore_order():
    load_source = function_source("load_ckpt")
    assert "_repair_verify_manifest_sidecar" in load_source
    assert "exact child optimizer state is incompatible" in load_source
    assert "exact child scaler state failed to load" in load_source
    assert "_load_module_state_exact(core" in load_source
    exact_loader = function_source("_load_module_state_exact")
    assert "missing=" in exact_loader
    assert "dtype" in exact_loader
    train_source = function_source("train")
    assert train_source.index("_production_preflight(args)") < train_source.index(
        "infer_cfg_from_ckpt(src_probe)"
    )
    assert "production child resume forbids optimizer/scaler reset" in train_source
    cfg_source = function_source("infer_cfg_from_ckpt")
    for state_key in ("core", "ar", "sat", "nat", "opt", "scaler"):
        assert repr(state_key) in cfg_source or f'"{state_key}"' in cfg_source
    assert "skip_keys=" in cfg_source
    phase_source = function_source("_train_phase")
    assert phase_source.index("_DBS = _dblock_init") < phase_source.index("if val_batches:")
    assert "while _phase_budget_remaining() > 0:" in phase_source
    assert "_step_batch_target" in phase_source
    assert "toks_processed = int(ids.numel())" in phase_source
    assert phase_source.index(
        'elif production_continuation:\n        phase_target_tokens'
    ) < phase_source.index('elif steps:')
    rejected_continue = phase_source.index(
        "if _dblock_attempt_rejected:\n            continue",
        phase_source.index("_val_due_time"),
    )
    assert rejected_continue < phase_source.index("_flush_sentinel =", rejected_continue)


def test_production_preflight_is_upfront_and_exact():
    source = function_source("_production_preflight")
    assert "before model allocation" in source
    assert "_AGILLM43_DEFAULT_ADDITIONAL_TOKENS" in source
    assert '"target_tokens"' in source
    assert '"dblock_blocks": 14' in source
    assert '"dblock_objective_mode": "committed_cycle"' in source
    assert '"token_param_ratio": 0.0' in source
    assert '"chat_messages_key": "messages"' in source
    assert '"sft_add_generation_prompt"' in source
    assert '"quality_gate_can_block": False' in source
    assert "val_file" not in source


def test_production_heldout_is_optional_telemetry():
    source = function_source("_build_val_set")
    assert "_production_continuation_active" in source
    assert "validation telemetry disabled" in source
    assert "without blocking training" in source
    assert source.index("if production_continuation:") < source.index(
        "_build_val_set_legacy"
    )


def test_production_checkpoint_pointer_contract():
    save_source = function_source("save_ckpt")
    assert "production checkpoint requires DBlock resume state" in save_source
    assert "_production_publish_pointer_bundle" in save_source
    publisher = function_source("_production_publish_pointer_bundle")
    assert "_production_checkpoint_package_receipt" in publisher
    assert '"training_latest.json"' in publisher
    assert '"latest.json"' in publisher
    assert '"quality_gate_can_block": False' in publisher
    package = function_source("_production_checkpoint_package_receipt")
    assert "_agillm43_verify_sharded_manifest_files" in package
    assert 'required_targets = {"core", "ar", "sat", "nat", "opt", "scaler"}' in package
    dblock = function_source("_production_dblock_validate_resume_payload")
    assert "production DBlock B must equal 14" in dblock
    assert "round-robin" in dblock
    assert "SAT/NAT/AR cycle" in dblock
    assert "Quality observations are validated only" in dblock


def test_legacy_quality_file_cannot_pin(mod):
    with tempfile.TemporaryDirectory() as tmp:
        gate_path = pathlib.Path(tmp) / "quality_gate.json"
        gate_path.write_text(json.dumps({
            "require_auto_infer_ok": True,
            "max_promoted_delta_step": 1,
        }), encoding="utf-8")
        prior = os.environ.get("AGILLM43_QUALITY_GATE")
        os.environ["AGILLM43_QUALITY_GATE"] = str(gate_path)
        try:
            proposed = {"path": "/new/checkpoint.pt", "step": 999}
            promoted, advisory = mod._agillm43_quality_gate_latest_payload(
                pathlib.Path(proposed["path"]), {"step": 999}, proposed
            )
        finally:
            if prior is None:
                os.environ.pop("AGILLM43_QUALITY_GATE", None)
            else:
                os.environ["AGILLM43_QUALITY_GATE"] = prior
        assert promoted["path"] == proposed["path"]
        assert promoted["quality_gate_can_block"] is False
        assert "ignored" in advisory


def main():
    mod = load_pure_contract()
    tests = [
        lambda: test_release_contract(mod),
        lambda: test_exact_200b_budget_and_final_batch(mod),
        lambda: test_budget_rejects_coerced_clock_types(mod),
        lambda: test_lr_uses_new_token_clock_not_raw_lifetime_counter(mod),
        lambda: test_finite_quality_is_telemetry_only(mod),
        test_no_finite_quality_control_sinks,
        test_exact_child_resume_and_restore_order,
        test_production_preflight_is_upfront_and_exact,
        test_production_heldout_is_optional_telemetry,
        test_production_checkpoint_pointer_contract,
        lambda: test_legacy_quality_file_cannot_pin(mod),
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS all {len(tests)} v15 tests")


if __name__ == "__main__":
    main()
