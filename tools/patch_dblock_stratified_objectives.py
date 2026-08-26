#!/usr/bin/env python3
"""Checkpoint-compatible AGILLM4.3 DBlock objective-composition upgrade.

Patches an AGILLM4.3 single-file trainer so stochastic AR/SAT/NAT objective
selection is stratified independently inside each DBlock. Expected objective
probabilities are unchanged, but sampling is without replacement inside short
shuffled quota windows, reducing block*objective coverage variance.

Runtime scheduler state only: no model, optimizer, head, or checkpoint tensors.
"""
from __future__ import annotations

import argparse
from pathlib import Path

MARKER_V1 = "AGILLM43-DBLOCK-STRATIFIED-OBJECTIVES-v1"
MARKER = "AGILLM43-DBLOCK-STRATIFIED-OBJECTIVES-v2"

SIG_OLD = "def _choose_objectives(state, args, ar_weight, sat_weight, nat_weight, do_sat_periodic, do_nat_periodic):"
SIG_NEW = "def _choose_objectives(state, args, ar_weight, sat_weight, nat_weight, do_sat_periodic, do_nat_periodic, block_idx=None):"

PICK_OLD = '    picked = random.choices(choices, weights=probs, k=1)[0]\n'
SIGNATURE_V1 = '''        signature = (
            tuple(choices),
            tuple(round(float(p), 12) for p in probs),
            int(window),
        )
'''
SIGNATURE_V2 = '''        assign_signature = tuple(
            tuple(int(layer) for layer in group) for group in state.get("assign", [])
        )
        signature = (
            tuple(choices),
            tuple(round(float(p), 12) for p in probs),
            int(window),
            int(state.get("B", -1)),
            assign_signature,
        )
'''
PICK_NEW = '''    # AGILLM43-DBLOCK-STRATIFIED-OBJECTIVES-v2
    # IID categorical draws preserve the global objective mix only in
    # expectation. With many DBlocks that creates unnecessary short-window
    # block*objective starvation. Build a shuffled, per-block quota window
    # instead. Systematic randomized rounding keeps E[quota_j] = W * p_j,
    # while shuffling avoids a fixed AR/SAT/NAT cycle.
    if bool(getattr(args, "dblock_objective_stratified", True)) and len(choices) > 1:
        window = max(len(choices), int(getattr(args, "dblock_objective_strata_window", 16) or 16))
        key = int(block_idx) if block_idx is not None else -1
        assign_signature = tuple(
            tuple(int(layer) for layer in group) for group in state.get("assign", [])
        )
        signature = (
            tuple(choices),
            tuple(round(float(p), 12) for p in probs),
            int(window),
            int(state.get("B", -1)),
            assign_signature,
        )
        if state.get("objective_strata_signature") != signature:
            state["objective_strata_signature"] = signature
            state["objective_strata_queues"] = {}
        queues = state.setdefault("objective_strata_queues", {})
        queue = queues.get(key)
        if not queue:
            u = random.random()
            edge = 0.0
            quota = []
            for name, prob in zip(choices, probs):
                next_edge = edge + float(prob) * float(window)
                n_pick = int(math.floor(next_edge + u) - math.floor(edge + u))
                if n_pick > 0:
                    quota.extend([name] * n_pick)
                edge = next_edge
            # Systematic rounding should sum exactly to W; keep an explicit
            # numerical guard instead of trusting accumulated float error.
            if len(quota) < window:
                quota.extend([choices[-1]] * (window - len(quota)))
            elif len(quota) > window:
                del quota[window:]
            random.shuffle(quota)
            queue = quota
            queues[key] = queue
        picked = queue.pop()
    else:
        picked = random.choices(choices, weights=probs, k=1)[0]
'''

CALL_OLD = '''    run_ar, run_sat, run_nat, objective = _choose_objectives(
        state, args, ar_weight, sat_weight, nat_weight, do_sat_periodic, do_nat_periodic
    )
'''
CALL_NEW = '''    run_ar, run_sat, run_nat, objective = _choose_objectives(
        state, args, ar_weight, sat_weight, nat_weight, do_sat_periodic, do_nat_periodic,
        block_idx=bi,
    )
'''

ARG_ANCHOR = '    tr.add_argument("--dblock_ar_prob", type=float, default=0.80, help="Stochastic DBlock probability for AR objective.")\n'
ARG_INSERT = '''    tr.add_argument("--dblock_objective_stratified", action=argparse.BooleanOptionalAction, default=True,
                    help="Stratify stochastic AR/SAT/NAT draws independently per DBlock using shuffled quota windows; preserves expected objective probabilities while reducing block*objective coverage variance.")
    tr.add_argument("--dblock_objective_strata_window", type=int, default=16,
                    help="Per-DBlock shuffled objective quota window for --dblock_objective_stratified.")
''' + ARG_ANCHOR


def apply_upgrade(text: str) -> str:
    if MARKER in text:
        return text

    # Migrate the first tested implementation without touching any other
    # trainer code. Block identity must be part of the signature because
    # hot/auto DBlock re-partitioning can reuse the same numeric block index.
    if MARKER_V1 in text:
        if SIGNATURE_V1 not in text:
            raise RuntimeError("v1 marker found but v1 strata signature anchor is missing")
        out = text.replace(MARKER_V1, MARKER, 1)
        out = out.replace(SIGNATURE_V1, SIGNATURE_V2, 1)
        if MARKER not in out:
            raise RuntimeError("v1->v2 migration marker missing")
        return out

    missing = []
    for name, anchor in (
        ("objective signature", SIG_OLD),
        ("objective picker", PICK_OLD),
        ("objective call", CALL_OLD),
        ("argument anchor", ARG_ANCHOR),
    ):
        if anchor not in text:
            missing.append(name)
    if missing:
        raise RuntimeError("refusing partial patch; missing anchors: " + ", ".join(missing))

    out = text.replace(SIG_OLD, SIG_NEW, 1)
    out = out.replace(PICK_OLD, PICK_NEW, 1)
    out = out.replace(CALL_OLD, CALL_NEW, 1)
    out = out.replace(ARG_ANCHOR, ARG_INSERT, 1)
    if MARKER not in out:
        raise RuntimeError("patch marker missing after replacement")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trainer", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--in-place", action="store_true")
    args = ap.parse_args()
    if args.in_place and args.output is not None:
        ap.error("use either --in-place or --output")
    src = args.trainer.read_text()
    out = apply_upgrade(src)
    dest = args.trainer if args.in_place else (args.output or args.trainer.with_suffix(args.trainer.suffix + ".stratified"))
    dest.write_text(out)
    print(f"patched={dest} changed={out != src} marker={MARKER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
