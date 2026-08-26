import math
import random
import unittest
from collections import Counter
from types import SimpleNamespace

from tools.patch_dblock_stratified_objectives import MARKER, apply_upgrade


FIXTURE = r'''
import argparse
import math
import random

def _dblock_hot_float(args, key, default, min_value=None):
    value = float(getattr(args, key, default))
    return max(min_value, value) if min_value is not None else value

def _choose_objectives(state, args, ar_weight, sat_weight, nat_weight, do_sat_periodic, do_nat_periodic):
    mode = str(getattr(args, "dblock_objective_mode", "periodic") or "periodic").lower()
    if mode != "stochastic":
        return ar_weight > 0.0, sat_weight > 0.0 and do_sat_periodic, nat_weight > 0.0 and do_nat_periodic, "periodic"
    choices = []
    probs = []
    if ar_weight > 0.0:
        choices.append("ar")
        probs.append(max(0.0, _dblock_hot_float(args, "dblock_ar_prob", 0.80, min_value=0.0)))
    if sat_weight > 0.0 and not getattr(args, "ar_only", False):
        choices.append("sat")
        probs.append(max(0.0, _dblock_hot_float(args, "dblock_sat_prob", 0.10, min_value=0.0)))
    if nat_weight > 0.0 and not getattr(args, "ar_only", False):
        choices.append("nat")
        probs.append(max(0.0, _dblock_hot_float(args, "dblock_nat_prob", 0.10, min_value=0.0)))
    if not choices:
        return False, False, False, "none"
    total = sum(probs)
    if total <= 0.0:
        probs = [1.0 / len(choices) for _ in choices]
    else:
        probs = [p / total for p in probs]
    picked = random.choices(choices, weights=probs, k=1)[0]
    return picked == "ar", picked == "sat", picked == "nat", picked

def _dummy_step(state, args, ar_weight, sat_weight, nat_weight, do_sat_periodic, do_nat_periodic, bi):
    run_ar, run_sat, run_nat, objective = _choose_objectives(
        state, args, ar_weight, sat_weight, nat_weight, do_sat_periodic, do_nat_periodic
    )
    return run_ar, run_sat, run_nat, objective

def _parser():
    tr = argparse.ArgumentParser()
    tr.add_argument("--dblock_ar_prob", type=float, default=0.80, help="Stochastic DBlock probability for AR objective.")
    return tr
'''


def load_patched_namespace():
    patched = apply_upgrade(FIXTURE)
    ns = {}
    exec(compile(patched, "<patched-fixture>", "exec"), ns, ns)
    return patched, ns


class StratifiedObjectiveTests(unittest.TestCase):
    def args(self, **kw):
        base = dict(
            dblock_objective_mode="stochastic",
            dblock_ar_prob=0.50,
            dblock_sat_prob=0.25,
            dblock_nat_prob=0.25,
            dblock_objective_stratified=True,
            dblock_objective_strata_window=16,
            ar_only=False,
        )
        base.update(kw)
        return SimpleNamespace(**base)

    def test_patch_is_idempotent_and_compiles(self):
        patched, _ = load_patched_namespace()
        self.assertIn(MARKER, patched)
        self.assertEqual(patched, apply_upgrade(patched))

    def test_live_mix_is_exact_per_block_window(self):
        _, ns = load_patched_namespace()
        choose = ns["_choose_objectives"]
        args = self.args()
        state = {}
        random.seed(12345)
        for block in range(14):
            counts = Counter()
            for _ in range(16):
                counts[choose(state, args, 1.0, 1.0, 1.0, True, True, block_idx=block)[3]] += 1
            self.assertEqual(counts, Counter({"ar": 8, "sat": 4, "nat": 4}))

    def test_arbitrary_mix_has_bounded_quota_error(self):
        _, ns = load_patched_namespace()
        choose = ns["_choose_objectives"]
        args = self.args(
            dblock_ar_prob=0.70,
            dblock_sat_prob=0.20,
            dblock_nat_prob=0.10,
            dblock_objective_strata_window=17,
        )
        random.seed(7)
        state = {}
        counts = Counter(
            choose(state, args, 1.0, 1.0, 1.0, True, True, block_idx=3)[3]
            for _ in range(17)
        )
        for name, prob in (("ar", 0.70), ("sat", 0.20), ("nat", 0.10)):
            target = 17 * prob
            self.assertIn(counts[name], {math.floor(target), math.ceil(target)})
        self.assertEqual(sum(counts.values()), 17)

    def test_probability_change_invalidates_old_queue(self):
        _, ns = load_patched_namespace()
        choose = ns["_choose_objectives"]
        state = {}
        random.seed(19)
        a = self.args()
        for _ in range(3):
            choose(state, a, 1.0, 1.0, 1.0, True, True, block_idx=2)
        old_sig = state["objective_strata_signature"]
        b = self.args(dblock_ar_prob=0.60, dblock_sat_prob=0.20, dblock_nat_prob=0.20)
        choose(state, b, 1.0, 1.0, 1.0, True, True, block_idx=2)
        self.assertNotEqual(old_sig, state["objective_strata_signature"])

    def test_iid_escape_hatch_still_works(self):
        _, ns = load_patched_namespace()
        choose = ns["_choose_objectives"]
        state = {}
        args = self.args(dblock_objective_stratified=False)
        random.seed(4)
        out = [choose(state, args, 1.0, 1.0, 1.0, True, True, block_idx=0)[3] for _ in range(20)]
        self.assertTrue(set(out) <= {"ar", "sat", "nat"})
        self.assertNotIn("objective_strata_queues", state)


if __name__ == "__main__":
    unittest.main()
