#!/usr/bin/env python3
"""Small reproducible coverage benchmark for DBlock objective stratification."""
from __future__ import annotations

import argparse
import math
import random
import statistics


def stratified_window(probs, window, rng):
    u = rng.random()
    edge = 0.0
    out = []
    for idx, prob in enumerate(probs):
        nxt = edge + float(prob) * window
        count = math.floor(nxt + u) - math.floor(edge + u)
        out.extend([idx] * count)
        edge = nxt
    if len(out) < window:
        out.extend([len(probs) - 1] * (window - len(out)))
    elif len(out) > window:
        del out[window:]
    rng.shuffle(out)
    return out


def run(blocks=14, visits=16, trials=10000, seed=1337):
    probs = (0.50, 0.25, 0.25)
    rng = random.Random(seed)
    iid_zero_sat = 0
    iid_zero_nat = 0
    iid_spreads = []
    strat_spreads = []
    for _ in range(trials):
        iid_counts = []
        strat_counts = []
        for _block in range(blocks):
            iid = [0, 0, 0]
            for _ in range(visits):
                x = rng.random()
                obj = 0 if x < probs[0] else (1 if x < probs[0] + probs[1] else 2)
                iid[obj] += 1
            st = [0, 0, 0]
            remaining = visits
            while remaining:
                w = min(16, remaining)
                for obj in stratified_window(probs, w, rng):
                    st[obj] += 1
                remaining -= w
            iid_counts.append(iid)
            strat_counts.append(st)
        if any(c[1] == 0 for c in iid_counts):
            iid_zero_sat += 1
        if any(c[2] == 0 for c in iid_counts):
            iid_zero_nat += 1
        iid_spreads.append(sum(statistics.pstdev([c[o] for c in iid_counts]) for o in range(3)))
        strat_spreads.append(sum(statistics.pstdev([c[o] for c in strat_counts]) for o in range(3)))

    print(f"blocks={blocks} visits_per_block={visits} trials={trials}")
    print(f"iid any-block zero SAT: {iid_zero_sat / trials:.4%}")
    print(f"iid any-block zero NAT: {iid_zero_nat / trials:.4%}")
    print(f"iid mean summed cross-block count stddev: {statistics.mean(iid_spreads):.4f}")
    print(f"stratified mean summed cross-block count stddev: {statistics.mean(strat_spreads):.4f}")
    if visits == 16:
        print("stratified 50/25/25 invariant per block: 8 AR / 4 SAT / 4 NAT")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=14)
    ap.add_argument("--visits", type=int, default=16)
    ap.add_argument("--trials", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    run(a.blocks, a.visits, a.trials, a.seed)


if __name__ == "__main__":
    main()
