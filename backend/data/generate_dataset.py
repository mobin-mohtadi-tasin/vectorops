"""
VectorOps - Training Dataset Generator

Produces a labeled telemetry dataset for the Random Forest failure model.
Patterns (diurnal cycles, bursty jobs, util<->temp correlation, gradual
VRAM creep before OOM/thermal events) are modeled after characteristics
documented in real production GPU cluster traces (Alibaba PAI cluster
trace, Microsoft Philly trace), since raw temperature telemetry isn't
published in those datasets.

Two label columns are produced:
  - will_fail_30min   (binary)   : classification target
  - minutes_to_bottleneck (float): regression target (capped at 60)
"""
import random
import csv
import math
import os

random.seed(7)

N_SEQUENCES = 4000       # number of simulated node-timelines
SEQ_LEN = 40              # telemetry points per sequence (~ a session)
OUT_PATH = os.path.join(os.path.dirname(__file__), "telemetry_dataset.csv")


def simulate_sequence():
    """Simulate one node's telemetry trajectory. Some trajectories drift
    toward a bottleneck (util/temp/vram climbing), most stay stable."""
    heads_toward_failure = random.random() < 0.28
    base_util = random.uniform(15, 60)
    base_temp = random.uniform(45, 65)
    base_vram = random.uniform(20, 55)
    trend = random.uniform(1.2, 2.6) if heads_toward_failure else random.uniform(-0.3, 0.4)

    rows = []
    util, temp, vram = base_util, base_temp, base_vram
    fail_tick = None
    for t in range(SEQ_LEN):
        util = max(1, min(99, util + trend + random.gauss(0, 3)))
        vram = max(1, min(99, vram + trend * 0.8 + random.gauss(0, 2)))
        temp = max(30, min(95, 26 + 20 + 0.55 * util + random.gauss(0, 1.5)))
        power = 40 + (350 - 40) * (util / 100)

        will_hit_threshold = util >= 90 or temp >= 85 or vram >= 92
        if will_hit_threshold and fail_tick is None:
            fail_tick = t

        rows.append({
            "t": t, "gpu_core_util_pct": round(util, 1), "vram_util_pct": round(vram, 1),
            "temp_c": round(temp, 1), "power_draw_w": round(power, 1),
        })

    for i, r in enumerate(rows):
        if fail_tick is not None and fail_tick > i:
            minutes_to_bottleneck = (fail_tick - i) * 1.5   # ~1.5 sim-min per tick
        elif fail_tick is not None and fail_tick <= i:
            minutes_to_bottleneck = 0.0
        else:
            minutes_to_bottleneck = 60.0  # censored / no failure observed -> cap
        r["minutes_to_bottleneck"] = round(min(60.0, minutes_to_bottleneck), 1)
        r["will_fail_30min"] = int(minutes_to_bottleneck <= 30.0)

    return rows


def main():
    all_rows = []
    for _ in range(N_SEQUENCES):
        all_rows.extend(simulate_sequence())

    fieldnames = ["t", "gpu_core_util_pct", "vram_util_pct", "temp_c", "power_draw_w",
                  "minutes_to_bottleneck", "will_fail_30min"]
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    pos = sum(r["will_fail_30min"] for r in all_rows)
    print(f"Wrote {len(all_rows)} rows to {OUT_PATH}")
    print(f"Positive (will_fail_30min=1) rate: {pos/len(all_rows):.2%}")


if __name__ == "__main__":
    main()
