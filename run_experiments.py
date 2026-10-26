#!/usr/bin/env python3
"""
Run the full experiment campaign described in CLAUDE.MD and produce the 5
required plots plus a raw-results JSON dump for reproducibility.

Usage:
    python3 run_experiments.py
"""
import json
import time

from sim.config import NUM_SEEDS, DEFAULT_UP, DEFAULT_UA, DEFAULT_M
from sim.experiments import run_campaign_A, run_campaign_B
from sim import plotting

M_VALUES = [2, 4, 8]
N_VALUES = [10, 20, 30, 40]
UA_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


def main():
    t0 = time.time()
    print(f"Running Campaign A (N-sweep) with NUM_SEEDS={NUM_SEEDS} ...")
    campaign_a = run_campaign_A(M_VALUES, N_VALUES, DEFAULT_UP, DEFAULT_UA, NUM_SEEDS, policy="tbs")
    print(f"  done in {time.time()-t0:.1f}s")

    t1 = time.time()
    print(f"Running Campaign B (Ua-sweep) with NUM_SEEDS={NUM_SEEDS} ...")
    campaign_b = run_campaign_B(DEFAULT_M, DEFAULT_M * 4, DEFAULT_UP, UA_VALUES, NUM_SEEDS,
                                 policies=("tbs", "background"))
    print(f"  done in {time.time()-t1:.1f}s")

    # Dump raw aggregated results for reproducibility / inspection
    dump = {
        "config": {
            "NUM_SEEDS": NUM_SEEDS, "M_VALUES": M_VALUES, "N_VALUES": N_VALUES,
            "UA_VALUES": UA_VALUES, "DEFAULT_UP": DEFAULT_UP, "DEFAULT_UA": DEFAULT_UA,
            "DEFAULT_M": DEFAULT_M,
        },
        "campaign_A": {f"m={m},n={n}": v for (m, n), v in campaign_a.items()},
        "campaign_B": {f"ua={u},policy={p}": v for (u, p), v in campaign_b.items()},
    }
    with open("results/raw_results.json", "w") as f:
        json.dump(dump, f, indent=2)
    print("Saved results/raw_results.json")

    print("Generating plots ...")
    plotting.plot1_normalized_response_vs_n(campaign_a, M_VALUES, N_VALUES,
                                             "figures/plot1_normalized_response_vs_n.png")
    plotting.plot2_tbs_vs_background(campaign_b, UA_VALUES,
                                      "figures/plot2_tbs_vs_background.png")
    plotting.plot3_core_utilization_bars(campaign_b, UA_VALUES, DEFAULT_M,
                                          "figures/plot3_core_utilization_bars.png")
    plotting.plot4_deadline_miss_rate(campaign_b, UA_VALUES,
                                       "figures/plot4_deadline_miss_rate.png")
    plotting.plot5_migrations_vs_total_load(campaign_b, UA_VALUES, DEFAULT_UP,
                                             "figures/plot5_migrations_vs_total_load.png")
    print("All 5 figures written to figures/")

    # Quick textual sanity summary
    print("\n--- Sanity summary ---")
    for u in UA_VALUES:
        mr_tbs = campaign_b[(u, "tbs")]["miss_rate_pct"]
        mr_bg = campaign_b[(u, "background")]["miss_rate_pct"]
        print(f"Ua={u:.2f}  periodic miss rate: TBS={mr_tbs:.3f}%  Background={mr_bg:.3f}%")

    print(f"\nTotal wall-clock time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
