"""
Experiment campaign runner.

Two campaigns cover all 5 required plots:

  Campaign A ("N-sweep"): m in {2,4,8}, N in {10,20,30,40}, Up=0.5, Ua=0.2,
      policy=TBS. Feeds Plot 1 (normalized aperiodic response time vs N).

  Campaign B ("Ua-sweep"): m=4, N=16 (=4m), Up=0.5, Ua in
      {0.05,0.10,...,0.40}, policy in {tbs, background}. Feeds:
        Plot 2 - TBS vs Background mean response time vs Ua
        Plot 3 - per-core total utilization bar chart vs Ua (TBS only)
        Plot 4 - periodic deadline miss rate vs Ua (both policies)
        Plot 5 - average migrations vs total load Up+Ua (TBS only)

Each (config, seed) draws an independent random periodic task set + aperiodic
arrival stream; the same periodic task set / partition is reused across
policies for a fair TBS-vs-Background comparison at a given seed.
"""
import random
import statistics as st

from .generators import gen_periodic_tasks, gen_aperiodic_arrivals, MEAN_APERIODIC_EXEC
from .partition import wfd_partition
from .simulator import Simulator
from .config import HORIZON_MS, MAX_PARTITION_ATTEMPTS


def _make_feasible_periodic_set(n, u_total, m, rng):
    """Regenerate periodic task sets until WFD partitioning succeeds."""
    for _ in range(MAX_PARTITION_ATTEMPTS):
        tasks = gen_periodic_tasks(n, u_total, rng)
        part = wfd_partition(tasks, m)
        if part is not None:
            assignment, core_util = part
            return tasks, assignment, core_util
    raise RuntimeError(f"Could not find feasible partition for n={n}, U_p={u_total}, m={m}")


def run_one_seed(n, u_p, m, u_a, seed, policy="tbs", horizon=HORIZON_MS):
    rng = random.Random(seed)
    tasks, assignment, core_util = _make_feasible_periodic_set(n, u_p, m, rng)
    aper_jobs = gen_aperiodic_arrivals(horizon, u_a, rng)
    sim = Simulator(tasks, assignment, m, aper_jobs, horizon, policy=policy)
    return sim.run()


def aggregate(results):
    """Average scalar metrics across a list of per-seed result dicts."""
    mean_rts = [r["mean_response_time"] for r in results if r["mean_response_time"] is not None]
    miss_rates = [r["miss_rate"] for r in results]
    migrations = [r["migrations"] for r in results]
    n_aper = [r["n_aperiodic"] for r in results]
    m = len(results[0]["core_total_util"])
    core_util_avg = [st.mean(r["core_total_util"][c] for r in results) for c in range(m)]
    return {
        "mean_response_time": st.mean(mean_rts) if mean_rts else None,
        "normalized_response_time": (st.mean(mean_rts) / MEAN_APERIODIC_EXEC) if mean_rts else None,
        "miss_rate_pct": 100.0 * st.mean(miss_rates),
        "avg_migrations": st.mean(migrations),
        "avg_n_aperiodic": st.mean(n_aper),
        "core_total_util_avg": core_util_avg,
        "n_seeds": len(results),
    }


def run_campaign_A(m_values, n_values, u_p, u_a, num_seeds, policy="tbs"):
    """N-sweep for Plot 1."""
    out = {}
    for m in m_values:
        for n in n_values:
            u_p_total = u_p * m
            u_a_total = u_a * m
            
            results = [run_one_seed(n, u_p_total, m, u_a_total, seed=1000 * m + 10 * n + s, policy=policy)
                       for s in range(num_seeds)]
            out[(m, n)] = aggregate(results)
    return out


def run_campaign_B(m, n, u_p, u_a_values, num_seeds, policies=("tbs", "background")):
    """Ua-sweep for Plots 2, 3, 4, 5."""
    out = {}
    u_p_total = u_p * m
    for u_a in u_a_values:
        u_a_total = u_a * m
        for policy in policies:
            results = [run_one_seed(n, u_p_total, m, u_a_total, seed=int(u_a * 10000) + s, policy=policy)
                       for s in range(num_seeds)]
            out[(u_a, policy)] = aggregate(results)
    return out
