# Combined Python Source

## `run_experiments.py`

```python
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
```

## `sim/__init__.py`

```python

```

## `sim/config.py`

```python
"""
Central experiment configuration.

NOTE ON SEED COUNT: the assignment sheet asks for >=100 unique task sets per
configuration. The simulator benchmarks at ~15-20ms/run worst case, so the
full campaign (28 configurations x 100 seeds = 2800 runs) completes in well
under a minute -- we use the full spec value of 100 directly.
"""

NUM_SEEDS = 100         # random task sets averaged per configuration (spec: >=100)
HORIZON_MS = 3000.0     # simulation time horizon per run, in ms
MAX_PARTITION_ATTEMPTS = 200  # regenerate task set this many times if WFD infeasible

# Fixed defaults used when a parameter isn't the one being swept
DEFAULT_UP = 0.5
DEFAULT_UA = 0.2
DEFAULT_M = 4
```

## `sim/experiments.py`

```python
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
            results = [run_one_seed(n, u_p, m, u_a, seed=1000 * m + 10 * n + s, policy=policy)
                       for s in range(num_seeds)]
            out[(m, n)] = aggregate(results)
    return out


def run_campaign_B(m, n, u_p, u_a_values, num_seeds, policies=("tbs", "background")):
    """Ua-sweep for Plots 2, 3, 4, 5."""
    out = {}
    for u_a in u_a_values:
        for policy in policies:
            results = [run_one_seed(n, u_p, m, u_a, seed=int(u_a * 10000) + s, policy=policy)
                       for s in range(num_seeds)]
            out[(u_a, policy)] = aggregate(results)
    return out
```

## `sim/generators.py`

```python
"""
Workload generators.

Conventions (documented, since the assignment leaves them underspecified):
  - Periods are drawn (with replacement) from the allowed discrete set.
  - C_i = U_i * P_i, rounded to 3 decimal places (sub-millisecond resolution).
    If rounding produces 0, we clamp to 0.001 ms so every task does *some* work.
  - Aperiodic job sizes are drawn from an Exponential(mean=MEAN_EXEC) distribution,
    clamped at a small positive minimum so no job has ~0 execution time.
  - Aperiodic arrivals follow a homogeneous Poisson process over [0, horizon).
    Target load U_a is realized approximately via lambda = U_a / mean_exec.
"""
import random

PERIODS = [10, 20, 50, 100, 200, 500, 1000]  # ms
MEAN_APERIODIC_EXEC = 5.0  # ms, documented default


def uunifast(n, u_sum, rng):
    """Standard UUniFast algorithm: n utilizations summing to u_sum, unbiased."""
    utils = []
    sum_u = u_sum
    for i in range(1, n):
        next_sum = sum_u * (rng.random() ** (1.0 / (n - i)))
        utils.append(sum_u - next_sum)
        sum_u = next_sum
    utils.append(sum_u)
    return utils


def gen_periodic_tasks(n, u_total, rng):
    """Generate n periodic tasks with total utilization u_total using UUniFast.
    D_i = P_i (implicit deadline), C_i = U_i * P_i (rounded)."""
    utils = uunifast(n, u_total, rng)
    tasks = []
    for i, u in enumerate(utils):
        period = rng.choice(PERIODS)
        c = round(u * period, 3)
        if c <= 0:
            c = 0.001
        tasks.append({
            "id": i,
            "period": float(period),
            "deadline": float(period),  # D_i = P_i
            "C": c,
            "U": c / period,
        })
    return tasks


def gen_aperiodic_arrivals(horizon, u_a, rng, mean_exec=MEAN_APERIODIC_EXEC):
    """Generate a Poisson arrival process of aperiodic jobs over [0, horizon).
    lambda chosen so that E[load] = lambda * mean_exec ~= u_a."""
    if u_a <= 0:
        return []
    lam = u_a / mean_exec  # jobs per ms
    jobs = []
    t = 0.0
    jid = 0
    while True:
        inter = rng.expovariate(lam)
        t += inter
        if t >= horizon:
            break
        exec_time = rng.expovariate(1.0 / mean_exec)
        exec_time = max(0.1, exec_time)
        jobs.append({"id": jid, "arrival": t, "C": exec_time})
        jid += 1
    return jobs
```

## `sim/partition.py`

```python
"""
Offline partitioning: Partitioned-EDF feasibility via Worst-Fit-Decreasing (WFD).

Procedure (as specified):
  1. Sort periodic tasks by decreasing utilization U_i.
  2. For each task, assign it to the core with the *largest remaining capacity*
     (i.e. smallest current utilization) among cores where it still fits
     (U_core + U_i <= 1).
  3. If no core fits the task, the task set is declared infeasible for this
     partitioning (caller should regenerate a different random task set).
"""


def wfd_partition(tasks, m):
    """Returns (assignment dict task_id->core, core_util list) or None if infeasible."""
    sorted_tasks = sorted(tasks, key=lambda t: t["U"], reverse=True)
    core_util = [0.0] * m
    assignment = {}
    for t in sorted_tasks:
        # Worst fit: try cores in order of *largest remaining capacity* first.
        order = sorted(range(m), key=lambda c: core_util[c])
        placed = False
        for c in order:
            if core_util[c] + t["U"] <= 1.0 + 1e-9:
                assignment[t["id"]] = c
                core_util[c] += t["U"]
                placed = True
                break
        if not placed:
            return None
    return assignment, core_util
```

## `sim/plotting.py`

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot1_normalized_response_vs_n(campaign_a, m_values, n_values, outpath):
    plt.figure(figsize=(7, 5))
    for m in m_values:
        ys = [campaign_a[(m, n)]["normalized_response_time"] for n in n_values]
        plt.plot(n_values, ys, marker="o", label=f"m = {m} cores")
    plt.xlabel("Number of periodic tasks, N")
    plt.ylabel("Normalized aperiodic response time\n(mean response time / E[E_k])")
    plt.title("Plot 1: Normalized Aperiodic Response Time vs N (TBS)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot2_tbs_vs_background(campaign_b, u_a_values, outpath):
    plt.figure(figsize=(7, 5))
    tbs = [campaign_b[(u, "tbs")]["mean_response_time"] for u in u_a_values]
    bg = [campaign_b[(u, "background")]["mean_response_time"] for u in u_a_values]
    plt.plot(u_a_values, tbs, marker="o", label="TBS")
    plt.plot(u_a_values, bg, marker="s", label="Background scheduling")
    plt.xlabel("Aperiodic load, U_a")
    plt.ylabel("Mean aperiodic response time (ms)")
    plt.title("Plot 2: TBS vs Background Mean Response Time (m = 4)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot3_core_utilization_bars(campaign_b, u_a_values, m, outpath):
    plt.figure(figsize=(8, 5))
    width = 0.8 / m
    x = list(range(len(u_a_values)))
    for c in range(m):
        ys = [campaign_b[(u, "tbs")]["core_total_util_avg"][c] for u in u_a_values]
        xpos = [xi + c * width for xi in x]
        plt.bar(xpos, ys, width=width, label=f"Core {c}")
    plt.xticks([xi + width * (m - 1) / 2 for xi in x], [f"{u:.2f}" for u in u_a_values])
    plt.xlabel("Aperiodic load, U_a")
    plt.ylabel("Per-core total utilization (periodic + aperiodic)")
    plt.title("Plot 3: Per-Core Utilization Distribution under Global TBS (m = 4)")
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot4_deadline_miss_rate(campaign_b, u_a_values, outpath):
    plt.figure(figsize=(7, 5))
    tbs = [campaign_b[(u, "tbs")]["miss_rate_pct"] for u in u_a_values]
    bg = [campaign_b[(u, "background")]["miss_rate_pct"] for u in u_a_values]
    plt.plot(u_a_values, tbs, marker="o", label="TBS")
    plt.plot(u_a_values, bg, marker="s", label="Background scheduling")
    plt.axhline(0, color="gray", linewidth=1)
    plt.ylim(-1, 5)
    plt.xlabel("Aperiodic load, U_a")
    plt.ylabel("Periodic deadline miss rate (%)")
    plt.title("Plot 4: Periodic Deadline Miss Rate vs U_a (must be 0%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot5_migrations_vs_total_load(campaign_b, u_a_values, u_p, outpath):
    plt.figure(figsize=(7, 5))
    total_load = [u_p + u for u in u_a_values]
    migs = [campaign_b[(u, "tbs")]["avg_migrations"] for u in u_a_values]
    plt.plot(total_load, migs, marker="o", color="tab:red")
    plt.xlabel("Total system load, U_p + U_a")
    plt.ylabel("Average migrations per run (aperiodic jobs)")
    plt.title("Plot 5: Migration Overhead vs Total System Load (TBS, m = 4)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
```

## `sim/simulator.py`

```python
"""
Discrete-event multi-core scheduling simulator.

Runtime rule: each core always runs the ready job with the earliest deadline
among what's assigned to it (EDF), where:
  - periodic jobs carry their *real* absolute deadline (release + D_i)
  - aperiodic jobs (under TBS) carry a *virtual* deadline assigned by the TBS
    server at arrival time; under Background policy they carry deadline=+inf
    (i.e. lowest priority: only run when no periodic job is ready).

Global aperiodic-to-core assignment (TBS): at arrival, the virtual deadline is
computed on *every* core:
    V_d,j(k) = max(A_k, V_d,prev,j) + E_k / U_s,j
and the job is handed to the core minimizing V_d,j(k). This is the "global
TBS allocation" that produces load balancing (plot 3).

Migration: when a running aperiodic job is preempted, we re-evaluate its
virtual deadline across all cores (using each core's *current* V_d,prev and
the *remaining* execution time). If a different core now offers a smaller
projected deadline, the job migrates there and a migration is counted. This
models the dynamic re-assignment allowed by the spec ("preemption -> may
re-evaluate and migrate"). To bound recursive re-scheduling chains we cap
migration cascades to at most `m` hops per event.
"""
import heapq


class Simulator:
    def __init__(self, periodic_tasks, assignment, m, aperiodic_jobs, horizon,
                 policy="tbs", enable_migration=True):
        self.m = m
        self.horizon = horizon
        self.policy = policy  # "tbs" or "background"
        self.enable_migration = enable_migration and policy == "tbs"

        self.periodic_tasks = periodic_tasks
        self.assignment = assignment

        self.core_putil = [0.0] * m
        for t in periodic_tasks:
            self.core_putil[assignment[t["id"]]] += t["U"]
        # TBS server bandwidth per core: capacity left for aperiodic work
        self.Us = [max(1e-6, 1.0 - self.core_putil[c]) for c in range(m)]
        self.vd_prev = [0.0] * m

        self.ready = [[] for _ in range(m)]
        self.current = [None] * m
        self.version = [0] * m
        self.last_update = [0.0] * m
        self.busy_time = [0.0] * m

        self.events = []
        self.seq = 0

        self.periodic_total = 0
        self.periodic_misses = 0
        self.aper_response_times = []
        self.migrations = 0
        self.aper_busy_time = [0.0] * m

        for t in periodic_tasks:
            k = 0
            while k * t["period"] < horizon:
                self._push(k * t["period"], "periodic_release", {"task": t, "k": k})
                k += 1

        for job in aperiodic_jobs:
            if job["arrival"] < horizon:
                self._push(job["arrival"], "aperiodic_arrival", {"job": job})

    def _push(self, time, typ, data):
        self.seq += 1
        heapq.heappush(self.events, (time, self.seq, typ, data))

    def _advance_core(self, c, t):
        cur = self.current[c]
        if cur is not None:
            elapsed = t - self.last_update[c]
            if elapsed > 0:
                cur["remaining"] -= elapsed
                self.busy_time[c] += elapsed
                if cur["type"] == "aperiodic":
                    self.aper_busy_time[c] += elapsed
        self.last_update[c] = t

    def _finish_check(self, c, t):
        cur = self.current[c]
        if cur is not None and cur["remaining"] <= 1e-9:
            if cur["type"] == "periodic":
                self.periodic_total += 1
                if t > cur["deadline"] + 1e-6:
                    self.periodic_misses += 1
            else:
                rt = t - cur["job"]["arrival"]
                self.aper_response_times.append(rt)
            self.current[c] = None
            return True
        return False

    def _best_core_for_migration(self, job, t, exclude_core):
        """Recompute a virtual deadline for an in-flight aperiodic job across all
        cores using its *remaining* execution time; return (core, vd) minimizing it."""
        best_c, best_vd = None, None
        for c in range(self.m):
            vd = max(t, self.vd_prev[c]) + job["remaining"] / self.Us[c]
            if best_vd is None or vd < best_vd:
                best_vd, best_c = vd, c
        return best_c, best_vd

    def _reschedule(self, c, t, depth=0):
        candidates = list(self.ready[c])
        if self.current[c] is not None:
            candidates.append(self.current[c])
        if not candidates:
            self.current[c] = None
            return
        best = min(candidates, key=lambda j: j["deadline"])
        if self.current[c] is not best:
            displaced = self.current[c]
            if best in self.ready[c]:
                self.ready[c].remove(best)
            self.current[c] = best

            if displaced is not None:
                migrated = False
                if (self.enable_migration and displaced["type"] == "aperiodic"
                        and depth < self.m):
                    new_c, new_vd = self._best_core_for_migration(displaced, t, c)
                    if new_c is not None and new_c != c and new_vd < displaced["deadline"] - 1e-9:
                        # Bring the destination core's bookkeeping up to date *before*
                        # inserting the migrated job, otherwise its 'current' job's
                        # remaining time is stale (computed at some earlier
                        # last_update) and comparisons/timing will drift.
                        self._advance_core(new_c, t)
                        self._finish_check(new_c, t)
                        self.vd_prev[new_c] = new_vd
                        displaced["deadline"] = new_vd
                        displaced["core"] = new_c
                        self.migrations += 1
                        self.ready[new_c].append(displaced)
                        migrated = True
                        self._reschedule(new_c, t, depth + 1)
                if not migrated:
                    self.ready[c].append(displaced)

        self.version[c] += 1
        self.last_update[c] = t
        if self.current[c] is not None and self.current[c]["remaining"] > 1e-9:
            self._push(t + self.current[c]["remaining"], "completion_check",
                       {"core": c, "version": self.version[c]})

    def _handle_periodic_release(self, t, data):
        task = data["task"]
        k = data["k"]
        c = self.assignment[task["id"]]
        self._advance_core(c, t)
        self._finish_check(c, t)
        job = {"type": "periodic", "id": f"{task['id']}_{k}", "remaining": task["C"],
               "deadline": t + task["deadline"], "release": t}
        self.ready[c].append(job)
        self._reschedule(c, t)

    def _handle_aperiodic_arrival(self, t, data):
        job = data["job"]
        if self.policy == "tbs":
            best_c, best_vd = None, None
            for c in range(self.m):
                vd = max(t, self.vd_prev[c]) + job["C"] / self.Us[c]
                if best_vd is None or vd < best_vd:
                    best_vd, best_c = vd, c
            self.vd_prev[best_c] = best_vd
            c = best_c
            deadline = best_vd
        else:  # background: lowest priority, never interferes with periodic jobs
            c = min(range(self.m), key=lambda cc: self.core_putil[cc])
            deadline = float("inf")

        self._advance_core(c, t)
        self._finish_check(c, t)
        aj = {"type": "aperiodic", "id": job["id"], "remaining": job["C"],
              "deadline": deadline, "job": job, "core": c}
        self.ready[c].append(aj)
        self._reschedule(c, t)

    def _handle_completion_check(self, t, data):
        c = data["core"]
        ver = data["version"]
        if ver != self.version[c]:
            return  # stale event (job was preempted/migrated before completing)
        self._advance_core(c, t)
        if self._finish_check(c, t):
            self._reschedule(c, t)

    def run(self):
        while self.events:
            t, seq, typ, data = heapq.heappop(self.events)
            if t > self.horizon:
                break
            if typ == "periodic_release":
                self._handle_periodic_release(t, data)
            elif typ == "aperiodic_arrival":
                self._handle_aperiodic_arrival(t, data)
            elif typ == "completion_check":
                self._handle_completion_check(t, data)
        return self.results()

    def results(self):
        miss_rate = (self.periodic_misses / self.periodic_total) if self.periodic_total else 0.0
        mean_rt = (sum(self.aper_response_times) / len(self.aper_response_times)
                   if self.aper_response_times else None)
        # core_total_util = [self.core_putil[c] + self.busy_time[c] / self.horizon
                            # for c in range(self.m)]
        core_total_util = [self.core_putil[c] + self.aper_busy_time[c] / self.horizon 
                           for c in range(self.m)]
        return {
            "periodic_total": self.periodic_total,
            "periodic_misses": self.periodic_misses,
            "miss_rate": miss_rate,
            "aper_response_times": self.aper_response_times,
            "mean_response_time": mean_rt,
            "migrations": self.migrations,
            "n_aperiodic": len(self.aper_response_times),
            "core_periodic_util": list(self.core_putil),
            "core_total_util": core_total_util,
        }
```
