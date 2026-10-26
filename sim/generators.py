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
