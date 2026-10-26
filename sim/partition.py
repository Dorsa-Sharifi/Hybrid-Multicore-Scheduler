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
