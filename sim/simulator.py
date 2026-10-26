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
