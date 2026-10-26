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
