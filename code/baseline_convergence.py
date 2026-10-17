#!/usr/bin/env python3
"""Timestep convergence study for the baseline explicit-Euler integrator.

This script does not modify the baseline implementation.  It imports the
baseline physical constants and helpers from ``baseline_simulation.py`` and
repeats the *same* explicit-Euler update with different numbers of steps per
theoretical gyroperiod: 50, 100, 200 and 400.  Every run covers 10 theoretical
gyroperiods, so the total simulated time is always 10 T.

The model is unchanged: one positron, q = +e, uniform static B = (0,0,B0),
zero electric field, initial momentum along +x, E0 = 10 GeV.  Momentum is never
renormalised and energy conservation is never enforced.

Outputs
-------
data/baseline_convergence.csv
figures/baseline_convergence.png

Run with:  python3 code/baseline_convergence.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

# Import the existing baseline model without running its main().
sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_simulation as base  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Study parameters (physical model is inherited from baseline_simulation.py)
# ---------------------------------------------------------------------------
STEPS_PER_PERIOD_LIST = (50, 100, 200, 400)
N_PERIODS = 10
STUDY_STEPS = [f * N_PERIODS for f in STEPS_PER_PERIOD_LIST]

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "baseline_convergence.csv"
FIGURE_PATH = ROOT / "figures" / "baseline_convergence.png"
BASELINE_CSV = ROOT / "data" / "baseline_simulation.csv"

OMEGA0 = base.OMEGA
R_L = base.R_L
T_GYRO = base.T_GYRO
REST2 = (base.M_E * base.C**2) ** 2


# ---------------------------------------------------------------------------
# Integration: identical explicit-Euler update, parameterised by step count
# ---------------------------------------------------------------------------
def simulate(steps_per_period: int) -> dict:
    """Repeat the baseline explicit-Euler integration for one step count.

    Update rule (identical to the baseline):
        p_{n+1} = p_n + dt * q * (v_n x B)
        r_{n+1} = r_n + dt * v_n
    with v_n = p_n c^2 / E_n and E_n = sqrt(p_n^2 c^2 + m_e^2 c^4).
    """
    dt = T_GYRO / steps_per_period
    n_steps = N_PERIODS * steps_per_period

    r = np.zeros(3)
    p = np.array([base.P0, 0.0, 0.0])

    time = np.empty(n_steps + 1)
    position = np.empty((n_steps + 1, 3))
    momentum = np.empty((n_steps + 1, 3))
    energy = np.empty(n_steps + 1)

    for n in range(n_steps + 1):
        e_n = base.total_energy(p)
        v_n = p * base.C**2 / e_n

        time[n] = n * dt
        position[n] = r
        momentum[n] = p
        energy[n] = e_n

        if n == n_steps:
            break

        p = p + dt * base.Q * np.cross(v_n, base.B_VEC)
        r = r + dt * v_n

    return {
        "steps_per_period": steps_per_period,
        "dt": dt,
        "n_steps": n_steps,
        "time": time,
        "position": position,
        "momentum": momentum,
        "energy": energy,
    }


# ---------------------------------------------------------------------------
# Metrics and analytical checks
# ---------------------------------------------------------------------------
def scalar_recurrence_prediction(n_steps: int, dt: float) -> float:
    """Exact discrete prediction for |p_final| from the same update rule.

    Because the momentum increment is perpendicular to the current momentum,
        p_{n+1}^2 = p_n^2 + (q B c^2 dt)^2 p_n^2 / E_n^2,
    which gives the closed scalar recurrence
        E_{n+1}^2 = E_n^2 + (q B c^2 dt)^2 [1 - (m_e c^2)^2 / E_n^2].
    This is an independent check of the vector cross-product integration.
    """
    amplitude = (base.Q * base.B0 * base.C**2 * dt) ** 2
    e2 = base.E0**2
    for _ in range(n_steps):
        e2 = e2 + amplitude * (1.0 - REST2 / e2)
    return float(np.sqrt(max(e2 - REST2, 0.0)) / base.C / base.P0)


def leading_order_prediction(steps_per_period: int) -> float:
    """Leading-order Euler prediction p_N/p0 = sqrt(1 + N (omega0 dt)^2).

    With N = 10 f, dt = T/f and omega0 T = 2 pi this reduces to
    sqrt(1 + 40 pi^2 / f).
    """
    return float(np.sqrt(1.0 + 40.0 * np.pi**2 / steps_per_period))


def analyse_run(sim: dict) -> dict:
    f = sim["steps_per_period"]
    dt = sim["dt"]
    position = sim["position"]
    momentum = sim["momentum"]
    energy = sim["energy"]

    rel_drift = (energy - base.E0) / base.E0
    max_abs_drift = float(np.max(np.abs(rel_drift)))
    final_drift = float(rel_drift[-1])

    final_p = float(np.linalg.norm(momentum[-1]))
    final_p_over_p0 = final_p / base.P0
    final_radial = float(np.linalg.norm(position[-1]))
    final_radial_over_r = final_radial / R_L

    max_abs_z = float(np.max(np.abs(position[:, 2])))
    energy_monotonic = bool(np.all(np.diff(energy) > 0.0))

    # Final-period circle fit: an orbital-size diagnostic that is much less
    # phase-sensitive than the single final position.
    final_fit = base.kasa_circle_fit(
        position[-f - 1 :, 0], position[-f - 1 :, 1]
    )
    dx = position[-1, 0] - final_fit["cx"]
    dy = position[-1, 1] - final_fit["cy"]
    final_phase_deg = float(np.degrees(np.arctan2(dy, dx)))

    predicted_p = leading_order_prediction(f)
    prediction_rel_error = final_p_over_p0 / predicted_p - 1.0

    p_scalar_over_p0 = scalar_recurrence_prediction(sim["n_steps"], dt)
    scalar_rel_error = final_p_over_p0 / p_scalar_over_p0 - 1.0

    predicted_drift = predicted_p - 1.0
    energy_prediction_rel_error = max_abs_drift / predicted_drift - 1.0

    return {
        "steps_per_period": f,
        "dt": dt,
        "n_steps": sim["n_steps"],
        "total_time": float(sim["time"][-1]),
        "max_abs_rel_energy_drift": max_abs_drift,
        "final_rel_energy_drift": final_drift,
        "final_energy_gev": float(energy[-1] / (1.0e9 * base.E_CHARGE)),
        "final_p_over_p0": final_p_over_p0,
        "final_radial_m": final_radial,
        "final_radial_over_rL": final_radial_over_r,
        "final_period_fit_radius_over_rL": final_fit["radius"] / R_L,
        "final_phase_deg": final_phase_deg,
        "max_abs_z_m": max_abs_z,
        "predicted_p_over_p0": predicted_p,
        "prediction_rel_error": prediction_rel_error,
        "scalar_recurrence_p_over_p0": p_scalar_over_p0,
        "scalar_recurrence_rel_error": scalar_rel_error,
        "predicted_max_rel_energy_drift": predicted_drift,
        "energy_prediction_rel_error": energy_prediction_rel_error,
        "energy_monotonic": energy_monotonic,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
CSV_COLUMNS = [
    "steps_per_gyroperiod",
    "dt_s",
    "n_steps",
    "total_time_s",
    "max_abs_rel_energy_drift",
    "final_rel_energy_drift",
    "final_energy_GeV",
    "final_p_over_p0",
    "final_radial_distance_m",
    "final_radial_over_rL",
    "final_period_fit_radius_over_rL",
    "final_phase_angle_deg",
    "max_abs_z_m",
    "predicted_p_over_p0_leading",
    "prediction_rel_error",
    "scalar_recurrence_p_over_p0",
    "scalar_recurrence_rel_error",
    "predicted_max_rel_energy_drift_leading",
    "energy_prediction_rel_error",
    "energy_monotonic",
]


def write_csv(results: list[dict]) -> None:
    with DATA_PATH.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for r in results:
            writer.writerow(
                [
                    r["steps_per_period"],
                    f"{r['dt']:.12e}",
                    r["n_steps"],
                    f"{r['total_time']:.12e}",
                    f"{r['max_abs_rel_energy_drift']:.12e}",
                    f"{r['final_rel_energy_drift']:.12e}",
                    f"{r['final_energy_gev']:.12e}",
                    f"{r['final_p_over_p0']:.12e}",
                    f"{r['final_radial_m']:.12e}",
                    f"{r['final_radial_over_rL']:.12e}",
                    f"{r['final_period_fit_radius_over_rL']:.12e}",
                    f"{r['final_phase_deg']:.12e}",
                    f"{r['max_abs_z_m']:.12e}",
                    f"{r['predicted_p_over_p0']:.12e}",
                    f"{r['prediction_rel_error']:.12e}",
                    f"{r['scalar_recurrence_p_over_p0']:.12e}",
                    f"{r['scalar_recurrence_rel_error']:.12e}",
                    f"{r['predicted_max_rel_energy_drift']:.12e}",
                    f"{r['energy_prediction_rel_error']:.12e}",
                    str(r["energy_monotonic"]),
                ]
            )


def convergence_statistics(results: list[dict]) -> list[dict]:
    """Observed order when the step count doubles (dt halves)."""
    stats = []
    for previous, current in zip(results[:-1], results[1:]):
        drift_ratio = (
            previous["max_abs_rel_energy_drift"]
            / current["max_abs_rel_energy_drift"]
        )
        orbit_error_ratio = (
            (previous["final_period_fit_radius_over_rL"] - 1.0)
            / (current["final_period_fit_radius_over_rL"] - 1.0)
        )
        stats.append(
            {
                "from": previous["steps_per_period"],
                "to": current["steps_per_period"],
                "drift_ratio": drift_ratio,
                "observed_order": float(np.log2(drift_ratio)),
                "orbit_error_ratio": orbit_error_ratio,
                "orbit_observed_order": float(np.log2(orbit_error_ratio)),
            }
        )
    return stats


def plot_convergence(results: list[dict]) -> None:
    f = np.array([r["steps_per_period"] for r in results], dtype=float)
    dt = np.array([r["dt"] for r in results])
    drift = np.array([r["max_abs_rel_energy_drift"] for r in results])
    p_error = np.array([r["final_p_over_p0"] - 1.0 for r in results])
    radial = np.array([r["final_radial_over_rL"] for r in results])
    fit_radius = np.array(
        [r["final_period_fit_radius_over_rL"] for r in results]
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.2))

    # --- Left: energy/momentum error convergence -------------------------
    ax1.loglog(dt, drift, "o-", color="#d62728", lw=1.5, ms=7,
               label=r"max $|(E-E_0)/E_0|$")
    ax1.loglog(dt, p_error, "s--", color="#1f77b4", lw=1.5, ms=6,
               label=r"$p_{\rm final}/p_0 - 1$")

    dt_dense = np.geomspace(dt.min() * 0.8, dt.max() * 1.2, 300)
    f_dense = T_GYRO / dt_dense
    analytic = np.sqrt(1.0 + 40.0 * np.pi**2 / f_dense) - 1.0
    ax1.loglog(dt_dense, analytic, ":", color="0.25", lw=1.4,
               label="leading-order Euler prediction")

    # First-order reference anchored at the 100-step point.
    anchor = 1
    ax1.loglog(dt_dense, drift[anchor] * dt_dense / dt[anchor], "--",
               color="0.6", lw=1.1,
               label="slope 1 reference (anchored at 100)")

    ax1.set_xlabel(r"time step $\Delta t$ [s]")
    ax1.set_ylabel("relative error")
    ax1.set_title("Energy and momentum errors converge")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(fontsize=8.5, loc="lower right")

    orders = []
    for previous, current in zip(drift[:-1], drift[1:]):
        orders.append(np.log2(previous / current))
    ax1.text(
        0.03,
        0.95,
        "observed order (doubling steps): "
        + ", ".join(f"{o:.2f}" for o in orders),
        transform=ax1.transAxes,
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.85},
    )

    # --- Right: final radial distance vs orbit size ----------------------
    ax2.semilogx(f, radial, "o-", color="#d62728", lw=1.5, ms=7,
                 label=r"$|\mathbf{r}(t=10T)|\,/\,r_L$ (phase-sensitive)")
    ax2.semilogx(f, fit_radius, "s-", color="#2ca02c", lw=1.5, ms=6,
                 label="final-period fitted radius / $r_L$")
    ax2.axhline(1.0, color="0.4", ls="--", lw=1.0,
                label="theoretical $r_L$")

    for i, (xi, yi) in enumerate(zip(f, radial)):
        right_edge = i == len(f) - 1
        ax2.annotate(
            f"{yi:.3f}",
            (xi, yi),
            textcoords="offset points",
            xytext=(-8 if right_edge else 8, 12),
            ha="right" if right_edge else "left",
            fontsize=8,
            color="#d62728",
        )
    for i, (xi, yi) in enumerate(zip(f, fit_radius)):
        right_edge = i == len(f) - 1
        ax2.annotate(
            f"{yi:.3f}",
            (xi, yi),
            textcoords="offset points",
            xytext=(-8 if right_edge else 8, -18),
            ha="right" if right_edge else "left",
            fontsize=8,
            color="#2ca02c",
        )

    ax2.set_xticks(f)
    ax2.set_xticklabels([f"{int(v)}" for v in f])
    ax2.set_xlim(38, 560)
    ax2.set_xlabel("steps per theoretical gyroperiod")
    ax2.set_ylabel(r"radius / $r_L$")
    ax2.set_title("Final radial distance is not a clean error metric")
    ax2.set_ylim(1.0, 3.35)
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(fontsize=8.5, loc="lower left", framealpha=0.95)

    fig.suptitle(
        "Explicit-Euler timestep convergence: 10 gyroperiods per run "
        "(B = $3\\times10^{-10}$ T, $E_0$ = 10 GeV)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURE_PATH, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def baseline_check(results: list[dict]) -> dict:
    """Check that the 100-step convergence run reproduces the baseline CSV."""
    if not BASELINE_CSV.exists():
        return {"available": False}
    data = np.genfromtxt(BASELINE_CSV, delimiter=",", names=True)
    baseline_max_drift = float(
        np.max(np.abs((data["energy_J"] - base.E0) / base.E0))
    )
    run100 = next(r for r in results if r["steps_per_period"] == 100)
    return {
        "available": True,
        "max_drift_diff": run100["max_abs_rel_energy_drift"]
        - baseline_max_drift,
        "final_p_diff": run100["final_p_over_p0"]
        - float(data["p_magnitude_kg_m_s"][-1] / data["p_magnitude_kg_m_s"][0]),
        "final_radial_diff": run100["final_radial_over_rL"]
        - float(data["radial_distance_m"][-1] / R_L),
        "max_z_diff": run100["max_abs_z_m"]
        - float(np.max(np.abs(data["z_m"]))),
    }


def print_summary(results: list[dict], stats: list[dict], check: dict) -> None:
    print("=" * 78)
    print("EXPLICIT-EULER TIMESTEP CONVERGENCE STUDY")
    print("=" * 78)
    print(f"r_L = {R_L:.12e} m   T = {T_GYRO:.12e} s   "
          f"omega0 = {OMEGA0:.12e} rad/s")
    print(f"Each run: {N_PERIODS} theoretical gyroperiods "
          f"(t_end = {N_PERIODS * T_GYRO:.12e} s)")
    print()
    header = (
        f"{'f':>4} {'dt [s]':>12} {'N':>5} "
        f"{'max|dE|/E0':>14} {'p_f/p0':>14} "
        f"{'|r_f|/rL':>12} {'fitR/rL':>10} {'max|z|':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['steps_per_period']:>4d} {r['dt']:>12.6f} "
            f"{r['n_steps']:>5d} {r['max_abs_rel_energy_drift']:>14.9e} "
            f"{r['final_p_over_p0']:>14.9f} "
            f"{r['final_radial_over_rL']:>12.6f} "
            f"{r['final_period_fit_radius_over_rL']:>10.6f} "
            f"{r['max_abs_z_m']:>9.1e}"
        )
    print()
    print("Analytical comparison (final p/p0)")
    print(
        f"{'f':>4} {'numerical':>18} {'leading-order':>18} "
        f"{'rel. error':>13} {'scalar recurrence':>18} {'rel. error':>13}"
    )
    for r in results:
        print(
            f"{r['steps_per_period']:>4d} "
            f"{r['final_p_over_p0']:>18.12f} "
            f"{r['predicted_p_over_p0']:>18.12f} "
            f"{r['prediction_rel_error']:>+13.3e} "
            f"{r['scalar_recurrence_p_over_p0']:>18.12f} "
            f"{r['scalar_recurrence_rel_error']:>+13.3e}"
        )
    print()
    print("Energy drift prediction (leading order)")
    for r in results:
        print(
            f"  f={r['steps_per_period']:>3d}: observed="
            f"{r['max_abs_rel_energy_drift']:.12f}  predicted="
            f"{r['predicted_max_rel_energy_drift']:.12f}  "
            f"relative error={r['energy_prediction_rel_error']:+.3e}"
        )
    print()
    print("Observed convergence when the step count doubles")
    for s in stats:
        print(
            f"  {s['from']:>3d} -> {s['to']:>3d}: "
            f"drift ratio={s['drift_ratio']:.6f} "
            f"(order {s['observed_order']:.4f}); "
            f"orbit-size error ratio={s['orbit_error_ratio']:.6f} "
            f"(order {s['orbit_observed_order']:.4f})"
        )
    print()
    print("Final radial distance (phase-sensitive, NOT monotonic):")
    for r in results:
        print(
            f"  f={r['steps_per_period']:>3d}: "
            f"|r(10T)|/rL={r['final_radial_over_rL']:.9f}  "
            f"final-period fit radius/rL="
            f"{r['final_period_fit_radius_over_rL']:.9f}  "
            f"final phase={r['final_phase_deg']:+.3f} deg"
        )
    print()
    if check.get("available"):
        print("Baseline reproduction check (100 steps vs baseline CSV):")
        print(f"  max drift difference   : {check['max_drift_diff']:+.3e}")
        print(f"  final p/p0 difference  : {check['final_p_diff']:+.3e}")
        print(f"  final radial difference: {check['final_radial_diff']:+.3e}")
        print(f"  max |z| difference     : {check['max_z_diff']:+.3e}")
    print()
    print("Interpretation: energy and momentum errors decrease monotonically")
    print("with smaller dt, approaching first-order convergence.  The final")
    print("radial distance does not decrease monotonically because it depends")
    print("on the accumulated phase at the fixed endpoint t=10T; it is not an")
    print("error norm.  The final-period fitted orbital radius, which is")
    print("phase-insensitive, does decrease monotonically toward r_L.")
    print("=" * 78)


def main() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for f in STEPS_PER_PERIOD_LIST:
        sim = simulate(f)
        results.append(analyse_run(sim))

    stats = convergence_statistics(results)
    check = baseline_check(results)

    write_csv(results)
    plot_convergence(results)
    print_summary(results, stats, check)

    print()
    print(f"Wrote {DATA_PATH}")
    print(f"Wrote {FIGURE_PATH}")


if __name__ == "__main__":
    main()
