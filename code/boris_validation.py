#!/usr/bin/env python3
"""Validation of a relativistic Boris pusher in the uniform-field baseline.

This is a NEW validation stage.  It does not modify the existing baseline or
convergence scripts, data, figures or notes.

Physical model (unchanged)
--------------------------
One positron, q = +e, in a uniform static magnetic field
B = (0, 0, B0), B0 = 3e-10 T, with zero electric field.  Initial position is
the origin, initial momentum is along +x, and the initial total energy is
10 GeV.  SI units are used throughout.

Integrator
----------
Synchronous relativistic Boris pusher.  Momenta and positions are stored at
integer time levels.  With gamma_n = E_n / (m_e c^2) and

    t_vec = q B dt / (2 m_e gamma_n)              (E = 0, so no half-kicks)

the Boris rotation is

    p'      = p_n + p_n x t_vec
    s_vec   = 2 t_vec / (1 + |t_vec|^2)
    p_{n+1} = p_n + p' x s_vec

and the position is advanced with the time-centered trapezoidal drift

    v_n     = p_n c^2 / E_n
    v_{n+1} = p_{n+1} c^2 / E_{n+1}
    r_{n+1} = r_n + 0.5 (v_n + v_{n+1}) dt

The gamma factor is taken from the evolved momentum at every step; this is a
fully relativistic rotation, not a non-relativistic one.  Momentum is never
renormalised.

Outputs
-------
data/boris_validation.csv
figures/boris_validation.png

Run with:  python3 code/boris_validation.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_simulation as base  # noqa: E402
import baseline_convergence as euler_ref  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Study parameters (physical model inherited from baseline_simulation.py)
# ---------------------------------------------------------------------------
STEPS_PER_PERIOD_LIST = (50, 100, 200, 400)
N_PERIODS = 10

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "boris_validation.csv"
FIGURE_PATH = ROOT / "figures" / "boris_validation.png"
EULER_CSV = ROOT / "data" / "baseline_convergence.csv"

OMEGA = base.OMEGA
R_L = base.R_L
T_GYRO = base.T_GYRO
REST2 = (base.M_E * base.C**2) ** 2
P0 = base.P0
E0 = base.E0

# Validation tolerances (relative).  These are deliberately set at a level
# well above double-precision round-off but far below the Euler errors.
ROUNDOFF_TOL = 1.0e-12


# ---------------------------------------------------------------------------
# Exact analytical uniform-field solution
# ---------------------------------------------------------------------------
def exact_position(t: np.ndarray) -> np.ndarray:
    """Exact position for the initial conditions used here."""
    t = np.asarray(t, dtype=float)
    x = R_L * np.sin(OMEGA * t)
    y = -R_L * (1.0 - np.cos(OMEGA * t))
    z = np.zeros_like(t)
    return np.column_stack([x, y, z])


def exact_momentum(t: np.ndarray) -> np.ndarray:
    """Exact momentum for the initial conditions used here."""
    t = np.asarray(t, dtype=float)
    px = P0 * np.cos(OMEGA * t)
    py = -P0 * np.sin(OMEGA * t)
    pz = np.zeros_like(t)
    return np.column_stack([px, py, pz])


# ---------------------------------------------------------------------------
# Relativistic Boris pusher
# ---------------------------------------------------------------------------
def boris_rotate(p: np.ndarray, t_vec: np.ndarray) -> np.ndarray:
    """Apply the Boris rotation to one momentum vector.

    The rotation angle is theta = 2 arctan(|t_vec|).  This preserves |p|
    exactly in exact arithmetic; the measured change is round-off only.
    """
    p_prime = p + np.cross(p, t_vec)
    s_vec = 2.0 * t_vec / (1.0 + np.dot(t_vec, t_vec))
    return p + np.cross(p_prime, s_vec)


def simulate(steps_per_period: int) -> dict:
    """Run the synchronous relativistic Boris pusher for 10 gyroperiods."""
    dt = T_GYRO / steps_per_period
    n_steps = N_PERIODS * steps_per_period

    x = np.zeros(3)
    p = np.array([P0, 0.0, 0.0])
    b_vec = base.B_VEC

    time = np.empty(n_steps + 1)
    position = np.empty((n_steps + 1, 3))
    momentum = np.empty((n_steps + 1, 3))
    energy = np.empty(n_steps + 1)
    max_step_momentum_change = 0.0

    for n in range(n_steps + 1):
        e_n = base.total_energy(p)
        v_n = p * base.C**2 / e_n

        time[n] = n * dt
        position[n] = x
        momentum[n] = p
        energy[n] = e_n

        if n == n_steps:
            break

        # Relativistic Boris rotation, E = 0.
        gamma_n = e_n / (base.M_E * base.C**2)
        t_vec = base.Q * b_vec * dt / (2.0 * base.M_E * gamma_n)
        p_new = boris_rotate(p, t_vec)

        # Per-step invariant check: the rotation must preserve |p|.
        norm_p = np.linalg.norm(p)
        max_step_momentum_change = max(
            max_step_momentum_change,
            abs(np.linalg.norm(p_new) - norm_p) / norm_p,
        )

        # Time-centered trapezoidal drift at integer time levels.
        e_new = base.total_energy(p_new)
        v_new = p_new * base.C**2 / e_new
        x = x + 0.5 * (v_n + v_new) * dt
        p = p_new

    return {
        "steps_per_period": steps_per_period,
        "dt": dt,
        "n_steps": n_steps,
        "time": time,
        "position": position,
        "momentum": momentum,
        "energy": energy,
        "max_step_momentum_change": max_step_momentum_change,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def phase_error_deg(
    time: np.ndarray, momentum: np.ndarray, position: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulated phase errors of momentum and position (degrees).

    Momentum phase is compared with the exact -omega t.  Position phase is
    measured about the exact gyrocenter (0, -r_L) and compared with the exact
    pi/2 - omega t.  The phases are unwrapped so accumulated multi-turn errors
    are not folded into [-pi, pi].  For Euler the accumulated phase error can
    exceed 360 degrees; that is reported explicitly rather than hidden.
    """
    phase_p = np.unwrap(np.arctan2(momentum[:, 1], momentum[:, 0]))
    phase_p_exact = -OMEGA * time
    d_phase_p = phase_p - phase_p_exact

    phase_r = np.unwrap(np.arctan2(position[:, 1] + R_L, position[:, 0]))
    phase_r_exact = 0.5 * np.pi - OMEGA * time
    d_phase_r = phase_r - phase_r_exact

    return np.degrees(d_phase_p), np.degrees(d_phase_r)


def analyse_run(sim: dict) -> dict:
    f = sim["steps_per_period"]
    dt = sim["dt"]
    time = sim["time"]
    position = sim["position"]
    momentum = sim["momentum"]
    energy = sim["energy"]

    e_ratio = energy / E0
    max_energy_drift = float(np.max(np.abs(e_ratio - 1.0)))
    max_momentum_drift = float(
        np.max(np.abs(np.linalg.norm(momentum, axis=1) / P0 - 1.0))
    )
    final_p_over_p0 = float(np.linalg.norm(momentum[-1]) / P0)

    # Trajectory radius: maximum deviation from the exact Larmor circle
    # centred at (0, -r_L), plus a circle-fit cross-check.
    distance_from_center = np.hypot(position[:, 0], position[:, 1] + R_L) / R_L
    max_radius_deviation = float(np.max(np.abs(distance_from_center - 1.0)))

    fit = base.kasa_circle_fit(position[:, 0], position[:, 1])
    fit_radius_over_r = fit["radius"] / R_L
    fit_radius_error = fit_radius_over_r - 1.0
    fit_rms_over_r = fit["rms_residual"] / R_L

    d_phase_p, d_phase_r = phase_error_deg(time, momentum, position)
    max_phase_error = float(np.max(np.abs(d_phase_p)))
    final_phase_error = float(d_phase_p[-1])
    # Momentum and position phase errors agree for this pusher; keep both and
    # check that in main().
    max_position_phase_error = float(np.max(np.abs(d_phase_r)))

    # Exact-solution position error.
    exact_pos = exact_position(time)
    position_error = np.linalg.norm(position - exact_pos, axis=1) / R_L
    max_position_error = float(np.max(position_error))
    rms_position_error = float(np.sqrt(np.mean(position_error**2)))
    final_position_error = float(position_error[-1])
    final_radial_over_r = float(np.linalg.norm(position[-1]) / R_L)

    max_abs_z = float(np.max(np.abs(position[:, 2])))

    # Analytical phase lag per step: theta_Boris = 2 atan(omega dt / 2)
    # while the exact rotation is omega dt.  Total lag over N steps.
    theta_exact = OMEGA * dt
    theta_boris = 2.0 * np.arctan(0.5 * theta_exact)
    predicted_phase_rad = sim["n_steps"] * (theta_exact - theta_boris)
    predicted_phase_deg = float(np.degrees(predicted_phase_rad))
    phase_prediction_rel_error = (
        max_phase_error / predicted_phase_deg - 1.0
        if predicted_phase_deg != 0.0
        else float("nan")
    )
    # Leading-order asymptotic lag, independent of the exact discrete formula:
    # N * (omega dt)^3 / 12 = 20 pi^3 / (3 f^2) radians.
    leading_phase_deg = float(
        np.degrees(20.0 * np.pi**3 / (3.0 * f**2))
    )
    phase_leading_rel_error = (
        max_phase_error / leading_phase_deg - 1.0
        if leading_phase_deg != 0.0
        else float("nan")
    )

    # A pure phase lag on the exact circle gives this final distance from the
    # origin after an integer number of periods.
    predicted_final_radial = float(2.0 * np.sin(0.5 * predicted_phase_rad))
    final_radial_prediction_rel_error = (
        final_radial_over_r / predicted_final_radial - 1.0
        if predicted_final_radial != 0.0
        else float("nan")
    )

    return {
        "sim": sim,
        "steps_per_period": f,
        "dt": dt,
        "n_steps": sim["n_steps"],
        "total_time": float(time[-1]),
        "max_energy_drift": max_energy_drift,
        "max_momentum_drift": max_momentum_drift,
        "final_p_over_p0": final_p_over_p0,
        "max_step_momentum_change": sim["max_step_momentum_change"],
        "fit_radius_over_r": fit_radius_over_r,
        "fit_radius_error": fit_radius_error,
        "fit_rms_over_r": fit_rms_over_r,
        "max_radius_deviation": max_radius_deviation,
        "fit_cx_over_r": fit["cx"] / R_L,
        "fit_cy_over_r": fit["cy"] / R_L,
        "max_abs_z": max_abs_z,
        "final_radial_over_r": final_radial_over_r,
        "max_phase_error": max_phase_error,
        "final_phase_error": final_phase_error,
        "max_position_phase_error": max_position_phase_error,
        "predicted_phase_deg": predicted_phase_deg,
        "phase_prediction_rel_error": phase_prediction_rel_error,
        "leading_phase_deg": leading_phase_deg,
        "phase_leading_rel_error": phase_leading_rel_error,
        "predicted_final_radial": predicted_final_radial,
        "final_radial_prediction_rel_error": final_radial_prediction_rel_error,
        "max_position_error": max_position_error,
        "rms_position_error": rms_position_error,
        "final_position_error": final_position_error,
    }


def euler_comparison(f: int) -> dict:
    """Metrics from the existing explicit-Euler integrator for comparison."""
    sim = euler_ref.simulate(f)
    time = sim["time"]
    position = sim["position"]
    momentum = sim["momentum"]
    energy = sim["energy"]

    max_energy_drift = float(np.max(np.abs(energy / E0 - 1.0)))
    final_p_over_p0 = float(np.linalg.norm(momentum[-1]) / P0)
    final_radial = float(np.linalg.norm(position[-1]) / R_L)

    fit = base.kasa_circle_fit(
        position[-f - 1 :, 0], position[-f - 1 :, 1]
    )
    fit_radius_over_r = fit["radius"] / R_L

    distance_from_center = np.hypot(position[:, 0], position[:, 1] + R_L) / R_L
    max_radius_deviation = float(np.max(np.abs(distance_from_center - 1.0)))

    d_phase_p, _ = phase_error_deg(time, momentum, position)
    final_phase_error = float(d_phase_p[-1])

    exact_pos = exact_position(time)
    final_position_error = float(
        np.linalg.norm(position[-1] - exact_pos[-1]) / R_L
    )

    return {
        "euler_max_energy_drift": max_energy_drift,
        "euler_final_p_over_p0": final_p_over_p0,
        "euler_final_fit_radius_over_r": fit_radius_over_r,
        "euler_max_radius_deviation": max_radius_deviation,
        "euler_final_radial_over_r": final_radial,
        "euler_final_phase_error": final_phase_error,
        "euler_final_position_error": final_position_error,
    }


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------
CSV_COLUMNS = [
    "steps_per_gyroperiod",
    "dt_s",
    "n_steps",
    "total_time_s",
    "max_abs_rel_energy_drift",
    "max_abs_rel_momentum_drift",
    "final_p_over_p0",
    "max_per_step_rel_momentum_change",
    "trajectory_fit_radius_over_rL",
    "trajectory_fit_radius_error",
    "trajectory_fit_rms_over_rL",
    "max_radius_deviation_from_exact_circle",
    "fit_center_x_over_rL",
    "fit_center_y_over_rL",
    "max_abs_z_m",
    "final_radial_over_rL",
    "max_phase_error_deg",
    "final_phase_error_deg",
    "predicted_phase_error_deg",
    "phase_prediction_rel_error",
    "predicted_phase_error_leading_deg",
    "phase_leading_prediction_rel_error",
    "predicted_final_radial_over_rL",
    "final_radial_prediction_rel_error",
    "max_position_error_over_rL",
    "rms_position_error_over_rL",
    "final_position_error_over_rL",
    "euler_max_abs_rel_energy_drift",
    "euler_final_p_over_p0",
    "euler_final_period_fit_radius_over_rL",
    "euler_max_radius_deviation_from_exact_circle",
    "euler_final_radial_over_rL",
    "euler_final_phase_error_deg",
    "euler_final_position_error_over_rL",
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
                    f"{r['max_energy_drift']:.12e}",
                    f"{r['max_momentum_drift']:.12e}",
                    f"{r['final_p_over_p0']:.12e}",
                    f"{r['max_step_momentum_change']:.12e}",
                    f"{r['fit_radius_over_r']:.12e}",
                    f"{r['fit_radius_error']:.12e}",
                    f"{r['fit_rms_over_r']:.12e}",
                    f"{r['max_radius_deviation']:.12e}",
                    f"{r['fit_cx_over_r']:.12e}",
                    f"{r['fit_cy_over_r']:.12e}",
                    f"{r['max_abs_z']:.12e}",
                    f"{r['final_radial_over_r']:.12e}",
                    f"{r['max_phase_error']:.12e}",
                    f"{r['final_phase_error']:.12e}",
                    f"{r['predicted_phase_deg']:.12e}",
                    f"{r['phase_prediction_rel_error']:.12e}",
                    f"{r['leading_phase_deg']:.12e}",
                    f"{r['phase_leading_rel_error']:.12e}",
                    f"{r['predicted_final_radial']:.12e}",
                    f"{r['final_radial_prediction_rel_error']:.12e}",
                    f"{r['max_position_error']:.12e}",
                    f"{r['rms_position_error']:.12e}",
                    f"{r['final_position_error']:.12e}",
                    f"{r['euler_max_energy_drift']:.12e}",
                    f"{r['euler_final_p_over_p0']:.12e}",
                    f"{r['euler_final_fit_radius_over_r']:.12e}",
                    f"{r['euler_max_radius_deviation']:.12e}",
                    f"{r['euler_final_radial_over_r']:.12e}",
                    f"{r['euler_final_phase_error']:.12e}",
                    f"{r['euler_final_position_error']:.12e}",
                ]
            )


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def plot_validation(results: list[dict]) -> None:
    f_arr = np.array([r["steps_per_period"] for r in results], dtype=float)
    dt_arr = np.array([r["dt"] for r in results])
    energy_drift = np.array([r["max_energy_drift"] for r in results])
    phase = np.array([r["max_phase_error"] for r in results])
    predicted_phase = np.array([r["predicted_phase_deg"] for r in results])
    radius_dev = np.array([r["max_radius_deviation"] for r in results])
    euler_phase = np.array([r["euler_final_phase_error"] for r in results])
    euler_radius = np.array([r["euler_max_radius_deviation"] for r in results])

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 10.0))
    ax_traj, ax_energy, ax_phase, ax_radius = axes.ravel()

    # --- Panel A: trajectory for f = 100 against the exact circle ---------
    target = next(r for r in results if r["steps_per_period"] == 100)
    sim = target["sim"]
    f = target["steps_per_period"]
    x = sim["position"][:, 0] / R_L
    y = sim["position"][:, 1] / R_L

    theta = np.linspace(0.0, 2.0 * np.pi, 720)
    ax_traj.plot(
        np.sin(theta),
        -1.0 + np.cos(theta),
        "--",
        color="0.35",
        lw=1.2,
        label="exact Larmor circle",
    )
    ax_traj.plot(x, y, color="#1f77b4", lw=1.0, label="Boris (100 steps/period)")
    ax_traj.plot(x[0], y[0], "o", color="black", ms=5)
    ax_traj.set_xlabel(r"$x/r_L$")
    ax_traj.set_ylabel(r"$y/r_L$")
    ax_traj.set_title(
        "Trajectory is on the correct circle\n"
        "solid: Boris (100 steps/period), dashed: exact"
    )
    ax_traj.set_aspect("equal", adjustable="datalim")
    ax_traj.grid(True, alpha=0.3)

    # Zoom near the origin at t = 10T: the final point is phase-shifted.
    axins = ax_traj.inset_axes([0.58, 0.56, 0.40, 0.38])
    axins.plot(
        np.sin(theta),
        -1.0 + np.cos(theta),
        "--",
        color="0.65",
        lw=0.9,
    )
    axins.plot(x[-f - 1 :], y[-f - 1 :], color="#1f77b4", lw=1.0)
    axins.plot(0.0, 0.0, "x", color="black", ms=7, label="exact end")
    axins.plot(
        x[-1], y[-1], "s", color="#d62728", ms=5,
        label="Boris end",
    )
    limit = 0.04
    axins.set_xlim(-limit, limit)
    axins.set_ylim(-limit, limit)
    axins.set_aspect("equal")
    axins.grid(True, alpha=0.3)
    axins.tick_params(labelsize=6)
    axins.set_title("final-point zoom", fontsize=7)
    axins.annotate(
        f"offset = {100.0 * target['final_radial_over_r']:.2f}% $r_L$",
        xy=(x[-1], y[-1]),
        xytext=(0.03, 0.80),
        textcoords="axes fraction",
        fontsize=7,
        arrowprops={"arrowstyle": "->", "lw": 0.7},
    )

    # --- Panel B: energy drift versus time -------------------------------
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(results)))
    for r, color in zip(results, colors):
        sim = r["sim"]
        drift = np.maximum(np.abs(sim["energy"] / E0 - 1.0), 1e-17)
        ax_energy.semilogy(
            sim["time"] / T_GYRO,
            drift,
            color=color,
            lw=1.3,
        )

    sim_e = euler_ref.simulate(100)
    euler_drift = np.maximum(np.abs(sim_e["energy"] / E0 - 1.0), 1e-17)
    ax_energy.semilogy(
        sim_e["time"] / T_GYRO,
        euler_drift,
        color="#d62728",
        lw=1.8,
        ls="--",
    )
    ax_energy.axhline(
        ROUNDOFF_TOL,
        color="0.35",
        ls=":",
        lw=1.0,
        label=r"$10^{-12}$ tolerance",
    )
    ax_energy.set_xlabel("time [theoretical gyroperiods]")
    ax_energy.set_ylabel(r"$|E/E_0-1|$")
    ax_energy.set_title("Energy: Boris round-off vs Euler secular growth")
    ax_energy.grid(True, which="both", alpha=0.25)
    ax_energy.set_ylim(1e-17, 30.0)
    ax_energy.annotate(
        "Boris (all f)",
        xy=(4.0, 1.2e-15),
        xytext=(12, 14),
        textcoords="offset points",
        color="#2a6f4e",
        fontsize=9,
    )
    ax_energy.annotate(
        "Euler (100 steps)",
        xy=(8.8, 1.0),
        xytext=(-10, 6),
        textcoords="offset points",
        ha="right",
        color="#d62728",
        fontsize=9,
    )
    ax_energy.text(
        0.8,
        1.5e-12,
        r"$10^{-12}$ tolerance",
        fontsize=8,
        color="0.35",
    )

    # --- Panel C: phase error vs resolution ------------------------------
    ax_phase.loglog(
        f_arr,
        phase,
        "o-",
        color="#1f77b4",
        lw=1.5,
        ms=6,
        label="Boris accumulated phase error",
    )
    f_dense = np.geomspace(f_arr.min() * 0.9, f_arr.max() * 1.1, 200)
    phase_analytic = np.degrees(20.0 * np.pi**3 / (3.0 * f_dense**2))
    ax_phase.loglog(
        f_dense,
        phase_analytic,
        ":",
        color="0.25",
        lw=1.2,
        label=r"Boris prediction $\propto f^{-2}$",
    )
    ax_phase.loglog(
        f_arr,
        euler_phase,
        "s--",
        color="#d62728",
        lw=1.4,
        ms=6,
        label="Euler final phase error",
    )
    ax_phase.set_xlabel("steps per theoretical gyroperiod")
    ax_phase.set_ylabel("accumulated phase error [deg]")
    ax_phase.set_title("Phase: energy conservation does not guarantee phase")
    ax_phase.set_xticks(f_arr)
    ax_phase.set_xticklabels([f"{int(v)}" for v in f_arr])
    ax_phase.grid(True, which="both", alpha=0.25)
    ax_phase.annotate(
        "Boris",
        xy=(f_arr[-1], phase[-1]),
        xytext=(-6, 12),
        textcoords="offset points",
        ha="right",
        color="#1f77b4",
        fontsize=9,
    )
    ax_phase.annotate(
        "Euler",
        xy=(f_arr[-1], euler_phase[-1]),
        xytext=(-6, -14),
        textcoords="offset points",
        ha="right",
        color="#d62728",
        fontsize=9,
    )
    ax_phase.text(
        0.03,
        0.06,
        "dashed: analytic $\\propto f^{-2}$",
        transform=ax_phase.transAxes,
        fontsize=8,
        color="0.25",
    )

    # --- Panel D: radius error vs resolution -----------------------------
    ax_radius.loglog(
        f_arr,
        np.maximum(radius_dev, 1e-17),
        "o-",
        color="#2ca02c",
        lw=1.5,
        ms=6,
        label="Boris max radius deviation",
    )
    ax_radius.loglog(
        f_arr,
        euler_radius,
        "s--",
        color="#d62728",
        lw=1.4,
        ms=6,
        label="Euler max radius deviation",
    )
    ax_radius.axhline(
        ROUNDOFF_TOL,
        color="0.35",
        ls=":",
        lw=1.0,
        label=r"$10^{-12}$ tolerance",
    )
    ax_radius.set_xlabel("steps per theoretical gyroperiod")
    ax_radius.set_ylabel(r"max $|\,|r-r_L|/r_L\,|$")
    ax_radius.set_title("Orbit radius: Boris is at round-off, Euler is not")
    ax_radius.set_xticks(f_arr)
    ax_radius.set_xticklabels([f"{int(v)}" for v in f_arr])
    ax_radius.grid(True, which="both", alpha=0.25)
    ax_radius.annotate(
        "Boris (round-off)",
        xy=(f_arr[-1], radius_dev[-1]),
        xytext=(-6, 14),
        textcoords="offset points",
        ha="right",
        color="#2ca02c",
        fontsize=8.5,
    )
    ax_radius.annotate(
        "Euler",
        xy=(f_arr[-1], euler_radius[-1]),
        xytext=(-6, 12),
        textcoords="offset points",
        ha="right",
        color="#d62728",
        fontsize=9,
    )
    ax_radius.text(
        f_arr.min() * 1.04,
        ROUNDOFF_TOL * 1.35,
        r"$10^{-12}$ tolerance",
        fontsize=8,
        color="0.35",
    )

    fig.suptitle(
        "Relativistic Boris validation: uniform B = $3\\times10^{-10}$ T, "
        "$E_0$ = 10 GeV, 10 gyroperiods",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURE_PATH, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Validation and reporting
# ---------------------------------------------------------------------------
def existing_euler_csv_check(results: list[dict]) -> dict:
    """Cross-check the Euler comparison against the existing convergence CSV."""
    if not EULER_CSV.exists():
        return {"available": False}
    data = np.genfromtxt(EULER_CSV, delimiter=",", names=True)
    by_f = {int(row["steps_per_gyroperiod"]): row for row in data}
    diffs = []
    for r in results:
        f = int(r["steps_per_period"])
        row = by_f[f]
        diffs.append(
            {
                "f": f,
                "energy": r["euler_max_energy_drift"]
                - float(row["max_abs_rel_energy_drift"]),
                "final_p": r["euler_final_p_over_p0"]
                - float(row["final_p_over_p0"]),
                "final_radial": r["euler_final_radial_over_r"]
                - float(row["final_radial_over_rL"]),
                "fit_radius": r["euler_final_fit_radius_over_r"]
                - float(row["final_period_fit_radius_over_rL"]),
            }
        )
    return {"available": True, "diffs": diffs}


def print_summary(results: list[dict], checks: dict) -> None:
    print("=" * 82)
    print("RELATIVISTIC BORIS VALIDATION")
    print("=" * 82)
    print(f"r_L = {R_L:.12e} m   T = {T_GYRO:.12e} s   "
          f"omega = {OMEGA:.12e} rad/s")
    print(f"Each run: {N_PERIODS} gyroperiods, t_end = {N_PERIODS * T_GYRO:.12e} s")
    print("Model: q = +e, B = (0,0,3e-10 T), E = 0, E0 = 10 GeV")
    print()
    header = (
        f"{'f':>4} {'dt [s]':>12} {'N':>5} "
        f"{'max|dE|/E0':>12} {'max|dp|/p0':>12} {'p_f/p0':>9} "
        f"{'maxRadDev':>11} {'maxPhase':>11} {'predPhase':>11} "
        f"{'maxPosErr':>11} {'max|z|':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['steps_per_period']:>4d} {r['dt']:>12.6f} {r['n_steps']:>5d} "
            f"{r['max_energy_drift']:>12.3e} {r['max_momentum_drift']:>12.3e} "
            f"{r['final_p_over_p0']:>9.6f} {r['max_radius_deviation']:>11.3e} "
            f"{r['max_phase_error']:>11.6f} {r['predicted_phase_deg']:>11.6f} "
            f"{r['max_position_error']:>11.3e} {r['max_abs_z']:>8.1e}"
        )
    print()
    print("Radius / trajectory diagnostics")
    for r in results:
        print(
            f"  f={r['steps_per_period']:>3d}: fit R/rL={r['fit_radius_over_r']:.12f} "
            f"(error {r['fit_radius_error']:+.2e}), "
            f"fit RMS/rL={r['fit_rms_over_r']:.2e}, "
            f"center=({r['fit_cx_over_r']:+.2e},{r['fit_cy_over_r']:+.2e}) rL, "
            f"final |r|/rL={r['final_radial_over_r']:.9f}, "
            f"final pos err/rL={r['final_position_error']:.9f}"
        )
    print()
    print("Phase check (accumulated, unwrapped)")
    for r in results:
        print(
            f"  f={r['steps_per_period']:>3d}: max={r['max_phase_error']:.9f} deg, "
            f"final={r['final_phase_error']:+.9f} deg, "
            f"predicted={r['predicted_phase_deg']:.9f} deg, "
            f"pred rel err={r['phase_prediction_rel_error']:+.3e}; "
            f"leading={r['leading_phase_deg']:.9f} deg, "
            f"leading rel err={r['phase_leading_rel_error']:+.3e}; "
            f"position phase={r['max_position_phase_error']:.9f} deg"
        )
    print()
    print("Validation checks")
    all_energy = all(r["max_energy_drift"] < ROUNDOFF_TOL for r in results)
    all_momentum = all(r["max_momentum_drift"] < ROUNDOFF_TOL for r in results)
    all_radius = all(r["max_radius_deviation"] < ROUNDOFF_TOL for r in results)
    all_z = all(r["max_abs_z"] == 0.0 for r in results)
    all_finite = all(
        np.isfinite(r["max_energy_drift"]) and np.isfinite(r["max_phase_error"])
        for r in results
    )
    phase_orders = []
    for a, b in zip(results[:-1], results[1:]):
        ratio = a["max_phase_error"] / b["max_phase_error"]
        phase_orders.append(float(np.log2(ratio)))
    print(f"  Energy drift < {ROUNDOFF_TOL:.0e}:        {all_energy}")
    print(f"  Momentum drift < {ROUNDOFF_TOL:.0e}:      {all_momentum}")
    print(f"  Radius deviation < {ROUNDOFF_TOL:.0e}:    {all_radius}")
    print(f"  max |z| == 0:                    {all_z}")
    print(f"  All metrics finite:              {all_finite}")
    print(f"  Phase error decreases with f:    "
          f"{all(x > 0 for x in phase_orders)}")
    print(f"  Observed phase orders (f doubling): "
          f"{', '.join(f'{o:.3f}' for o in phase_orders)}")
    print(f"  Momentum and position phase agree: "
          f"{all(abs(r['max_phase_error'] - r['max_position_phase_error']) < 1e-9 for r in results)}")
    print(f"  Max per-step |d|p||/p0:          "
          f"{max(r['max_step_momentum_change'] for r in results):.3e}")
    print("  Phase error below 0.1 deg:       "
          + ", ".join(f"f={r['steps_per_period']}:{r['max_phase_error'] < 0.1}"
                      for r in results))
    print("  Phase error below 1 deg:         "
          + ", ".join(f"f={r['steps_per_period']}:{r['max_phase_error'] < 1.0}"
                      for r in results))
    print()
    print("Energy / radius / phase comparison with explicit Euler")
    print(
        f"{'f':>4} {'Boris dE':>11} {'Euler dE':>11} "
        f"{'Boris RadDev':>13} {'Euler RadDev':>13} "
        f"{'Boris phase':>12} {'Euler phase':>13}"
    )
    for r in results:
        print(
            f"{r['steps_per_period']:>4d} {r['max_energy_drift']:>11.3e} "
            f"{r['euler_max_energy_drift']:>11.3e} "
            f"{r['max_radius_deviation']:>13.3e} "
            f"{r['euler_max_radius_deviation']:>13.3e} "
            f"{r['max_phase_error']:>12.6f} "
            f"{r['euler_final_phase_error']:>13.3f}"
        )
    print()
    if checks.get("available"):
        print("Euler cross-check against existing data/baseline_convergence.csv:")
        worst = {k: 0.0 for k in ("energy", "final_p", "final_radial", "fit_radius")}
        for d in checks["diffs"]:
            for k in worst:
                worst[k] = max(worst[k], abs(d[k]))
        for k, v in worst.items():
            print(f"  max |difference| in {k}: {v:.3e}")
    else:
        print("Existing Euler convergence CSV not found; Euler comparison "
              "used the existing integrator only.")
    print()
    print("Interpretation: Boris conserves energy, momentum magnitude and")
    print("orbit radius to double-precision round-off, but its half-angle")
    print("rotation under-rotates by about (omega dt)^3/12 per step.  The")
    print("result is an O(dt^2) phase lag: the orbit is the right circle,")
    print("but the particle is at the wrong phase.  Euler fails in both")
    print("energy and phase.")
    print("=" * 82)


def main() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for f in STEPS_PER_PERIOD_LIST:
        metrics = analyse_run(simulate(f))
        metrics.update(euler_comparison(f))
        results.append(metrics)

    # Euler CSV cross-check.
    checks = existing_euler_csv_check(results)

    write_csv(results)
    plot_validation(results)
    print_summary(results, checks)

    print()
    print(f"Wrote {DATA_PATH}")
    print(f"Wrote {FIGURE_PATH}")


if __name__ == "__main__":
    main()
