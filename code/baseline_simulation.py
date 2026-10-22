#!/usr/bin/env python3
"""Baseline simulation: one relativistic positron in a uniform static B field.

Physical model
--------------
A single positron (q = +e) moves in a uniform, static magnetic field
B = (0, 0, B0) with B0 = 3e-10 T.  There is no electric field and no
energy-loss or scattering term.  The initial position is the origin and the
initial momentum is along +x with a total energy of 10 GeV.

The relativistic equations of motion are integrated with the explicit
(Euler) method.  Momentum is *not* renormalised and energy conservation is
*not* enforced at any step; the energy drift reported below is therefore a
genuine property of the discretisation, not something removed by hand.

Outputs
-------
data/baseline_simulation.csv
figures/baseline_trajectory.png
figures/baseline_energy.png

Run with:  python3 code/baseline_simulation.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Matplotlib needs a writable cache directory; use a temporary one so the
# repository is not polluted by cache files.
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "crp-mpl-cache")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# Physical constants (using the SI units)

E_CHARGE = 1.602176634e-19  # elementary charge [C]
M_E = 9.1093837015e-31  # electron/positron mass [kg]
C = 299792458.0  # speed of light in vacuum [m/s]

# Model parameters

B0 = 3.0e-10  # magnetic field strength along +z [T]
B_VEC = np.array([0.0, 0.0, B0])
E_FIELD = np.array([0.0, 0.0, 0.0])  # no electric field [V/m]
Q = +E_CHARGE  # positron charge [C]

E0_GEV = 10.0  # initial total energy [GeV]
E0 = E0_GEV * 1.0e9 * E_CHARGE  # initial total energy [J]

STEPS_PER_PERIOD = 100  # time steps per theoretical gyroperiod
N_PERIODS = 10  # number of theoretical gyroperiods to simulate
N_STEPS = STEPS_PER_PERIOD * N_PERIODS

# Repository paths, resolved from this file so the script can be run anywhere.
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"
CSV_PATH = DATA_DIR / "baseline_simulation.csv"
TRAJECTORY_PATH = FIGURE_DIR / "baseline_trajectory.png"
ENERGY_PATH = FIGURE_DIR / "baseline_energy.png"


# Theoretical calculations

def momentum_from_energy(energy: float) -> float:
    """Return |p| from E_total^2 = p^2 c^2 + m_e^2 c^4."""
    return float(np.sqrt(energy**2 - (M_E * C**2) ** 2) / C)


def total_energy(momentum: np.ndarray) -> float:
    """Return E_total from the momentum vector."""
    return float(np.sqrt(np.dot(momentum, momentum) * C**2 + (M_E * C**2) ** 2))


def velocity_from_momentum(momentum: np.ndarray) -> np.ndarray:
    """Return v = p c^2 / E_total (component-wise)."""
    return momentum * C**2 / total_energy(momentum)


def larmor_radius(momentum_perp: float) -> float:
    """Theoretical Larmor radius r_L = p_perp / (|q| B)."""
    return momentum_perp / (abs(Q) * B0)


def gyroperiod(energy: float) -> float:
    """Theoretical gyroperiod T = 2 pi gamma m_e / (|q| B).

    For the initial energy only; this is the period used to choose dt.
    """
    gamma = energy / (M_E * C**2)
    return 2.0 * np.pi * gamma * M_E / (abs(Q) * B0)


P0 = momentum_from_energy(E0)
V0 = velocity_from_momentum(np.array([P0, 0.0, 0.0]))
R_L = larmor_radius(P0)
OMEGA = abs(Q) * B0 * C**2 / E0  # relativistic gyroangular frequency gamma*m
T_GYRO = gyroperiod(E0)
DT = T_GYRO / STEPS_PER_PERIOD
T_END = N_PERIODS * T_GYRO


# Integration: explicit Euler

def simulate() -> dict:
    """Integrate dp/dt = q (v x B) and dr/dt = v with explicit Euler.

    Update order at step n:
        p_{n+1} = p_n + dt * q * (v_n x B)
        r_{n+1} = r_n + dt * v_n
    with v_n = p_n c^2 / E_n and E_n = sqrt(p_n^2 c^2 + m_e^2 c^4).

    The momentum is never rescaled after the update.
    """
    r = np.zeros(3)
    p = np.array([P0, 0.0, 0.0])

    time = np.empty(N_STEPS + 1)
    position = np.empty((N_STEPS + 1, 3))
    momentum = np.empty((N_STEPS + 1, 3))
    energy = np.empty(N_STEPS + 1)
    speed = np.empty(N_STEPS + 1)

    for n in range(N_STEPS + 1):
        e_n = total_energy(p)
        v_n = p * C**2 / e_n

        time[n] = n * DT
        position[n] = r
        momentum[n] = p
        energy[n] = e_n
        speed[n] = float(np.linalg.norm(v_n))

        if n == N_STEPS:
            break

        # Explicit Euler.  The force uses the *old* velocity and the position
        # uses the *old* velocity; nothing is corrected or renormalised.
        accel_term = Q * np.cross(v_n, B_VEC)  # dp/dt (Electricity field is zero)
        p = p + DT * accel_term
        r = r + DT * v_n

    return {
        "time": time,
        "position": position,
        "momentum": momentum,
        "energy": energy,
        "speed": speed,
    }


# Geometry diagnostics

def kasa_circle_fit(x: np.ndarray, y: np.ndarray) -> dict:
    """Algebraic (Kasa) least-squares circle fit.

    Minimises sum_i [(x_i - cx)^2 + (y_i - cy)^2 - R^2]^2.  The RMS radial
    residual is returned as an indicator of how poorly a circle describes the
    fitted points (it is large when the trajectory spirals).
    """
    a = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    b = x * x + y * y
    solution, *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy = float(solution[0]), float(solution[1])
    radius = float(np.sqrt(solution[2] + cx * cx + cy * cy))

    residuals = np.hypot(x - cx, y - cy) - radius
    rms = float(np.sqrt(np.mean(residuals**2)))
    return {"cx": cx, "cy": cy, "radius": radius, "rms_residual": rms}


def circumradius(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Radius of the circle through three points (local osculating radius)."""
    side_ab = np.linalg.norm(b - a)
    side_bc = np.linalg.norm(c - b)
    side_ca = np.linalg.norm(c - a)
    twice_area = np.linalg.norm(np.cross(b - a, c - a))
    if twice_area == 0.0:
        return float("nan")
    # R = abc / (4 A) and 4 A = 2 * twice_area.
    return float(side_ab * side_bc * side_ca / (2.0 * twice_area))


def compute_diagnostics(sim: dict) -> dict:
    time = sim["time"]
    position = sim["position"]
    momentum = sim["momentum"]
    energy = sim["energy"]
    speed = sim["speed"]

    x = position[:, 0]
    y = position[:, 1]
    radial = np.hypot(x, y)
    momentum_magnitude = np.linalg.norm(momentum, axis=1)

    rel_energy_drift = (energy - E0) / E0
    max_abs_rel_drift = float(np.max(np.abs(rel_energy_drift)))
    max_signed_rel_drift = float(np.max(rel_energy_drift))
    final_rel_drift = float(rel_energy_drift[-1])
    energy_monotonic = bool(np.all(np.diff(energy) > 0.0))

    # Local osculating radius from consecutive triples of positions.
    local_radius = np.array(
        [
            circumradius(position[i], position[i + 1], position[i + 2])
            for i in range(N_STEPS - 1)
        ]
    )
    # Triples fully inside the first gyroperiod: start indices 0..98.
    local_first_period = local_radius[: STEPS_PER_PERIOD - 1]

    # Circle fits: first period, every individual period, and whole run.
    first_period_fit = kasa_circle_fit(
        x[: STEPS_PER_PERIOD + 1], y[: STEPS_PER_PERIOD + 1]
    )
    period_fits = []
    for k in range(N_PERIODS):
        lo = k * STEPS_PER_PERIOD
        hi = (k + 1) * STEPS_PER_PERIOD + 1
        period_fits.append(kasa_circle_fit(x[lo:hi], y[lo:hi]))
    whole_run_fit = kasa_circle_fit(x, y)

    radius_first_fit = first_period_fit["radius"]
    radius_first_fit_rel = (radius_first_fit - R_L) / R_L
    radius_local_initial = float(local_radius[0])
    radius_local_initial_rel = (radius_local_initial - R_L) / R_L
    radius_local_median_first = float(np.median(local_first_period))
    radius_local_median_first_rel = (radius_local_median_first - R_L) / R_L

    return {
        "p0": P0,
        "v0": float(np.linalg.norm(V0)),
        "v0_over_c": float(np.linalg.norm(V0) / C),
        "gamma0": E0 / (M_E * C**2),
        "r_larmor": R_L,
        "omega": OMEGA,
        "gyroperiod": T_GYRO,
        "dt": DT,
        "t_end": T_END,
        "n_steps": N_STEPS,
        "steps_per_period": STEPS_PER_PERIOD,
        "max_abs_rel_energy_drift": max_abs_rel_drift,
        "max_signed_rel_energy_drift": max_signed_rel_drift,
        "final_rel_energy_drift": final_rel_drift,
        "final_energy_j": float(energy[-1]),
        "final_energy_gev": float(energy[-1] / (1e9 * E_CHARGE)),
        "energy_monotonic": energy_monotonic,
        "max_abs_z": float(np.max(np.abs(position[:, 2]))),
        "radial_min": float(np.min(radial)),
        "radial_max": float(np.max(radial)),
        "radial_max_over_r": float(np.max(radial) / R_L),
        "fit_first": first_period_fit,
        "radius_first_fit": radius_first_fit,
        "radius_first_fit_rel": radius_first_fit_rel,
        "radius_local_initial": radius_local_initial,
        "radius_local_initial_rel": radius_local_initial_rel,
        "radius_local_median_first": radius_local_median_first,
        "radius_local_median_first_rel": radius_local_median_first_rel,
        "period_fits": period_fits,
        "radius_whole_fit": whole_run_fit["radius"],
        "radius_whole_fit_rel": (whole_run_fit["radius"] - R_L) / R_L,
        "whole_fit_rms": whole_run_fit["rms_residual"],
        "sim": sim,
        "radial": radial,
        "momentum_magnitude": momentum_magnitude,
        "rel_energy_drift": rel_energy_drift,
    }


# Output writers

def write_csv(diag: dict) -> None:
    sim = diag["sim"]
    time = sim["time"]
    position = sim["position"]
    momentum = sim["momentum"]
    energy = sim["energy"]
    speed = sim["speed"]

    columns = np.column_stack(
        [
            time,
            time / T_GYRO,
            position[:, 0],
            position[:, 1],
            position[:, 2],
            momentum[:, 0],
            momentum[:, 1],
            momentum[:, 2],
            diag["momentum_magnitude"],
            speed,
            energy,
            energy / (1.0e9 * E_CHARGE),
            diag["radial"],
        ]
    )
    header = (
        "time_s,time_gyroperiods,x_m,y_m,z_m,"
        "px_kg_m_s,py_kg_m_s,pz_kg_m_s,p_magnitude_kg_m_s,"
        "speed_m_s,energy_J,energy_GeV,radial_distance_m"
    )
    np.savetxt(
        CSV_PATH,
        columns,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.12e",
    )


def plot_trajectory(diag: dict) -> None:
    sim = diag["sim"]
    x = sim["position"][:, 0] / R_L
    y = sim["position"][:, 1] / R_L

    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    ax.plot(x, y, color="#1f77b4", linewidth=1.4, label="Explicit Euler trajectory")

    theta = np.linspace(0.0, 2.0 * np.pi, 720)
    ax.plot(
        np.sin(theta),
        -1.0 + np.cos(theta),
        linestyle="--",
        color="0.35",
        linewidth=1.2,
        label=r"Theoretical Larmor circle ($r_L$)",
    )

    ax.plot(x[0], y[0], marker="o", color="black", markersize=6, label="start")
    ax.plot(
        x[-1],
        y[-1],
        marker="s",
        color="#d62728",
        markersize=6,
        label="end (t = 10 T)",
    )

    ax.set_xlabel(r"$x \,/\, r_L$")
    ax.set_ylabel(r"$y \,/\, r_L$")
    ax.set_title(
        "Single 10 GeV positron in $B = 3\\times10^{-10}$ T "
        "(explicit Euler, 100 steps/gyroperiod)"
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    ax.annotate(
        "outward spiral from numerical\nenergy growth (not physical)",
        xy=(x[-1], y[-1]),
        xytext=(0.55, 0.18),
        textcoords="axes fraction",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "0.4"},
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.85},
    )

    # Zoom on the first gyroperiod, where the orbit is closest to the
    # theoretical circle.

    n_first = STEPS_PER_PERIOD
    axins = ax.inset_axes([0.58, 0.60, 0.40, 0.36])
    axins.plot(x[: n_first + 1], y[: n_first + 1], color="#1f77b4", linewidth=1.3)
    axins.plot(np.sin(theta), -1.0 + np.cos(theta), "--", color="0.35", linewidth=1.0)
    axins.set_title("first gyroperiod", fontsize=8)
    axins.set_aspect("equal", adjustable="datalim")
    axins.grid(True, alpha=0.3)
    axins.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(TRAJECTORY_PATH, dpi=200)
    plt.close(fig)


def plot_energy(diag: dict) -> None:
    sim = diag["sim"]
    t_over_t = sim["time"] / T_GYRO
    energy_gev = sim["energy"] / (1.0e9 * E_CHARGE)

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.plot(t_over_t, energy_gev, color="#d62728", linewidth=1.6, label="Euler total energy")
    ax.axhline(
        E0_GEV,
        linestyle="--",
        color="0.35",
        linewidth=1.2,
        label=f"initial energy = {E0_GEV:.0f} GeV",
    )

    ax.set_xlabel("time [theoretical gyroperiods]")
    ax.set_ylabel("total energy [GeV]")
    ax.set_title("Total energy versus time (no renormalisation, no energy correction)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    ax.annotate(
        "max relative drift = +{:.1f}%\n(final E = {:.2f} GeV)".format(
            100.0 * diag["max_abs_rel_energy_drift"], diag["final_energy_gev"]
        ),
        xy=(t_over_t[-1], energy_gev[-1]),
        xytext=(0.47, 0.24),
        textcoords="axes fraction",
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": "0.4"},
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "alpha": 0.9},
    )

    fig.tight_layout()
    fig.savefig(ENERGY_PATH, dpi=200)
    plt.close(fig)


# Reporting

def print_summary(diag: dict) -> None:
    print("=" * 72)
    print("BASELINE SIMULATION: single relativistic positron in uniform B")
    print("=" * 72)
    print("Physical constants")
    print(f"  e                 = {E_CHARGE:.15e} C")
    print(f"  m_e               = {M_E:.15e} kg")
    print(f"  c                 = {C:.1f} m/s")
    print(f"  m_e c^2           = {M_E * C**2:.15e} J")
    print()
    print("Initial conditions")
    print(f"  B0                = {B0:.3e} T")
    print(f"  E0                = {E0_GEV:.6f} GeV = {E0:.15e} J")
    print(f"  |p0|              = {diag['p0']:.15e} kg m/s")
    print(f"  |v0|/c            = {diag['v0_over_c']:.15e}")
    print(f"  gamma0            = {diag['gamma0']:.6f}")
    print(f"  p0 direction      = +x, B direction = +z (perpendicular)")
    print()
    print("Theory (independent of the numerical run)")
    print(f"  r_L = p0/(eB)     = {diag['r_larmor']:.12e} m")
    print(f"      (in AU)       = {diag['r_larmor'] / 1.495978707e11:.6f} AU")
    print(f"  omega             = {diag['omega']:.12e} rad/s")
    print(f"  T = 2*pi*gamma*m/eB = {diag['gyroperiod']:.12e} s")
    print(f"      (in minutes)  = {diag['gyroperiod'] / 60.0:.6f} min")
    print("  consistency: 2*pi*r_L/|v0| = "
          f"{2.0 * np.pi * diag['r_larmor'] / diag['v0']:.12e} s")
    print(f"  consistency: |v0|/omega    = {diag['v0'] / diag['omega']:.12e} m")
    print()
    print("Numerical setup")
    print(f"  steps per period  = {diag['steps_per_period']}")
    print(f"  periods           = {N_PERIODS}")
    print(f"  dt                = {diag['dt']:.12e} s")
    print(f"  total time        = {diag['t_end']:.12e} s")
    print(f"  total steps       = {diag['n_steps']}")
    print()
    print("Results")
    print(f"  max |dE|/E0       = {diag['max_abs_rel_energy_drift']:.12e} "
          f"({100.0 * diag['max_abs_rel_energy_drift']:.4f}%)")
    print(f"  final dE/E0       = {diag['final_rel_energy_drift']:.12e} "
          f"({100.0 * diag['final_rel_energy_drift']:.4f}%)")
    print(f"  final E           = {diag['final_energy_gev']:.9f} GeV")
    print(f"  energy monotonic  = {diag['energy_monotonic']}")
    print(f"  max |z|           = {diag['max_abs_z']:.3e} m")
    print(f"  radial min / max  = {diag['radial_min']:.6e} / "
          f"{diag['radial_max']:.6e} m")
    print(f"  radial max / r_L  = {diag['radial_max_over_r']:.6f}")
    print()
    print("Radius estimates (dimensionless, in units of theoretical r_L)")
    fit_first = diag["fit_first"]
    print("  first-period circle fit:")
    print(f"    R                 = {diag['radius_first_fit'] / R_L:.9f} r_L")
    print(f"    relative diff     = {diag['radius_first_fit_rel']:+.9e} "
          f"({100.0 * diag['radius_first_fit_rel']:+.4f}%)")
    print(f"    center            = ({fit_first['cx'] / R_L:+.6f}, "
          f"{fit_first['cy'] / R_L:+.6f}) r_L")
    print(f"    RMS residual      = {fit_first['rms_residual']:.6e} m "
          f"({fit_first['rms_residual'] / R_L:.4%} of r_L)")
    print("  local osculating radius from first three positions:")
    print(f"    R(t=0+)           = {diag['radius_local_initial'] / R_L:.9f} r_L "
          f"({100.0 * diag['radius_local_initial_rel']:+.4f}%)")
    print("  local osculating radius, median over first period:")
    print(f"    R_med             = {diag['radius_local_median_first'] / R_L:.9f} r_L "
          f"({100.0 * diag['radius_local_median_first_rel']:+.4f}%)")
    print("  per-period circle fits (spiral growth):")
    for k, fit in enumerate(diag["period_fits"], start=1):
        rel = (fit["radius"] - R_L) / R_L
        print(f"    period {k:2d}: R = {fit['radius'] / R_L:.6f} r_L  "
              f"({100.0 * rel:+.3f}%)  center_y = {fit['cy'] / R_L:+.4f} r_L")
    print(f"  whole-run circle fit: R = {diag['radius_whole_fit'] / R_L:.6f} r_L "
          f"({100.0 * diag['radius_whole_fit_rel']:+.2f}%), "
          f"RMS residual = {diag['whole_fit_rms'] / R_L:.4f} r_L")
    print()
    print("Interpretation: the explicit Euler scheme is not energy conserving.")
    print("The orbit spirals outward, so no single radius describes the whole run.")
    print("The first-period fit is already biased high by the growth within that")
    print("period and must not be quoted as an accurate radius measurement.")
    print("=" * 72)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    sim = simulate()
    diag = compute_diagnostics(sim)
    write_csv(diag)
    plot_trajectory(diag)
    plot_energy(diag)
    print_summary(diag)

    print()
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {TRAJECTORY_PATH}")
    print(f"Wrote {ENERGY_PATH}")


if __name__ == "__main__":
    main()
