# Explicit-Euler Timestep Convergence Study

## 1. Purpose and scope

This note documents a timestep convergence study of the **existing** explicit
Euler integrator from the baseline simulation. The physical model is unchanged:
one positron with charge $q=+e$ in a uniform, static magnetic field
$\mathbf{B}=(0,0,B_0)$, zero electric field, initial momentum along $+x$, and
initial total energy $E_0=10$ GeV. There is no turbulence, no Monte Carlo
sampling, no energy loss, and no additional physics in any run.

The baseline implementation (`code/baseline_simulation.py`) was **not**
modified. The new script `code/baseline_convergence.py` imports the baseline
constants and helpers, then repeats the same explicit-Euler update with 50,
100, 200, and 400 steps per theoretical gyroperiod. Every run covers exactly
10 theoretical gyroperiods, so the final time is always $t_{\rm end}=10T$.

Momentum is never renormalised and energy conservation is never enforced in
any run.

## 2. Numerical method

For each step count $f\in\{50,100,200,400\}$:

$$
\Delta t=\frac{T}{f},
\qquad
N=10f,
\qquad
t_{\rm end}=N\Delta t=10T.
$$

The update is identical to the baseline:

$$
\mathbf{p}_{n+1}=\mathbf{p}_n+\Delta t\,q\,(\mathbf{v}_n\times\mathbf{B}),
\qquad
\mathbf{r}_{n+1}=\mathbf{r}_n+\Delta t\,\mathbf{v}_n,
$$

with

$$
\mathbf{v}_n=\frac{\mathbf{p}_n c^2}{E_n},
\qquad
E_n=\sqrt{|\mathbf{p}_n|^2c^2+m_e^2c^4}.
$$

The only change between runs is $\Delta t$ (and therefore $N$). The stored
energy is always recomputed from the evolved momentum; no correction is
applied.

The convergence study's 100-step run reproduces the baseline CSV to the
precision of that file:

| Comparison (100 steps vs baseline) | Difference |
|---|---:|
| Maximum relative energy drift | $+6.861\times10^{-14}$ |
| Final $p/p_0$ | $+6.910\times10^{-13}$ |
| Final radial distance / $r_L$ | $+2.882\times10^{-13}$ |
| Maximum $\lvert z\rvert$ | $0$ |

These differences are at the 12-digit CSV rounding level, confirming that the
study uses the same integrator and physical model.

## 3. Analytical expectation for explicit Euler

### 3.1 Per-step growth and phase lag

The momentum increment is perpendicular to the current momentum:

$$
\boldsymbol{\delta}_n=\Delta t\,q\,(\mathbf{v}_n\times\mathbf{B}),
\qquad
|\boldsymbol{\delta}_n|=\Delta t\,qB\,|\mathbf{v}_n|
= \Delta t\,qB\,\frac{|\mathbf{p}_n|c^2}{E_n}.
$$

Since $\boldsymbol{\delta}_n\perp\mathbf{p}_n$,

$$
|\mathbf{p}_{n+1}|^2=|\mathbf{p}_n|^2+|\boldsymbol{\delta}_n|^2
=|\mathbf{p}_n|^2\left[1+(\omega_n\Delta t)^2\right],
\qquad
\omega_n=\frac{qBc^2}{E_n}.
$$

Thus each step increases $|\mathbf{p}|$ by a factor
$\sqrt{1+(\omega_n\Delta t)^2}>1$, while the momentum direction rotates by
$\arctan(\omega_n\Delta t)$, which is slightly **less** than the exact rotation
$\omega_n\Delta t$. The orbit therefore both grows and lags in phase. The
energy grows because $|\mathbf{p}|$ grows; there is no physical energy source.

### 3.2 Leading-order closed form

For an ultra-relativistic positron, $|\mathbf{v}|\approx c$ and
$|\mathbf{p}|\approx E/c$, so $\delta/|\mathbf{p}_0|\approx\omega_0\Delta t$.
Summing the independent perpendicular increments gives

$$
\frac{p_N}{p_0}\approx\sqrt{1+N(\omega_0\Delta t)^2}.
$$

With $N=10f$, $\Delta t=T/f$, and $\omega_0T=2\pi$, this becomes

$$
\boxed{\frac{p_N}{p_0}\approx\sqrt{1+\frac{40\pi^2}{f}}}
\qquad\Longrightarrow\qquad
\frac{E_N}{E_0}-1\approx\sqrt{1+\frac{40\pi^2}{f}}-1.
$$

This is a leading-order prediction; it neglects the small change of
$|\mathbf{v}|$ during the run and the rest-mass term.

### 3.3 Exact discrete scalar recurrence

The magnitude update can be reduced to an exact scalar recurrence. From
$|\mathbf{p}_{n+1}|^2=|\mathbf{p}_n|^2[1+(qBc^2\Delta t)^2/E_n^2]$ and
$E_n^2=|\mathbf{p}_n|^2c^2+m_e^2c^4$:

$$
E_{n+1}^2
= E_n^2+(qBc^2\Delta t)^2
\left(1-\frac{(m_ec^2)^2}{E_n^2}\right).
$$

Iterating this scalar recurrence is an independent check of the vector
cross-product integration: it tests the momentum magnitude without using the
vector rotation. It reproduces the full vector run to machine precision
(Section 4.2).

## 4. Results

### 4.1 Required metrics

All runs cover 10 theoretical gyroperiods. Theoretical values:
$r_L=1.111880315876\times10^{11}$ m,
$T=2330.328828076$ s,
$\omega_0=2.696265536210\times10^{-3}$ rad/s,
$t_{\rm end}=23303.28828076$ s.

| Steps/period $f$ | $\Delta t$ (s) | $N$ | Max $\lvert(E-E_0)/E_0\rvert$ | Drift (%) | Final $p/p_0$ | Final $\lvert\mathbf{r}\rvert/r_L$ | Max $\lvert z\rvert$ (m) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 46.60657656 | 500 | 1.982563245791 | +198.2563% | 2.982563249248 | 1.994014597 | 0 |
| 100 | 23.30328828 | 1000 | 1.224374464036 | +122.4374% | 2.224374466353 | 2.149886676 | 0 |
| 200 | 11.65164414 | 2000 | 0.724505980671 | +72.4506% | 1.724505982166 | 2.414510368 | 0 |
| 400 | 5.825822070 | 4000 | 0.409595842189 | +40.9596% | 1.409595843103 | 1.964326218 | 0 |

Final total energies are 29.825632458, 22.243744640, 17.245059807, and
14.095958422 GeV, respectively. The energy increased monotonically in every
run (no step decreased it).

The raw final radial distances are
$2.217106\times10^{11}$, $2.390417\times10^{11}$,
$2.684647\times10^{11}$, and $2.184096\times10^{11}$ m for
$f=50,100,200,400$.

### 4.2 Comparison with the analytical expectation

| $f$ | Numerical $p_N/p_0$ | Leading-order $\sqrt{1+40\pi^2/f}$ | Relative difference | Exact scalar recurrence | Relative difference |
|---:|---:|---:|---:|---:|---:|
| 50 | 2.982563249248 | 2.982563246751 | $+8.370\times10^{-10}$ | 2.982563249248 | $-4.441\times10^{-16}$ |
| 100 | 2.224374466353 | 2.224374464976 | $+6.194\times10^{-10}$ | 2.224374466353 | $-1.443\times10^{-15}$ |
| 200 | 1.724505982166 | 1.724505981497 | $+3.880\times10^{-10}$ | 1.724505982166 | $+8.882\times10^{-16}$ |
| 400 | 1.409595843103 | 1.409595842825 | $+1.973\times10^{-10}$ | 1.409595843103 | $+4.441\times10^{-16}$ |

The leading-order closed form agrees with the full simulation to
$2\times10^{-10}$–$8\times10^{-10}$. The small residual is consistent with the
neglected rest-mass and $\lvert\mathbf{v}\rvert<c$ corrections. The exact
scalar recurrence agrees to machine precision, which independently validates
the vector implementation.

Energy-drift comparison with
$\sqrt{1+40\pi^2/f}-1$:

| $f$ | Observed max drift | Predicted drift | Relative difference |
|---:|---:|---:|---:|
| 50 | 1.982563245791 | 1.982563246751 | $-4.841\times10^{-10}$ |
| 100 | 1.224374464036 | 1.224374464976 | $-7.673\times10^{-10}$ |
| 200 | 0.724505980671 | 0.724505981497 | $-1.139\times10^{-9}$ |
| 400 | 0.409595842189 | 0.409595842825 | $-1.553\times10^{-9}$ |

### 4.3 Convergence behaviour

When the step count doubles ($\Delta t$ halves), the energy-drift error
decreases. The observed order

$$
p=\log_2\left(\frac{e_f}{e_{2f}}\right)
$$

is approaching 1 (first order), but is not yet asymptotic at these relatively
coarse step counts:

| Doubling | Drift ratio | Observed order | Orbit-size error ratio | Observed order |
|---|---:|---:|---:|---:|
| 50 → 100 | 1.619246 | 0.6953 | 1.623743 | 0.6993 |
| 100 → 200 | 1.689944 | 0.7570 | 1.695416 | 0.7616 |
| 200 → 400 | 1.768831 | 0.8228 | 1.774537 | 0.8274 |

The "orbit-size error" here is
$(\text{final-period fitted radius}/r_L)-1$, which is the dimensionless
orbital-radius error (see Section 4.4). It follows the same first-order trend
as the energy error.

### 4.4 The final radial distance is not a clean error metric

The requested final radial distance
$\lvert\mathbf{r}(t_{\rm end})\rvert/r_L$ is **not monotonic**:

| $f$ | Final $\lvert\mathbf{r}\rvert/r_L$ | Final-period fitted radius / $r_L$ | Final phase (deg) |
|---:|---:|---:|---:|
| 50 | 1.994014597 | 2.914570404 | $+83.555$ |
| 100 | 2.149886676 | 2.179109033 | $+17.463$ |
| 200 | 2.414510368 | 1.695468944 | $-32.760$ |
| 400 | 1.964326218 | 1.391915691 | $-18.335$ |

The final-period fitted orbital radius decreases monotonically with smaller
$\Delta t$ (2.9146 → 2.1791 → 1.6955 → 1.3919 $r_L$), consistent with the
energy/momentum error decreasing. The **distance from the origin** does not,
because at the fixed endpoint $t=10T$ it depends on both the orbit size and the
accumulated phase. The final phase changes sign and magnitude between runs
(+83.6°, +17.5°, −32.8°, −18.3°), so the endpoint lands at different positions
even as the orbit itself shrinks. It is therefore a phase-sensitive endpoint
diagnostic, not an error norm, and it must not be used to claim or reject
convergence. In the limit $\Delta t\to0$ it should tend to zero (the exact
orbit is a closed circle returning to the origin after 10 periods), but the
four coarse runs do not show a monotonic approach.

### 4.5 Out-of-plane displacement

For every run, $\max\lvert z\rvert=0$ exactly. The initial momentum and the
magnetic field lie in the $x$–$y$ plane, and
$\mathbf{v}\times\mathbf{B}$ has no $z$ component, so the plane motion is
preserved at machine precision.

## 5. Validation checks

| Check | Result |
|---|---|
| Each run covers 10 theoretical gyroperiods | $N=10f$, $t_{\rm end}=10T$ for all four runs |
| 100-step run reproduces baseline | Differences $\le6.9\times10^{-13}$ (CSV precision) |
| CSV re-read and independently re-integrated | Final $p/p_0$ and radial distance match to 12 digits |
| Energy monotonic in every run | Yes |
| $\max\lvert z\rvert$ | Zero in all runs |
| Leading-order analytical prediction | Agrees to $2\times10^{-10}$–$8\times10^{-10}$ |
| Exact scalar recurrence | Agrees to $\sim10^{-15}$ |
| Baseline source file modified? | No; imported only |
| Unrelated files modified? | No |

The final figure was also checked programmatically for layout: every
annotation lies inside its axes and inside the figure, no annotation overlaps
another annotation or the legend, and the legend does not cover any data
marker.

## 6. Interpretation: physical behaviour versus numerical error

- The physical solution in a uniform static field with no losses is a closed
  circle of radius $r_L$, constant energy, constant speed, and period $T$.
- Explicit Euler produces a monotonic increase of $\lvert\mathbf{p}\rvert$ and
  $E$ because each update adds a perpendicular momentum increment of fixed
  sign. The orbit spirals outward and the phase lags. This is a numerical
  artefact, not a physical effect.
- Reducing $\Delta t$ reduces the artefact monotonically for energy and
  momentum, with approximately first-order convergence. It does not make the
  method exact.
- The final radial distance is a poor convergence diagnostic because it is
  evaluated at a single, phase-sensitive endpoint.

## 7. Is explicit Euler practical for the long-duration study?

No, not in its current form.

At the finest tested resolution ($f=400$), 10 gyroperiods still produce a
**+40.96%** energy drift and an orbit radius about 39% above $r_L$. The
convergence is only first order. Using the leading-order relation

$$
\text{drift}\approx\sqrt{1+\frac{40\pi^2}{f}}-1,
$$

the approximate step counts needed for a given drift over 10 periods are:

| Target energy drift | Approx. steps/period $f$ | Approx. total steps for 10 periods |
|---:|---:|---:|
| 1% | $1.96\times10^{4}$ | $1.96\times10^{5}$ |
| 0.1% | $1.97\times10^{5}$ | $1.97\times10^{6}$ |

These are rough leading-order estimates, but they show the cost: reaching 1%
over only 10 periods requires about 49 times more steps than the finest run
here, and 0.1% requires about 493 times more. For a long-duration propagation
study, the accumulated drift also grows with the number of simulated periods,
so the requirement becomes more severe still.

A first-order, non-symplectic scheme with secular energy growth is therefore
not a practical foundation for the intended long integrations. The next step
should be to **evaluate a more suitable particle integrator**, not merely to
shrink $\Delta t$ further. The natural candidate is the **Boris pusher**, which
is the standard relativistic charged-particle integrator for magnetic fields:
for a uniform field it reproduces the gyration with constant energy to
round-off and avoids the systematic spiral seen here. A higher-order
symplectic or Runge–Kutta scheme is an alternative, but Boris is the standard
starting point for magnetised transport and would provide a much better
baseline for later turbulence studies. This convergence study does not
implement that change; it only establishes that the change is needed.

## 8. Limitations

1. Only four step counts were tested (50, 100, 200, 400); the observed order
   has not reached its asymptotic value of 1, and no Richardson extrapolation
   was performed.
2. The leading-order formula neglects rest-mass and $\lvert\mathbf{v}\rvert<c$
   corrections; it agrees to $\sim10^{-9}$, while the exact scalar recurrence
   is needed for machine-precision agreement.
3. The final radial distance is phase-sensitive and non-monotonic; it is
   reported as requested but is not a convergence metric.
4. The study measures accuracy only, not runtime or memory cost. The step-count
   estimates in Section 7 are therefore lower bounds on the practical cost.
5. The physical model remains the idealised baseline: one particle, uniform
   static field, no turbulence, no losses, no scattering, no electric field.
6. No uncertainty quantification or parameter sampling is included.

## 9. Reproduction and output paths

Environment: Python 3.9.6, NumPy 1.26.4, Matplotlib 3.9.4 (macOS).

```bash
python3 code/baseline_convergence.py
```

Outputs:

- `code/baseline_convergence.py` — this study's script (new file)
- `data/baseline_convergence.csv` — one row per step count, 20 columns
- `figures/baseline_convergence.png` — two-panel convergence plot

The baseline files were left unchanged. No commit or push was performed.
