# Baseline Simulation Validation

## 1. Purpose and scope

This note documents the first baseline simulation for the cosmic-ray positron
propagation project. It follows **one** relativistic positron in a uniform,
static magnetic field using the explicit Euler method. There is no electric
field, no turbulence, no scattering, no Monte Carlo sampling, no energy loss,
and no additional physical effect. The purpose is to establish the numerical
baseline and to characterise the integration error before any turbulent field
is introduced.

The source is `code/baseline_simulation.py`; the numerical output is
`data/baseline_simulation.csv`; the figures are
`figures/baseline_trajectory.png` and `figures/baseline_energy.png`.

## 2. Physical model and constants

SI units are used throughout.

| Constant | Symbol | Value |
|---|---|---|
| Elementary charge | $e$ | $1.602176634\times10^{-19}$ C |
| Positron mass | $m_e$ | $9.1093837015\times10^{-31}$ kg |
| Speed of light | $c$ | $2.99792458\times10^{8}$ m/s |
| Rest energy | $m_e c^2$ | $8.187105776824\times10^{-14}$ J ($0.51099895$ MeV) |

Model parameters:

| Parameter | Value |
|---|---|
| Charge | $q=+e$ |
| Magnetic field | $\mathbf{B}=(0,0,B_0)$, $B_0=3\times10^{-10}$ T ($0.3$ nT) |
| Electric field | $\mathbf{E}=(0,0,0)$ |
| Initial position | $\mathbf{r}_0=(0,0,0)$ |
| Initial total energy | $E_0=10$ GeV $=1.602176634\times10^{-9}$ J |
| Initial momentum direction | $+x$ (perpendicular to $\mathbf{B}$) |

The initial momentum magnitude is obtained from

$$
E_0^2 = p_0^2 c^2 + m_e^2 c^4
\quad\Longrightarrow\quad
p_0=\frac{\sqrt{E_0^2-(m_e c^2)^2}}{c}
     =5.344285985701\times10^{-18}\ \mathrm{kg\,m/s}.
$$

The initial velocity is

$$
\mathbf{v}=\frac{\mathbf{p}\,c^2}{E_{\mathrm{total}}},
\qquad
\frac{|\mathbf{v}_0|}{c}=0.999999998694,
\qquad
\gamma_0=\frac{E_0}{m_e c^2}=19569.511836.
$$

## 3. Equations of motion

The relativistic Lorentz-force equation with zero electric field is

$$
\frac{d\mathbf{p}}{dt}=q\,(\mathbf{v}\times\mathbf{B}),
\qquad
\frac{d\mathbf{r}}{dt}=\mathbf{v},
$$

with the velocity evaluated from the **current** momentum at every step,

$$
\mathbf{v}=\frac{\mathbf{p}\,c^2}{E_{\mathrm{total}}},
\qquad
E_{\mathrm{total}}=\sqrt{|\mathbf{p}|^2c^2+m_e^2c^4}.
$$

For $\mathbf{B}=(0,0,B_0)$,

$$
\mathbf{v}\times\mathbf{B}
= (v_yB_0,\,-v_xB_0,\,0),
$$

so the motion remains in the $x$–$y$ plane.

### Explicit Euler update

At step $n$, with $\mathbf{p}_n$ and $\mathbf{v}_n$ known:

$$
\mathbf{p}_{n+1}=\mathbf{p}_n+\Delta t\,q\,(\mathbf{v}_n\times\mathbf{B}),
\qquad
\mathbf{r}_{n+1}=\mathbf{r}_n+\Delta t\,\mathbf{v}_n.
$$

No momentum renormalisation and no energy-conservation correction are applied
at any step. The stored energy at each step is recomputed from the evolved
momentum, so the reported drift is a genuine output of the scheme.

## 4. Theoretical Larmor radius and gyroperiod

These are calculated directly from the initial conditions and field, not from
the simulation:

$$
r_L=\frac{p_{0,\perp}}{|q|B_0}
   =\frac{p_0}{eB_0}
   =1.111880315876\times10^{11}\ \mathrm{m}
   \;(0.743246\ \mathrm{AU}),
$$

$$
T=\frac{2\pi\gamma_0 m_e}{|q|B_0}
  =\frac{2\pi E_0}{|q|B_0c^2}
  =2.330328828076\times10^{3}\ \mathrm{s}
  =38.838814\ \mathrm{min},
$$

$$
\omega=\frac{|q|B_0}{\gamma_0m_e}
       =2.696265536210\times10^{-3}\ \mathrm{rad/s}.
$$

Algebraic consistency checks (not independent physical measurements):
$2\pi r_L/|\mathbf{v}_0|=2.330328828076\times10^3$ s and
$|\mathbf{v}_0|/\omega=1.111880315876\times10^{11}$ m, both equal to the
values above.

## 5. Numerical setup

| Quantity | Value |
|---|---|
| Steps per theoretical gyroperiod | 100 |
| Number of gyroperiods | 10 |
| Time step | $\Delta t=T/100=23.30328828076$ s |
| Total simulated time | $2.330328828076\times10^4$ s ($6.4731$ h) |
| Total steps | 1000 |
| Stored rows | 1001 (including $t=0$) |

The CSV stores time (s and gyroperiods), position components (m), momentum
components (kg m/s), momentum magnitude, speed (m/s), total energy (J and
GeV), and radial distance $\sqrt{x^2+y^2}$ (m).

## 6. Results

### 6.1 Total energy drift

The energy is **not** conserved by explicit Euler. It increases monotonically
at every step.

| Quantity | Value |
|---|---|
| Maximum relative drift $(E-E_0)/E_0$ | $+1.224374464036$ |
| Maximum relative drift (%) | $+122.4374\%$ |
| Final relative drift | $+122.4374\%$ |
| Final total energy | $22.243744640$ GeV |
| Energy monotonic? | Yes (every step increases) |

Energy at selected times:

| $t/T$ | $E$ (GeV) | $(E-E_0)/E_0$ | $p/p_0$ |
|---:|---:|---:|---:|
| 0 | 10.000000 | $0.0000\%$ | 1.000000 |
| 1 | 11.810098 | $+18.1010\%$ | 1.181010 |
| 2 | 13.377475 | $+33.7747\%$ | 1.337747 |
| 5 | 17.245060 | $+72.4506\%$ | 1.724506 |
| 10 | 22.243745 | $+122.4374\%$ | 2.224374 |

### 6.2 Numerical trajectory radius

The trajectory is **visibly spiraling outward**, so no single radius
characterises the whole run. Three estimates are reported, each with an
explicit definition:

1. **First-gyroperiod algebraic circle fit (headline estimate).**
   A Kåsa least-squares circle is fitted to the 101 stored positions from
   $t=0$ to $t=T$:
   - fitted radius $R_{\mathrm{fit}}=1.092609662\,r_L$;
   - relative difference from theory: $+9.260966\times10^{-2}$
     ($+9.2610\%$);
   - fitted centre: $(-0.060151,\,-1.011800)\,r_L$ (theory: $(0,-1)\,r_L$);
   - RMS radial residual: $3.047351\times10^{9}$ m, i.e. $2.7407\%$ of $r_L$.

   This fit is **not an accurate radius measurement**. Even within the first
   period the orbit grows, so a circle is already a poor model; the fit absorbs
   part of that growth and is biased high.

2. **Local osculating radius from the first three positions.**
   The circumradius of $\mathbf{r}_0,\mathbf{r}_1,\mathbf{r}_2$ gives
   $R(t=0^+)=1.001478861\,r_L$, i.e. $+0.1479\%$ from theory. This is
   consistent with the initial curvature of the discrete orbit, but it is an
   instantaneous value at $t\approx0$ only.

3. **Local osculating radius, median over the first period.**
   The median circumradius over all consecutive triples inside the first
   period is $1.093802685\,r_L$ ($+9.3803\%$).

Per-period circle-fit radii show the growth directly:

| Period | Fitted radius $(r_L)$ | Relative difference | Fitted centre $y$ $(r_L)$ |
|---:|---:|---:|---:|
| 1 | 1.092610 | $+9.261\%$ | $-1.0118$ |
| 2 | 1.260545 | $+26.054\%$ | $-1.0502$ |
| 3 | 1.408560 | $+40.856\%$ | $-1.0208$ |
| 4 | 1.542420 | $+54.242\%$ | $-0.9531$ |
| 5 | 1.665543 | $+66.554\%$ | $-1.0325$ |
| 6 | 1.780162 | $+78.016\%$ | $-0.9869$ |
| 7 | 1.887826 | $+88.783\%$ | $-1.0011$ |
| 8 | 1.989668 | $+98.967\%$ | $-1.0030$ |
| 9 | 2.086541 | $+108.654\%$ | $-0.9994$ |
| 10 | 2.179109 | $+117.911\%$ | $-0.9931$ |

A single circle fitted to all 1001 points gives $R=1.723107\,r_L$
($+72.31\%$) with an RMS residual of $0.3454\,r_L$. That fit is meaningless as
a radius: it is simply an average over a spiral.

### 6.3 Out-of-plane displacement

$$
\max |z| = 0\ \mathrm{m}
$$

exactly (to floating-point zero). This is expected: the initial momentum and
the field both lie in the $x$–$y$ plane, and $\mathbf{v}\times\mathbf{B}$ has no
$z$ component.

The radial distance from the origin ranges from $0$ m at $t=0$ to
$3.431856\times10^{11}$ m $=3.086534\,r_L$. This maximum is a distance from the
origin, **not** an orbital radius: the theoretical circle is centred at
$(0,-r_L)$, so a point on it can be up to $2r_L$ from the origin. The radial
distance therefore oscillates with orbital phase while the spiral grows.

### 6.4 Independent checks on the stored CSV

The CSV was re-read and checked without relying on the simulation's own
diagnostics:

| Check | Result |
|---|---|
| Rows | 1001 data rows, 13 columns |
| $E$ recomputed from stored $p$ vs stored $E$ | max relative difference $1.24\times10^{-12}$ |
| $\|\mathbf{p}\|$ recomputed from components | max absolute difference $8.8\times10^{-30}$ kg m/s |
| $v=pc^2/E$ recomputed vs stored speed | max relative difference $7.5\times10^{-13}$ |
| $z$ column | identically zero |
| Radial distance vs $\sqrt{x^2+y^2}$ | max difference $0.10$ m (12-digit CSV rounding only) |
| Time spacing and final time | uniform, $t_{\max}=10.000000\,T$ |
| Energy monotonic | Yes |

## 7. Validation checks required by the task

| # | Requirement | Result | Assessment |
|---:|---|---|---|
| 1 | Theoretical Larmor radius and gyroperiod | $r_L=1.111880315876\times10^{11}$ m; $T=2330.328828$ s | Computed independently from constants |
| 2 | Maximum relative energy drift | $+122.4374\%$ (final $22.2437$ GeV) | Large, monotonic growth; expected for explicit Euler |
| 3 | Numerical radius vs theory | First-period fit $1.09261\,r_L$ ($+9.261\%$); initial local curvature $1.00148\,r_L$ ($+0.148\%$) | Fit is biased by the spiral; **not** an accurate radius |
| 4 | Maximum absolute $z$ displacement | $0$ m | Exact plane motion, as expected |
| 5 | Match to explicit-Euler expectations | Outward spiral, monotonic energy growth, no renormalisation | Qualitatively and quantitatively consistent |

The large energy drift is **not** a code failure: it is the expected behaviour
of explicit Euler applied to a charged particle in a magnetic field. The
simulation is validated as an explicit-Euler baseline, not as an
energy-conserving integrator.

### Quantitative expectation for the energy growth

At each step the momentum increment

$$
\delta = |q|\,|\mathbf{v}\times\mathbf{B}|\,\Delta t \approx eB_0c\,\Delta t
$$

is perpendicular to $\mathbf{p}$, so to leading order

$$
p_N^2 \approx p_0^2 + N\delta^2
\quad\Longrightarrow\quad
\frac{p_N}{p_0}\approx\sqrt{1+N(\omega_0\Delta t)^2},
\qquad \omega_0\Delta t=\frac{2\pi}{100}.
$$

For $N=1000$,

$$
\sqrt{1+1000\,(2\pi/100)^2}=2.224374465.
$$

The simulation gives $p_{\mathrm{final}}/p_0=2.224374466$ and
$E_{\mathrm{final}}/E_0=2.224374464$, matching the explicit-Euler prediction to
about one part in $10^{9}$–$10^{10}$. This confirms that the observed drift is
the known discretisation error of the method, not an implementation mistake.

## 8. Physical behaviour versus numerical error

- **Physics:** with a static uniform $\mathbf{B}$, zero $\mathbf{E}$, and no
  losses, the exact solution has constant $|\mathbf{p}|$ and $E$, constant
  speed, a closed circle of radius $r_L$ centred at $(0,-r_L)$, constant
  gyroperiod, and $z=0$.
- **Numerics:** explicit Euler adds a perpendicular momentum increment at every
  step. The updated momentum is slightly longer than the exact one and rotated
  slightly behind the exact rotation. Both effects accumulate, so $|\mathbf{p}|$
  and $E$ grow monotonically, the local gyroradius grows, and the orbit spirals
  outward. The speed remains close to $c$ (it increases from
  $0.999999998694c$ to $0.999999999736c$, a change of about
  $1.0\times10^{-9}$ in $v/c$), so the growth is almost entirely in
  momentum/energy and radius rather than in speed.
- The trajectory figure shows the theoretical circle and the simulated spiral
  separately; the spiral is numerical error, not a physical radiation or
  propagation effect. No physical effect in this model can change the energy.

## 9. Limitations

1. **One particle only.** No ensemble, no Monte Carlo sampling, no statistics.
2. **Idealised field.** Uniform, static $\mathbf{B}$; no turbulence, no
   spatial variation, no time dependence, no electric field.
3. **No physical processes.** No energy losses, no radiation reaction, no
   scattering, no diffusion, no adiabatic or synchrotron terms.
4. **First-order, non-symplectic integrator.** Explicit Euler is not
   energy-conserving; the $+122\%$ energy drift over 10 periods makes it
   unsuitable for long integrations. A symplectic or higher-order scheme (or
   much smaller $\Delta t$) would be required for quantitative long-time
   dynamics.
5. **Fixed step size.** $\Delta t=T/100$ is fixed from the initial gyroperiod;
   no convergence study in $\Delta t$ was performed. As the numerical energy
   grows, the instantaneous gyroperiod also grows, so the effective number of
   steps per instantaneous gyroperiod increases during the run.
6. **Radius estimate is ambiguous by construction.** Because the orbit
   spirals, a single "numerical radius" is not well defined. The first-period
   fit is biased high ($+9.26\%$), and even it has a $2.74\%\,r_L$ RMS residual.
   The per-period values in Section 6.2 are descriptive of the spiral, not
   accurate measurements of $r_L$.
7. **No uncertainty quantification.** No parameter sampling, no error bars, no
   comparison of field configurations.
8. **Astrophysical realism.** The parameters give $r_L\approx0.74$ AU and
   $T\approx39$ min, which are baseline test values, not a realistic Galactic
   propagation setup.

## 10. Reproduction

Environment used: Python 3.9.6, NumPy 1.26.4, Matplotlib 3.9.4 (macOS).

```bash
python3 code/baseline_simulation.py
```

The script creates/overwrites only:

- `data/baseline_simulation.csv`
- `figures/baseline_trajectory.png`
- `figures/baseline_energy.png`

and prints the diagnostics above. The first run built the Matplotlib font cache
(informational only); the final run completed with exit code 0 and no numerical
warnings or errors. Nothing was committed or pushed.
