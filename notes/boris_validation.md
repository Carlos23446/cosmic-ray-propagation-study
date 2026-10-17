# Relativistic Boris Pusher Validation

## 1. Purpose and scope

This is a new validation stage for the uniform-magnetic-field baseline. It
introduces a **relativistic Boris pusher** and compares it with the exact
analytical orbit and with the existing explicit-Euler results. The physical
model is unchanged:

- one positron, charge $q=+e$;
- uniform, static magnetic field $\mathbf{B}=(0,0,B_0)$,
  $B_0=3\times10^{-10}$ T;
- zero electric field;
- initial position $\mathbf{r}_0=(0,0,0)$;
- initial momentum along $+x$;
- initial total energy $E_0=10$ GeV.

No turbulence, scattering, energy loss, Monte Carlo sampling, or additional
physics is included. The existing baseline and convergence scripts, data,
figures and notes were not modified. The new implementation is
`code/boris_validation.py`.

## 2. Exact uniform-field solution

The exact relativistic gyroangular frequency, Larmor radius and gyroperiod are

$$
\omega=\frac{|q|B_0}{\gamma_0m_e}
      =\frac{|q|B_0c^2}{E_0}
      =2.696265536210\times10^{-3}\ \mathrm{rad/s},
$$

$$
r_L=\frac{p_0}{|q|B_0}
   =1.111880315876\times10^{11}\ \mathrm{m},
\qquad
T=\frac{2\pi}{\omega}
  =2330.328828076\ \mathrm{s}.
$$

For the initial conditions used here, the exact position and momentum are

$$
x(t)=r_L\sin(\omega t),\qquad
y(t)=-r_L\left[1-\cos(\omega t)\right],\qquad
z(t)=0,
$$

$$
p_x(t)=p_0\cos(\omega t),\qquad
p_y(t)=-p_0\sin(\omega t),\qquad
p_z(t)=0,
$$

with $p_0=5.344285985701\times10^{-18}$ kg m/s and
$|\mathbf{v}_0|/c=0.999999998694$. The exact orbit is a circle of radius
$r_L$ centred at $(0,-r_L)$.

## 3. Relativistic Boris update

At integer time level $n$, with $\mathbf{p}_n$ and
$E_n=\sqrt{|\mathbf{p}_n|^2c^2+m_e^2c^4}$:

$$
\gamma_n=\frac{E_n}{m_ec^2},
\qquad
\mathbf{t}=\frac{q\,\mathbf{B}\,\Delta t}{2m_e\gamma_n}.
$$

Because $\mathbf{E}=0$, there are no electric half-kicks. The magnetic rotation
is

$$
\mathbf{p}'=\mathbf{p}_n+\mathbf{p}_n\times\mathbf{t},
\qquad
\mathbf{s}=\frac{2\mathbf{t}}{1+|\mathbf{t}|^2},
\qquad
\mathbf{p}_{n+1}=\mathbf{p}_n+\mathbf{p}'\times\mathbf{s}.
$$

The velocity is always obtained from the relativistic momentum–velocity
relation

$$
\mathbf{v}_n=\frac{\mathbf{p}_n c^2}{E_n},
$$

and the position is advanced with the time-centered trapezoidal drift

$$
\mathbf{r}_{n+1}
=\mathbf{r}_n+\frac{\Delta t}{2}\left(\mathbf{v}_n+\mathbf{v}_{n+1}\right).
$$

**Implementation choice.** This is the synchronous form of the Boris pusher:
momenta and positions are stored at integer times. The canonical PIC leapfrog
stores velocities at half-integer times and advances the position with
$\mathbf{v}_{n+1/2}$; the synchronous trapezoidal drift used here is a
second-order variant that allows direct comparison with the Euler baseline at
the same integer times. The Boris rotation itself is unchanged. The $\gamma$
factor in $\mathbf{t}$ is evaluated from the evolved momentum at every step;
this is a fully relativistic rotation, not a non-relativistic one. Momentum is
never renormalised.

### 3.1 Energy invariant

The Boris rotation is a rotation of $\mathbf{p}$ in the plane perpendicular to
$\mathbf{B}$, so it preserves $|\mathbf{p}|$ exactly in exact arithmetic.
Therefore $E=\sqrt{|\mathbf{p}|^2c^2+m_e^2c^4}$ is also preserved. The
measured drift is only the accumulation of floating-point round-off.

### 3.2 Why the orbit radius is also preserved

For a circular orbit the displacement over a rotation angle $\theta$ has chord
length $2r_L\sin(\theta/2)$. The trapezoidal drift gives

$$
\frac{\Delta t}{2}\left(\mathbf{v}_n+\mathbf{v}_{n+1}\right)
=\Delta t\,v_0\cos(\theta/2)\,\hat{\mathbf{e}}_\perp
=r_L\omega\Delta t\cos(\theta/2)\,\hat{\mathbf{e}}_\perp.
$$

With the Boris rotation angle

$$
\theta=2\arctan\left(\frac{\omega\Delta t}{2}\right),
\qquad
\Delta t=\frac{2\tan(\theta/2)}{\omega},
$$

the trapezoidal displacement equals

$$
2r_L\sin(\theta/2),
$$

which is exactly the chord displacement on the circle. Thus the discrete map
rotates both position and momentum by $\theta$ around the gyrocentre: the
orbit stays on the exact circle of radius $r_L$, but it advances by the wrong
angle per step.

### 3.3 Phase error

The exact rotation per step is $\omega\Delta t$, while Boris uses
$\theta=2\arctan(\omega\Delta t/2)$. The per-step phase lag is

$$
\delta=\omega\Delta t-2\arctan\left(\frac{\omega\Delta t}{2}\right)
\approx\frac{(\omega\Delta t)^3}{12}.
$$

Over $N=10f$ steps, with $\omega\Delta t=2\pi/f$,

$$
\Delta\varphi_N
\approx N\frac{(\omega\Delta t)^3}{12}
=\frac{20\pi^3}{3f^2}\ \mathrm{rad}
=\frac{11843.5}{f^2}\ \mathrm{deg}.
$$

This is an $O(\Delta t^2)$ accumulated phase error. It is **not** an energy
error: the particle is on the right circle, but at the wrong phase. For a pure
phase lag at $t=10T$ (where the exact position is the origin), the final radial
distance is

$$
|\mathbf{r}(10T)|=2r_L\sin\left(\frac{\Delta\varphi_N}{2}\right).
$$

## 4. Numerical results

All four runs cover exactly 10 theoretical gyroperiods:
$t_{\rm end}=10T=23303.28828076$ s. The time step is
$\Delta t=T/f$ and $N=10f$.

### 4.1 Required metrics

| $f$ | $\Delta t$ (s) | $N$ | Max $\lvert\Delta E/E_0\rvert$ | Max $\lvert\Delta\lvert p\rvert/p_0\rvert$ | Final $p/p_0$ | Max radius deviation | Max phase error | Max position error | Max $\lvert z\rvert$ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 46.60657656 | 500 | $6.661\times10^{-16}$ | $6.661\times10^{-16}$ | 1.000000000000 | $2.887\times10^{-15}$ | 4.726220° | $8.246\times10^{-2}$ | 0 |
| 100 | 23.30328828 | 1000 | $2.220\times10^{-15}$ | $2.109\times10^{-15}$ | 1.000000000000 | $3.664\times10^{-15}$ | 1.183652° | $2.066\times10^{-2}$ | 0 |
| 200 | 11.65164414 | 2000 | $1.554\times10^{-15}$ | $1.332\times10^{-15}$ | 1.000000000000 | $2.665\times10^{-15}$ | 0.296044° | $5.167\times10^{-3}$ | 0 |
| 400 | 5.825822070 | 4000 | $1.776\times10^{-15}$ | $1.776\times10^{-15}$ | 1.000000000000 | $6.439\times10^{-15}$ | 0.074019° | $1.292\times10^{-3}$ | 0 |

The maximum energy drift over all runs is
$2.220\times10^{-15}$; the maximum momentum-magnitude drift is
$2.109\times10^{-15}$; the maximum radius deviation is
$6.439\times10^{-15}$. These are a few to a few tens of double-precision
machine epsilons and are consistent with round-off accumulation, not with a
systematic energy error. The maximum **per-step** relative momentum change was
$2.883\times10^{-16}$.

### 4.2 Radius and trajectory diagnostics

| $f$ | Fit $R/r_L$ | Fit radius error | Fit RMS$/r_L$ | Fit centre | Final $\lvert r\rvert/r_L$ | Final position error$/r_L$ |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 1.000000000000 | $+1.11\times10^{-15}$ | $1.33\times10^{-15}$ | $(-6.6\times10^{-17},-1.000000000000)$ | 0.082464718 | 0.082464718 |
| 100 | 1.000000000000 | $-2.22\times10^{-16}$ | $1.09\times10^{-15}$ | $(-4.5\times10^{-16},-1.000000000000)$ | 0.020658252 | 0.020658252 |
| 200 | 1.000000000000 | $0.00$ | $8.87\times10^{-16}$ | $(-3.2\times10^{-16},-1.000000000000)$ | 0.005166942 | 0.005166942 |
| 400 | 1.000000000000 | $-2.78\times10^{-15}$ | $4.47\times10^{-15}$ | $(+3.8\times10^{-16},-1.000000000000)$ | 0.001291880 | 0.001291880 |

The fitted circle is the exact Larmor circle to the precision shown. The
maximum deviation of the trajectory from the exact circle
$(0,-r_L)$ is $2.887\times10^{-15}$, $3.664\times10^{-15}$,
$2.665\times10^{-15}$ and $6.439\times10^{-15}$ for
$f=50,100,200,400$. The final radial distance is therefore entirely a phase
effect, not a radius error.

### 4.3 Phase comparison

| $f$ | Measured phase error | Exact discrete prediction | Leading-order $11843.5/f^2$ | Leading-order relative difference | Predicted final radial | Final radial relative difference |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 4.726220132° | 4.726220132° | 4.737410113° | $-2.362\times10^{-3}$ | 0.082464718 | $5.22\times10^{-14}$ |
| 100 | 1.183651677° | 1.183651677° | 1.184352528° | $-5.918\times10^{-4}$ | 0.020658252 | $-2.90\times10^{-12}$ |
| 200 | 0.296044306° | 0.296044306° | 0.296088132° | $-1.480\times10^{-4}$ | 0.005166942 | $1.88\times10^{-12}$ |
| 400 | 0.074019294° | 0.074019293° | 0.074022033° | $-3.701\times10^{-5}$ | 0.001291880 | $3.58\times10^{-11}$ |

The exact discrete prediction
$N[\omega\Delta t-2\arctan(\omega\Delta t/2)]$ agrees with the measured phase
error at the $10^{-13}$–$10^{-11}$ relative level. This is a consistency check
of the phase accumulation. The **independent leading-order** formula
$20\pi^3/(3f^2)$ agrees to 0.24% at 50 steps/period and improves to 0.0037% at
400 steps/period, as expected for an asymptotic $O(\Delta t^2)$ law.

The observed phase orders when the step count doubles are **1.997, 1.999 and
2.000**, confirming second-order phase convergence. Momentum phase and
position phase agree to better than $10^{-9}$ degrees in every run.

The final radial distance and the final position error are identical because
the exact position at $t=10T$ is the origin. They confirm the pure phase-lag
interpretation: at $f=100$, the orbit radius is exact to $<10^{-14}r_L$, yet
the particle is displaced from the origin by $2.07\%\,r_L$ after 10 periods.

## 5. Validation checks

| Check | Result | Status |
|---|---|---|
| Energy drift $<10^{-12}$ (all runs) | max $2.220\times10^{-15}$ | **PASS** |
| Momentum-magnitude drift $<10^{-12}$ (all runs) | max $2.109\times10^{-15}$ | **PASS** |
| Radius deviation $<10^{-12}$ (all runs) | max $6.439\times10^{-15}$ | **PASS** |
| $\max\lvert z\rvert=0$ (all runs) | exactly 0 | **PASS** |
| All metrics finite; no NaN/Inf | yes | **PASS** |
| Per-step momentum invariance | max relative change $2.883\times10^{-16}$ | **PASS** |
| Phase error decreases with step count | orders 1.997, 1.999, 2.000 | **PASS** |
| Momentum and position phase agree | $<10^{-9}$ deg | **PASS** |
| Leading-order phase law | 0.24% at $f=50$, improving with $f$ | **PASS** |
| Predicted final radial distance | relative difference $\le3.6\times10^{-11}$ | **PASS** |
| Euler comparison against existing CSV | max difference $\le4.9\times10^{-13}$ | **PASS** |
| Phase error below 0.1° per 10 periods | only $f=400$ (0.0740°); $f=50,100,200$ fail | **resolution limitation** |
| Phase error below 1° per 10 periods | $f=200,400$ pass; $f=50,100$ fail | **resolution limitation** |

No invariant check failed. The only "failures" are phase-accuracy targets at
coarse resolution: the implementation is behaving as designed, but 50–200
steps per period are not sufficient for 0.1° phase accuracy over 10 periods.
This is a resolution limitation, not evidence of an implementation bug.
Energy/radius conservation at round-off does **not** imply phase accuracy.

## 6. Comparison with explicit Euler

The Euler columns use the existing explicit-Euler integrator
(`baseline_convergence.simulate`) for the same four step counts. Existing
`data/baseline_convergence.csv` values were cross-checked; the maximum
differences are $\le4.9\times10^{-13}$.

| $f$ | Boris max energy drift | Euler max energy drift | Boris max radius deviation | Euler max radius deviation | Boris phase error | Euler accumulated phase error |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | $6.661\times10^{-16}$ | 1.982563 | $2.887\times10^{-15}$ | 1.982563 | 4.726220° | 1792.916° |
| 100 | $2.220\times10^{-15}$ | 1.224374 | $3.664\times10^{-15}$ | 1.224374 | 1.183652° | 1367.339° |
| 200 | $1.554\times10^{-15}$ | 0.724506 | $2.665\times10^{-15}$ | 0.724506 | 0.296044° | 957.445° |
| 400 | $1.776\times10^{-15}$ | 0.409596 | $6.439\times10^{-15}$ | 0.409596 | 0.074019° | 611.991° |

Euler's accumulated phase errors are multiple complete revolutions
(1792.9° ≈ 5.0 turns at $f=50$, down to 612.0° ≈ 1.7 turns at $f=400$).
Its energy and orbit radius are wrong by +41% to +198% across these four
resolutions. Boris reduces the energy/radius error by 14–15 orders of
magnitude and the phase error by roughly 2.5–4 orders of magnitude, while
still leaving an $O(\Delta t^2)$ phase lag.

## 7. Practical implications

The Boris pusher is a clear improvement for energy and orbit-radius
conservation and should replace explicit Euler for the production
propagation study. The measured invariants support round-off-level
conservation (at the few-$10^{-15}$ level) in this uniform field.

Phase accuracy is a separate requirement. At fixed $\Delta t$ the phase error
grows linearly with simulated time because it accumulates per step:

- $f=100$: 1.1837° per 10 periods, so about **118° per 1000 periods**;
- $f=400$: 0.0740° per 10 periods, so about **7.4° per 1000 periods**.

Keeping the phase error below 0.1° over 1000 periods would require roughly
$f\approx3.4\times10^3$ steps per period with this half-angle rotation, or an
exact-tangent rotation. The exact-tangent variant sets
$\mathbf{t}=\tan(\omega\Delta t/2)\hat{\mathbf{B}}$, which makes the uniform-B
rotation angle exactly $\omega\Delta t$ and removes this particular phase lag
while retaining the norm-preserving Boris rotation. That variant should be
evaluated next if phase coherence over long durations is required. For a
turbulent field, the standard Boris rotation is still the natural workhorse,
but phase-error convergence must be checked separately from energy
conservation.

## 8. Failures and limitations

1. **No invariant failures.** Energy, momentum magnitude, radius and plane
   motion all pass at the stated tolerances.
2. **Phase target failures at coarse resolution.** Only 400 steps/period meets
   a 0.1° phase target over 10 periods; 50–200 steps/period do not. This is the
   expected $O(\Delta t^2)$ half-angle error, not a coding error.
3. **Synchronous drift variant.** The position update is trapezoidal at
   integer time levels. The canonical PIC leapfrog uses half-step velocities;
   the two are both second-order but not identical. The choice is documented
   here and in the code.
4. **Idealised field.** Uniform, static $\mathbf{B}$, zero $\mathbf{E}$, no
   turbulence, no losses, no scattering. The exact-circle radius invariant is
   specific to this uniform-field case.
5. **Only four step counts.** The observed phase order is already 2.00, but no
   Richardson extrapolation or longer-duration run was performed.
6. **No runtime or memory measurement.** The long-duration cost estimates in
   Section 7 are based on the phase-error scaling only.
7. **Phase unwrapping.** Euler's accumulated phase error exceeds $2\pi$; the
   reported Euler values are unwrapped multi-turn errors, not a single-orbit
   phase angle. This is stated explicitly rather than folding them into
   $[-\pi,\pi]$.
8. **Round-off claim is bounded.** The largest measured energy/momentum drift
   is $2.22\times10^{-15}$ and the largest radius deviation is
   $6.44\times10^{-15}$; the claim of round-off-level conservation is limited
   to these measured values, over 10 gyroperiods and these four resolutions.

## 9. Reproduction and output paths

Environment: Python 3.9.6, NumPy 1.26.4, Matplotlib 3.9.4 (macOS).

```bash
python3 code/boris_validation.py
```

New outputs:

- `code/boris_validation.py` — relativistic Boris implementation and validation
- `data/boris_validation.csv` — one row per step count, 34 columns
- `figures/boris_validation.png` — trajectory, energy, phase and radius panels
- `notes/boris_validation.md` — this report

The existing baseline and convergence files were preserved. No commit or push
was performed.
