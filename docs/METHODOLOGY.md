# Methodology

Full derivations for the three pieces of the benchmark that deserve a
blackboard: the kinematic bicycle update, the Hanley–McNeil variance used in
the power analysis, and the regression-flag criterion.

## 1. Kinematic bicycle model (rear-axle reference point)

**State.** The vehicle state is `(x, y, ψ, v, δ)` — rear-axle position,
heading (yaw), longitudinal speed, and front steer angle.

**Continuous kinematics.** With the rear axle as the reference point, the
rear-axle velocity vector is aligned with the heading by construction, and the
heading rate follows from the front-wheel geometry:

```
ẋ  = v cos ψ
ẏ  = v sin ψ
ψ̇  = (v / L) tan δ        (L = wheelbase)
v̇  = a                    (a = longitudinal acceleration command)
```

The third equation comes from the bicycle geometry: the instantaneous centre
of rotation lies at the intersection of the two wheel axes, at distance
`R = L / tan δ` from the rear axle, so `ψ̇ = v / R = (v / L) tan δ`. The
measured yaw rate reported in trajectories is exactly this quantity,

```
ω = (v / L) tan δ.
```

**Discretization.** Explicit (forward) Euler at fixed step `dt`:

```
x_{k+1} = x_k + v_k cos ψ_k · dt
y_{k+1} = y_k + v_k sin ψ_k · dt
ψ_{k+1} = ψ_k + (v_k / L) tan δ̂_k · dt
v_{k+1} = max(0, v_k + â_k · dt)
δ_{k+1} = δ̂_k
```

where the commanded inputs are saturated before integration:

```
δ̂ = clamp(δ_cmd, −δ_max, +δ_max)
â = min(a_cmd, a_max)   if a_cmd ≥ 0
â = max(a_cmd, −a_dec)  if a_cmd < 0
```

The speed floor `max(0, ·)` prevents reverse motion (out of scope for the
benchmark). Heading is kept continuous (never wrapped) so heading-change
integrals in features and labels are well-defined.

**Why the rear axle.** The pure-pursuit controller's lookahead geometry is
defined from the rear axle, and with the rear-axle reference the rear-axle
velocity is always along the heading — which makes the Euler update exact for
straight driving and keeps the closed-form straight-line test (exact to 1e-9)
meaningful.

## 2. Hanley–McNeil variance of the AUC and the power analysis

**AUC as a probability.** With positive-class scores `X ~ F` and
negative-class scores `Y ~ G`,

```
AUC = P(X > Y) + ½ P(X = Y).
```

**Variance.** The empirical AUC is the two-sample Mann–Whitney U statistic
normalized by `n₁ n₀`. Its exact variance is a sum of four terms — one for
independent score pairs, two for pairs that share a positive or a negative
observation, and one covariance term. Hanley & McNeil (1982) bound the pair
terms by the quantities

```
Q₁ = A / (2 − A)          (bound on P(X₁, X₂ both > Y))
Q₂ = 2 A² / (1 + A)       (bound on P(X > Y₁, Y₂))
```

giving the variance approximation used throughout this repo:

```
SE²(AUC) = [ A(1 − A) + (n₁ − 1)(Q₁ − A²) + (n₀ − 1)(Q₂ − A²) ] / (n₁ n₀).
```

(When both classes are balanced, `n₁ = n₀ = n/2`.)

**Power.** The one-sided test "H₀: AUC ≤ A₀" rejects when
`(Â − A₀) / SE > z_{1−α}`. Plugging in the target AUC `A` for `Â`,

```
power(A, n, α, A₀) = Φ( (A − A₀) / SE(A, n/2, n/2) − z_{1−α} ).
```

`required_samples` inverts this over even `n`; the README headline uses
`A = 0.55`, `A₀ = 0.5`, `α = 0.05`, `power = 0.8`.

## 3. Regression-flag criterion

For every slice `s` the harness stores a baseline JSON containing the slice
AUC `b_s` measured on a reference run. A later evaluation produces, for the
same slice, a scene-clustered BCa bootstrap CI with lower bound `l_s`. With
threshold `τ` (default 0.05), the regression flag is

```
flag = ∃ s :  l_s < b_s − τ      (slice got detectably worse)
     ∨ ∃ s :  s ∈ baseline ∧ s ∉ current report   (slice disappeared)
```

Notes:

- The criterion uses the **CI lower bound**, not the point estimate, so a
  flagged regression is "statistically detectable", not just unlucky sampling;
  point estimates that dip inside the CI do not flag.
- `τ` is the smallest drop considered material (0.05 AUC by default); it is
  configurable via `--threshold` in `python -m dsdbench.evaluate`.
- A missing slice always flags, regardless of `τ`: a model that can no longer
  be scored on a maneuver is a regression by definition.
- The flag is returned by `check_regression` and turned into a non-zero exit
  code by the evaluate CLI, so CI can gate merges on it.

## 4. Why scene-level resampling

Segments from one scene share geometry, traffic, and (for sim segments) the
same rollout machinery, so treating segments as i.i.d. underestimates
uncertainty. All bootstrap CIs in this repo resample scenes with replacement
(Field & Welsh 2007) and the jackknife for the BCa acceleration also drops
scenes. This is the same reason the train/val/test split is assigned per
scene: a segment's neighbors leaking across splits would inflate every metric.
