# Predator–prey experiment (Lotka–Volterra)

`logic.py` contains `PredatorPreyExperiment`.

## Model

```
dPrey/dt      = a·Prey·(1 − Prey/K) − b·Prey·Predators / (1 + b·h·Prey)
dPredators/dt = c·Prey·Predators / (1 + b·h·Prey) − d·Predators
```

| Symbol | Config key | Meaning |
|---|---|---|
| a | `prey_growth_rate` | intrinsic prey growth rate |
| b | `prey_death_rate` | predation rate |
| c | `predator_growth_rate` | conversion of prey into predator growth |
| d | `predator_death_rate` | predator death rate |
| K | `prey_carrying_capacity` | optional; logistic prey growth (default: none, i.e. K = ∞) |
| h | `handling_time` | optional; with `functional_response: holling2` (default: `linear`, h = 0) |

With the defaults (no K, linear response) this is the classical Lotka–Volterra system, which conserves V = c·x − d·ln x + b·y − a·ln y.

## Numerical options

| Key | Default | Notes |
|---|---|---|
| `n_steps` | (required) | number of steps |
| `dt` | `1.0` | time step; the output time axis is `step · dt` |
| `integrator` | `euler` | `euler` (first order) or `rk4` (fourth order) |
| `clamp_negative` | `true` | clip negative populations to 0 after each step |

The defaults reproduce version 0.1.0 exactly (forward Euler, dt = 1, clamping). That configuration is kept on purpose as a simple **teaching / audit model**: its trajectories contain visible integrator error, and it violates the conserved quantity by more than 1 %. It is useful for testing whether a method notices model error, but it is not reference data.

For reference-quality trajectories use, for example:

```yaml
integrator: rk4
dt: 0.01
clamp_negative: false
```

`tests/test_verification.py` checks this configuration:

- it agrees with SciPy's DOP853 solver (rtol 1e-12) to 1e-7;
- its observed convergence order is 4;
- it conserves the Lotka–Volterra invariant to 1e-8 over t = 50;
- the Holling II and logistic options match their references or closed forms.

## Counterfactual laws

`functional_response: holling2` and `prey_carrying_capacity` are natural targets for the engine's `law_shift:` option. The model then follows a modified law, and `ground_truth.json` records both the original and the shifted setting.
