# Example 10: law discovery

A benchmark for discovering laws from noisy data. Each run hides a law y = f(x) from one of nine
families and shows only noisy samples of it. The figure is drawn without the exact curve.

| Family | Form (textbook structural constant) |
|---|---|
| `power_law` | a·x^b (b = 1.5) |
| `inverse_square` | a / x^p (p = 2) |
| `exponential_decay` | a·exp(−k·x) |
| `logistic` | K / (1 + exp(−r·(x − x0))) |
| `michaelis_menten` | V·x / (K + x) |
| `hill` | V·x^n / (K^n + x^n) (n = 2) |
| `damped_oscillation` | A·exp(−γ·x)·cos(ω·x) |
| `gaussian_peak` | A·exp(−(x − μ)² / (2σ²)) |
| `arrhenius` | A·exp(−E / x) |

Three families have a structural constant (in brackets). With `counterfactual: true` it is
moved at least 15 % away from its textbook value (an inverse-2.3 law instead of inverse-square, a
Hill coefficient of 2.6), so a system that recalls
textbook formulas instead of fitting the data is caught.

Besides the generic figure questions (maximum, minimum, final value), each bundle asks three
questions about the law. The generic `argmax` and `peak_count` questions are excluded in
`config.yaml`: for mostly monotone laws their answers hardly vary, and a blind guess answers
them 56–78 % of the time (example 11 measures this).

* **law family**: a multiple-choice question among four families;
* **law formula**: answered with a formula. It is scored by evaluating the formula against the
  noise-free law on the data range *and half of that range beyond it* (relative RMS error), so
  `2.3*x**1.5`, `y = 2.3 x^{1.5}` and `2.3*sqrt(x)**3` all count. On the data range alone a
  polynomial fitted to the data would pass for five of the nine families without finding the law.
  The tolerance is the larger of 1 % and the error a least-squares fit of the true family reaches
  in 90 % of replicates of the run's own noise (1 % at 40 dB; at 20 dB a median of 4 %, at most 10 %): a fixed 1 %
  could not be met at 20 dB even with the right law. `formula_tolerance: 0.01` fixes it instead.
  Power laws stay partly within a polynomial's reach (exponents near 1 or 2), so each formula
  question carries a private label `polynomial_reachable`, and the baseline's formula score is
  also printed without those items;
* **extrapolation**: the value of y a quarter of the range beyond the data.

## Run

```bash
python -m examples.example_10_law_discovery.run_experiment             # suite + baseline
python -m examples.example_10_law_discovery.run_experiment --export    # + Hugging Face dataset + leak audit
python -m examples.example_10_law_discovery.run_experiment --env 12    # interactive environment
```

The suite is a full design of 9 families × textbook/counterfactual × SNR 40/20 dB (36 runs).
All families share the x range [0.5, 8], so the range does not give the family away. The rows
are shuffled in a fresh, unrecorded order on every run (`--suite-seed N` fixes it): a known order
would map each run's position back to its family. So each invocation draws different laws.

The baseline fits every family with `scipy.optimize.curve_fit` from three starts and keeps the
best by AIC (all constants positive). Over three invocations it scored 0.86–0.94 (family),
0.89–0.97 (formula) and 0.89–0.92 (extrapolation), and a mean reward of 1.00 in 12 environment
episodes. The formula score can exceed the family score: at 20 dB two families can fit the same
samples almost equally well, and a formula of the wrong family may still match the law within the
noise-set tolerance on the scored range.

`--env N` plays the interactive version (`env.yaml`): an agent chooses the measurement range
within [0.5, 5] (up to three experiments), then answers with a formula, which is scored on
[0.5, 7.25] within 2 %. When the scored range lay inside the measurable one, a polynomial fitted
to three full-range experiments passed 95 % of episodes; now it passes about 25 %, all of them
power laws. The baseline's mean reward is printed.

`--export` writes a Hugging Face dataset and then runs `scidatasim.audit.audit_export` on it,
using the private runs as the answer key. It should print `PASS`: answers, seeds, config hashes,
the hidden design columns and the design-point index are not in the public files, and runs are
listed in the order of their random ids.

In the environment, the hidden `law_seed` is drawn from [0, 4·10^15], too many to enumerate. An
experiment that fails (for example `x_min > x_max`) returns only a generic error and uses budget,
so failures cannot be used to probe the hidden law for free.

## Files

* `config.yaml`: base configuration (the suite sweeps `law`, `counterfactual` and the noise level).
* `env.yaml`: the interactive environment (hidden law and parameters, controllable x range).
* `run_experiment.py`: the suite, the baseline (`fit_best_law`), the environment loop and the export.
