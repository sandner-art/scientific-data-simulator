# Example 14: a simulation study in the ADEMP structure

A simulation study compares methods on data whose truth is known. `scidatasim.ademp` runs one in
the structure of Morris, White and Crowther (2019): **A**ims, **D**ata-generating mechanisms (the
rows of a design table, applied to any simulator configuration), **E**stimands (read from each
run's ground truth), **M**ethods (functions that see only the observations) and **P**erformance
measures with Monte Carlo standard errors.

```bash
python -m examples.example_14_simulation_study.run_experiment            # about 10 s
python -m examples.example_14_simulation_study.run_experiment --quick    # about 3 s
```

Here three estimators recover the slope of y = 2 x + 1 (Gaussian noise, sigma 1) with 20 or 100
points, when the measurements are clean, have 10 % outliers (the `outliers` operator), or have a
detection limit at y = 6 (the `censor` operator). The number of repetitions, 475, gives a Monte
Carlo SE of about 0.01 for a 95 % coverage (`n_sim_for(0.01, "coverage")`).

## Results (seed 14)

| problem | n | method | bias (MCSE) | empirical SE | coverage (MCSE) |
|---|---|---|---|---|---|
| none | 20 | OLS | 0.001 (0.003) | 0.071 | 0.964 (0.009) |
| none | 20 | Theil-Sen | 0.000 (0.003) | 0.075 | 0.966 (0.008) |
| outliers 10 % | 20 | OLS | -0.033 (0.036) | 0.782 | 0.956 (0.009) |
| outliers 10 % | 20 | Theil-Sen | -0.006 (0.005) | 0.116 | 0.943 (0.011) |
| outliers 10 % | 20 | Huber | -0.012 (0.018) | 0.392 | |
| detection limit | 100 | OLS | -0.324 (0.001) | 0.025 | 0.000 (0.000) |
| detection limit | 100 | Theil-Sen | -0.314 (0.001) | 0.029 | 0.000 (0.000) |

- Without problems the three agree, and OLS and Theil-Sen cover close to 95 %.
- Outliers leave every method unbiased (their signs are random) but make OLS about seven times
  less precise than Theil-Sen; OLS's interval stays valid only because its SE grows too.
- A detection limit biases every method by about -0.32, and more data only narrows the
  intervals around the wrong value: coverage falls to zero. No method in this study handles
  censoring; a censored-regression (Tobit) method would be the next one to add.

`report.md` has every mechanism and measure; `estimates.csv` holds one row per repetition, method
and estimand, with the seed that re-creates that repetition alone; `study.json` records the
ADEMP description and the simulator version.

## Writing your own

```python
from scidatasim.ademp import Estimate, SimulationStudy, n_sim_for

def my_method(data):          # data: the run's observations and coordinates only
    ...
    return {"slope": Estimate(estimate, se=se)}   # or lower/upper, p_value; or just a number

study = SimulationStudy(aims="...", base_config=config, mechanisms=design_table,
                        estimands={"slope": "m"}, methods={"mine": my_method, "ols": ols},
                        n_sim=n_sim_for(0.01, "coverage"))
result = study.run("my_study")
print(result.markdown())
```

An estimand is the name of a parameter in the run's ground truth, a label output, or a function
of the run. Methods that raise or return no finite estimate are counted as failures and left out
of the measures. Reference: T. P. Morris, I. R. White, M. J. Crowther (2019), *Using simulation
studies to evaluate statistical methods*, Statistics in Medicine 38:2074-2102.
