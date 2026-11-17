# Example 15: agent-based models with Mesa, and a planted effect to recover

[Mesa](https://mesa.readthedocs.io) is the standard Python framework for agent-based models.
`scidatasim.adapters.mesa.MesaExperiment` runs any Mesa model under the simulator: recorded seeds,
output roles, observation operators, bundles, designs, the leak audit and simulation studies.
The agent-based pack (`scidatasim.experiments.social`) adds three models with a **planted
effect**, a hidden parameter that drives an emergent outcome and that bundles ask back:

| Model | Field | Planted effect | Outputs |
|---|---|---|---|
| `SegregationExperiment` (Schelling) | social science | `homophily`: the share of similar neighbours a resident wants | segregation, unhappy residents, the final grid |
| `EpidemicTrialExperiment` | epidemiology, causal inference | `effect`: how much a randomly assigned treatment lowers the chance of infection per contact | S, I, R; new infections and susceptibles per arm |
| `WealthExchangeExperiment` | economics | `saving`: the share of wealth each agent keeps in a trade | Gini, top-10 % share, total wealth, the final wealth distribution |

All three are labelled `reference`: tests check their invariants (residents, people and wealth are
conserved) and known results (no moves with homophily 0; nobody treated is infected with effect 1;
the wealth distribution's Gini against the Gamma law of Patriarca, Chakraborti and Kaski).

```bash
pip install "scidatasim[mesa]"       # Mesa 3.0 or newer
python -m examples.example_15_agent_based.run_experiment           # about 45 s
python -m examples.example_15_agent_based.run_experiment --quick   # about 12 s
```

## Part 2: three estimators of the trial's effect

The study runs the trial 100 times for each planted effect (0, 0.3, 0.6) in a small and a large
epidemic, and compares three standard estimators computed from case counts per arm:

| mechanism | attack-rate ratio | hazard ratio | stratified hazard ratio |
|---|---|---|---|
| effect 0.3, small epidemic | -0.094 (0.023) | -0.034 (0.025) | -0.038 (0.025) |
| effect 0.6, small epidemic | -0.129 (0.036) | -0.089 (0.037) | -0.092 (0.036) |
| effect 0.3, large epidemic | -0.256 (0.002) | -0.083 (0.008) | -0.053 (0.008) |
| effect 0.6, large epidemic | -0.395 (0.004) | -0.080 (0.007) | -0.062 (0.006) |

Bias (Monte Carlo SE). With no effect, all three are unbiased.

- The **attack-rate ratio** (1 - share of treated infected / share of controls infected) fails in
  large epidemics: almost everyone is infected in both arms, so it estimates 0.3 as about 0.04.
- The **hazard ratio** (infections per susceptible-step) is much closer, and comparing the arms
  within each time step (Mantel-Haenszel) helps a little more.
- A bias of about -0.06 remains. Controls in places with much infection are infected first, so
  the controls still susceptible later are less exposed than the treated still susceptible: the
  *selective depletion of susceptibles* that makes hazard ratios hard to read (M. A. Hernán,
  "The hazards of hazard ratios", Epidemiology 21:13-15, 2010). The planted effect makes this
  visible; in real data the truth is unknown.

## Wrapping your own Mesa model

```yaml
experiment_type: scidatasim.adapters.mesa.MesaExperiment
model: mymodels.city.CityModel            # its constructor must accept rng= (or seed=)
model_params: {density: 0.8, homophily: 0.4}
n_steps: 100
reporters: {happy: n_happy, segregation: "mymodels.city:segregation_index"}
snapshots: {grid: grid_array}
observation: {targets: [happy], operators: [{type: poisson, scale: 1.0}]}
```

Reporters are attributes or functions of the model and must give numbers; snapshots give arrays
(recorded after the last step). Use `scidatasim.adapters.mesa.init_mesa_model(self, rng)` in the
model's `__init__` to pass the seed to Mesa whatever its version. Models from outside the
simulator are labelled `unverified`.
