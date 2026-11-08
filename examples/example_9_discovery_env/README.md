# Example 9: an interactive discovery environment

`env.yaml` defines a task. A predator–prey system has a hidden prey growth rate, drawn anew in every episode. The agent may run three experiments, each time choosing the initial populations, and gets noisy, sampled counts back. It then reports its estimate, and an answer within 5 % earns reward 1.

```bash
python -m examples.example_9_discovery_env.run_experiment --episodes 20
python -m examples.example_9_discovery_env.run_experiment --show          # print one transcript
```

The scripted baseline removes the predators, fits the exponential growth of the prey, and solves almost every episode. That shows the task is solvable and the scoring works. To make it harder, forbid removing the predators (`initial_predators: [1, 20]`), hide more parameters, reduce the budget, or ask for the functional form instead (`hidden: {functional_response: {choices: [linear, holling2]}}`).

## Using it with agents and RL frameworks

```python
from simulator.config import yaml_load
from simulator.envs import ExperimentEnv

env = ExperimentEnv(yaml_load(open("examples/example_9_discovery_env/env.yaml")))
obs = env.reset(seed=0)                     # obs["task"] is the instruction text
out = env.step({"type": "experiment", "inputs": {"initial_prey": 20, "initial_predators": 2}})
out = env.step({"type": "answer", "value": 1.1})
out["reward"], out["info"]["truth"]
```

- **OpenEnv:** `simulator.export.rl.export_openenv("examples/example_9_discovery_env/env.yaml", "rl/", name="lv_discovery")` writes an OpenEnv package. Serve it with `uvicorn server.app:app` inside the package, or as a Docker image or a Hugging Face Space.
- **verifiers:** the `openenv` taskset of verifiers can drive that server for multi-turn training. See the module docstring of `simulator/export/rl.py`.

Each episode is reproducible from its seed. Every experiment uses its own seed, derived from the episode seed.

## A note on the noise model

The example uses multiplicative (log-normal) noise. With predators removed, the prey grows by orders of magnitude, and `snr_db` would set *one* noise level from the RMS of the whole series. That noise would swamp the small early values, which carry most of the information about the growth rate.
