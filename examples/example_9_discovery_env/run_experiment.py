#!/usr/bin/env python3
# examples/example_9_discovery_env/run_experiment.py
"""Play the discovery environment with a simple scripted baseline agent.

The agent removes the predators (initial_predators = 0), so the prey grows
exponentially, x(t) = x0 exp(a t), and estimates the growth rate a from a
straight-line fit to log(prey) in the early, low-noise part of the series.
This shows the task is solvable; replace `baseline_agent` with an LLM or RL
policy (see simulator.export.rl for OpenEnv and verifiers packaging).

    python -m examples.example_9_discovery_env.run_experiment --episodes 20
"""
import argparse
import os

import numpy as np

from simulator.config import yaml_load

from simulator.envs import ExperimentEnv, render_observation


def baseline_agent(env, seed):
    obs = env.reset(seed=seed)
    transcript = [render_observation(obs)]
    result = env.step({"type": "experiment", "inputs": {"initial_prey": 5, "initial_predators": 0}})
    transcript.append(render_observation(result["observation"]))
    m = result["observation"]["measurements"]["prey_population_observed"]
    t, x = np.array(m["time_observed"], dtype=float), np.array(m["values"], dtype=float)
    ok = np.isfinite(x) & (x > 0) & (t <= 3.0)       # early growth: highest relative precision
    slope = np.polyfit(t[ok], np.log(x[ok]), 1)[0]
    result = env.step({"type": "answer", "value": float(slope)})
    return result, transcript


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--spec", default=os.path.join(here, "env.yaml"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--show", action="store_true", help="print the first transcript")
    args = parser.parse_args()
    with open(args.spec, encoding="utf-8") as f:
        env = ExperimentEnv(yaml_load(f))

    rewards, errors = [], []
    for episode in range(args.episodes):
        result, transcript = baseline_agent(env, seed=episode)
        rewards.append(result["reward"])
        errors.append(result["info"]["relative_error"])
        if args.show and episode == 0:
            print("\n\n".join(transcript))
    print(f"baseline: mean reward {np.mean(rewards):.2f} over {args.episodes} episodes, "
          f"median relative error {np.median(errors):.3f}")


if __name__ == "__main__":
    main()
