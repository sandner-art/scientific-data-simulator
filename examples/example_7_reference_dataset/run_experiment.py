#!/usr/bin/env python3
# examples/example_7_reference_dataset/run_experiment.py
"""Build a small reference dataset: noise level x replicate design.

Every run writes:
  data/ground_truth/   exact trajectories (latent)
  data/observation/    sampled, noisy observations
  data/coordinate/     time axes
  ground_truth.json    parameters, realised noise, law shift (the answers)
  experiment_record.json  provenance: seed, config hash, code hash, versions, checksums
  bundle/              figure + plotting code + table + caption + Q&A + mechanism graph

and the design directory gets design.csv / design_manifest.json /
design_results.csv (one row per run, with its seed and experiment ID), plus
questions.jsonl / qa.jsonl collecting every run's bundle questions.

    python -m examples.example_7_reference_dataset.run_experiment
    python -m examples.example_7_reference_dataset.run_experiment --law-shift
"""
import argparse
import logging
import os

from simulator.config import load_config
from simulator.doe import create_doe_table, run_design


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--config", default=os.path.join(here, "config.yaml"))
    parser.add_argument("--output_dir", default=os.path.join("experiments_output", "example_7_reference_dataset"))
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--law-shift", action="store_true",
                        help="use a counterfactual law: saturating (Holling II) predation")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    base = load_config(args.config)
    if args.law_shift:
        base["law_shift"] = {"changes": {"functional_response": "holling2", "handling_time": 0.3},
                             "description": "Holling type II predation instead of mass action"}

    # Noise levels to sweep; the dotted key addresses the first observation operator.
    design = create_doe_table({"observation.operators.0.snr_db": [30, 20, 10]})
    out = run_design(base, design, output_dir=args.output_dir, replicates=args.replicates,
                     seed_mode="per_replicate", n_jobs=args.n_jobs)
    # seed_mode="per_replicate": the three noise levels of one replicate share a
    # seed, so they observe the same latent trajectory with different noise.

    print(out.table[["design_point", "replicate", "observation.operators.0.snr_db", "seed", "experiment_id"]])
    print(f"\nDataset written to {out.output_dir} (base seed {out.base_seed})")


if __name__ == "__main__":
    main()
