#!/usr/bin/env python3
# examples/example_8_geometry_benchmark/run_experiment.py
"""Generate a small geometry benchmark and export it as a Hugging Face dataset.

Each run draws one random triangle (seeded) and writes a diagram bundle:
figure.png drawn by its own plot_figure.py from elements.csv, a caption, and
questions with exact answers (area, angles, circle radii, triangle type, ...).
The design collects all questions into questions.jsonl / qa.jsonl; the export
writes a dataset folder with a dataset card and Croissant metadata, with the
answers withheld.

    python -m examples.example_8_geometry_benchmark.run_experiment --n 20
    python -m examples.example_8_geometry_benchmark.run_experiment --n 20 --bare   # no axes: harder
"""
import argparse
import logging
import os

import pandas as pd

from simulator.config import load_config
from simulator.doe import run_design


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--config", default=os.path.join(here, "config.yaml"))
    parser.add_argument("--output_dir", default=os.path.join("experiments_output", "example_8_geometry_benchmark"))
    parser.add_argument("--n", type=int, default=12, help="number of triangles")
    parser.add_argument("--bare", action="store_true", help="draw diagrams without coordinate axes")
    parser.add_argument("--no-export", action="store_true", help="skip the Hugging Face export")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    base = load_config(args.config)
    if args.bare:
        base["bundle"]["figure"]["axes"] = False
    # One design point repeated n times: every replicate gets its own seed, so its own triangle.
    design = run_design(base, pd.DataFrame({"extent": [base["extent"]]}), output_dir=args.output_dir,
                        replicates=args.n)
    print(f"{len(design.table)} triangles in {design.output_dir} (base seed {design.base_seed})")

    if not args.no_export:
        from simulator.export.huggingface import export_huggingface
        path = export_huggingface(args.output_dir, os.path.join(args.output_dir, "hf_dataset"),
                                  name="sds-triangle-diagrams", overwrite=True,
                                  description="Triangle diagrams with exact constructions and verifiable questions.")
        print(f"Hugging Face dataset folder: {path}")


if __name__ == "__main__":
    main()
