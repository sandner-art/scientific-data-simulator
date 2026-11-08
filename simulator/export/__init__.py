# simulator/export/__init__.py
"""Exporters from simulator runs to the formats other tools consume.

* :mod:`simulator.export.tasks`       - bundle questions as scored tasks (JSONL)
* :mod:`simulator.export.sbi`         - (theta, x) pairs and a batch simulator for simulation-based inference
* :mod:`simulator.export.huggingface` - Hugging Face dataset folder with dataset card and Croissant metadata
* :mod:`simulator.export.rl`          - OpenEnv environment packages and verifiers tasksets
* :mod:`simulator.export.the_well`    - PDE fields as HDF5 in the schema of The Well (PolymathicAI)

Run :func:`simulator.audit.audit_export` on an export before publishing it.

None of these modules needs the target library to *write* its format; the
target library is only needed to *use* the result.
"""
