# scidatasim/export/__init__.py
"""Exporters from scidatasim runs to the formats other tools consume.

* :mod:`scidatasim.export.tasks`       - bundle questions as scored tasks (JSONL)
* :mod:`scidatasim.export.sbi`         - (theta, x) pairs and a batch simulator for simulation-based inference
* :mod:`scidatasim.export.huggingface` - Hugging Face dataset folder with dataset card and Croissant metadata
* :mod:`scidatasim.export.rl`          - OpenEnv environment packages and verifiers tasksets
* :mod:`scidatasim.export.the_well`    - PDE fields as HDF5 in the schema of The Well (PolymathicAI)

Run :func:`scidatasim.audit.audit_export` on an export before publishing it.

None of these modules needs the target library to *write* its format; the
target library is only needed to *use* the result.
"""
