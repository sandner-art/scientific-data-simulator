# simulator/rng.py
"""Seeded, recorded randomness.

Every run has one integer ``seed`` (taken from the config, or drawn and then
recorded when absent). From it, independent named streams are derived with
``numpy.random.SeedSequence``:

* ``model``       - passed to the ExperimentLogic as ``self.rng``
* ``observation`` - used by observation operators (noise, missingness, ...)
* ``bundle``      - used to pick question positions in run bundles

``SeedSequence.spawn`` children are keyed by index, so adding a stream never
changes the existing ones.

Because the streams are independent, changing the noise model does not change
the realisation of the mechanism: the latent trajectory for a given seed is the
same whatever observation operators are applied to it.
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

STREAMS = ("model", "observation", "bundle")
MAX_SEED = 2 ** 63 - 1


def new_seed() -> int:
    """A fresh random seed suitable for recording in JSON/YAML."""
    return secrets.randbits(63)


def validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be a non-negative integer, got {seed!r}")
    seed = int(seed)
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be in [0, 2**63 - 1], got {seed}")
    return seed


def derive_seeds(base_seed: int, n: int) -> List[int]:
    """``n`` independent child seeds derived deterministically from ``base_seed``.

    Each child is an ordinary integer, so any single row of a design can be
    re-run on its own with ``seed: <child>``.
    """
    children = np.random.SeedSequence(validate_seed(base_seed)).spawn(n)
    return [int(c.generate_state(1, np.uint64)[0] >> np.uint64(1)) for c in children]


@dataclass
class RunRandomness:
    seed: int
    generated: bool  # True if the seed was drawn because the config had none

    @classmethod
    def from_config_seed(cls, seed: Optional[int]) -> "RunRandomness":
        if seed is None:
            return cls(new_seed(), True)
        return cls(validate_seed(seed), False)

    def streams(self) -> Dict[str, np.random.Generator]:
        children = np.random.SeedSequence(self.seed).spawn(len(STREAMS))
        return {name: np.random.Generator(np.random.PCG64(child)) for name, child in zip(STREAMS, children)}

    def to_dict(self) -> Dict[str, object]:
        return {"seed": self.seed, "seed_generated": self.generated,
                "bit_generator": "PCG64", "streams": list(STREAMS),
                "derivation": "numpy.random.SeedSequence(seed).spawn(len(streams))"}
