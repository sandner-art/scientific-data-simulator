# experiments/example_random_walk/logic.py
"""Symmetric 1-D lattice random walk: x_{k+1} = x_k + step_size * s_k, s_k in {-1, +1}.

``generator`` chooses where the coin flips come from:

* ``numpy`` (default, ``reference``) - the engine's seeded ``numpy.random.Generator`` (PCG64).
* ``randu`` (``audit``) - IBM's RANDU, x_{k+1} = 65539 x_k mod 2^31, seeded from the engine's
  generator. RANDU is a classic *bad* generator: every three consecutive outputs satisfy
  x_{k+2} = 6 x_{k+1} - 9 x_k (mod 2^31), so triples of uniforms lie on just 15 planes in the unit
  cube, and the low bits are periodic. Steps take the top bit, so the walk looks fine to the eye
  and to mean/variance checks; the raw uniforms (``uniforms``, auxiliary) let a method find the
  defect. Kept as a deliberately weak model for testing randomness diagnostics.
"""
import numpy as np

from simulator.base import ExperimentLogic
from simulator.utils import DataDescriptor, DataType

GENERATORS = ("numpy", "randu")
RANDU_MULTIPLIER = 65539
RANDU_MODULUS = 2 ** 31


def randu(seed, n):
    """``n`` RANDU uniforms in (0, 1) from an odd integer ``seed`` (forced odd)."""
    x = (int(seed) % RANDU_MODULUS) | 1
    out = np.empty(n)
    for i in range(n):
        x = (RANDU_MULTIPLIER * x) % RANDU_MODULUS
        out[i] = x / RANDU_MODULUS
    return out


class RandomWalkExperiment(ExperimentLogic):
    """Symmetric 1-D lattice random walk; see the module docstring for the ``generator`` option."""

    inferable_parameters = ("step_size",)

    def __init__(self, config):
        self.n_steps = int(config['n_steps'])
        self.step_size = config['step_size']
        self.generator = config.get('generator', 'numpy')
        if self.generator not in GENERATORS:
            raise ValueError(f"generator must be one of {GENERATORS}")
        self.positions = [0]
        self.steps = [0]
        self.uniforms = None

    def fidelity(self, config):
        if self.generator == "randu":
            return {"level": "audit", "note": (
                "Coin flips from RANDU (x_{k+1} = 65539 x_k mod 2^31), a known defective generator: "
                "consecutive uniform triples lie on 15 planes. Means and variances look right; "
                "use it to test whether randomness diagnostics catch a bad generator.")}
        return {"level": "reference", "note": (
            "Exact simulation of the lattice walk; seeded reproducibility and increment statistics "
            "(mean 0, variance step_size^2) are tested.")}

    def initialize(self, config):
        rng = self.get_rng(config)  # engine-provided generator, or one seeded from config['seed']
        self.positions = [0]
        self.steps = [0]
        if self.generator == "randu":
            self.uniforms = randu(rng.integers(1, RANDU_MODULUS), self.n_steps)
        return {'step': 0, 'position': 0}

    def run_step(self, state, step):
        if self.generator == "randu":
            step_direction = 1 if self.uniforms[step] >= 0.5 else -1
        else:
            step_direction = self.get_rng().choice([-1, 1])
        new_position = state['position'] + self.step_size * step_direction
        self.steps.append(step + 1)
        self.positions.append(new_position)
        return {'step': step + 1, 'position': new_position}

    def get_results(self):
        results = {
            "step": {
                "data": np.array(self.steps),
                "descriptor": DataDescriptor("step", DataType.NDARRAY, units="steps", group="time_series", x_axis='step', role="coordinate")
            },
            "position": {
                "data": np.array(self.positions, dtype=float),
                "descriptor": DataDescriptor("position", DataType.NDARRAY, units="dimensionless", group="time_series", plot_type="line", x_axis="step", role="latent")
            }
        }
        if self.uniforms is not None:
            results["uniforms"] = {
                "data": self.uniforms[:len(self.steps) - 1].copy(),
                "descriptor": DataDescriptor("uniforms", DataType.NDARRAY, units="dimensionless", group="generator",
                                             role="auxiliary"),
            }
        return results
