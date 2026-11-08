# simulator/units.py
"""Physical units (optional; needs ``pint``: ``pip install "scientific-data-simulator[units]"``).

Two services:

1. **Unit-bearing parameters.** A model declares the units it computes in::

       class Oscillator(ExperimentLogic):
           parameter_units = {"mass": "kg", "stiffness": "N/m", "x0": "m"}

   and a config may then give values with any compatible units::

       stiffness: "2 kN/m"                 # string form
       x0: {value: 5, units: "cm"}         # mapping form

   The engine converts them to the declared units before the model sees them
   (``2 kN/m`` -> ``2000``), keeps the original strings in the stored config,
   and records the declared units in ``ground_truth.parameter_units``.
   Incompatible units (``"2 kg"`` for a stiffness) are an error.

2. **Unit checks of outputs.** Each output's ``DataDescriptor.units`` is parsed;
   unknown strings and known traps are reported in the run record
   (``provenance.units``). Pint parses some everyday labels as real units -
   ``"units"`` is a *micronit* (luminance) and ``"au"`` an *astronomical unit* -
   so those are flagged rather than silently accepted.

The registry adds a few simulator units: ``individual``/``individuals`` and
``step``/``steps`` (counts), ``index``, ``arbitrary_unit`` (its own dimension, so
it never combines with physical units) and ``time_unit``/``tu`` (a model time of
unspecified physical length).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

#: Strings pint accepts but that almost never mean what the author intended.
TRAPS = {
    "units": "parses as 'micronit' (a luminance unit); use 'dimensionless' or a real unit",
    "unit": "parses as 'micronit'-like nonsense; use 'dimensionless' or a real unit",
    "au": "parses as 'astronomical_unit' (a length); use 'arbitrary_unit' for arbitrary units",
    "a.u.": "is not a valid unit; use 'arbitrary_unit' (or 'astronomical_unit' if meant)",
}

_DEFINITIONS = [
    "individual = 1 * count = _ = individuals",
    "step = 1 * count = _ = steps",
    "index = 1 * count",
    "arbitrary_unit = [arbitrary] = arb",
    "time_unit = [model_time] = tu",
]


def available() -> bool:
    try:
        import pint  # noqa: F401
        return True
    except ImportError:
        return False


@lru_cache(maxsize=1)
def registry():
    """The shared pint registry with the simulator's unit definitions."""
    try:
        import pint
    except ImportError as e:
        raise ImportError("Physical units need 'pint': pip install 'scientific-data-simulator[units]'") from e
    ureg = pint.UnitRegistry()
    for definition in _DEFINITIONS:
        ureg.define(definition)
    return ureg


def parse_units(text: Optional[str]):
    """pint ``Unit`` for ``text`` (``None``/empty -> dimensionless)."""
    ureg = registry()
    if text is None or str(text).strip() == "":
        return ureg.dimensionless
    return ureg.parse_units(str(text))


def check_units(text: Optional[str]) -> Dict[str, Any]:
    """``{"units", "ok", "canonical", "dimensionality", "problem"}`` for a unit string."""
    info: Dict[str, Any] = {"units": text, "ok": True, "canonical": None, "dimensionality": None, "problem": None}
    if text is not None and str(text).strip().lower() in TRAPS:
        info["ok"] = False
        info["problem"] = f"'{text}' {TRAPS[str(text).strip().lower()]}"
    try:
        unit = parse_units(text)
        info["canonical"] = f"{unit:~}" or "dimensionless"
        info["dimensionality"] = str(unit.dimensionality)
    except ImportError:
        raise
    except Exception as e:
        info["ok"] = False
        info["problem"] = info["problem"] or f"unknown unit '{text}' ({type(e).__name__})"
    return info


def check_output_units(results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Unit check for every output descriptor."""
    return {name: check_units(info["descriptor"].units) for name, info in results.items()}


def _split_quantity(value: Any) -> Optional[Tuple[float, str]]:
    """``(magnitude, units)`` if ``value`` carries units, else None."""
    if isinstance(value, dict) and set(value) == {"value", "units"}:
        return float(value["value"]), str(value["units"])
    if isinstance(value, str):
        ureg = registry()
        try:
            quantity = ureg.Quantity(value)
        except Exception:
            return None
        if isinstance(quantity.magnitude, (int, float)) and not quantity.dimensionless:
            return float(quantity.magnitude), f"{quantity.units}"
        if isinstance(quantity.magnitude, (int, float)) and value.strip() != str(quantity.magnitude):
            return float(quantity.magnitude), f"{quantity.units}"
    return None


def declared_parameter_units(cls: type, config: Dict[str, Any]) -> Dict[str, str]:
    """A model's ``parameter_units``: a dict, or a classmethod ``(config) -> dict``
    when the units depend on the configuration (e.g. a ``length_units`` option)."""
    declared = getattr(cls, "parameter_units", None)
    if callable(declared):
        declared = declared(config)
    return dict(declared or {})


def convert_parameters(config: Dict[str, Any], parameter_units: Dict[str, str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Convert unit-bearing config values to the model's declared units.

    Returns ``(converted_config, conversions)`` where ``conversions`` maps each
    converted key to ``{"given": ..., "value": ..., "units": ...}``.
    """
    if not parameter_units:
        return config, {}
    converted = dict(config)
    conversions: Dict[str, Any] = {}
    for key, target in parameter_units.items():
        if key not in config:
            continue
        value = config[key]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            continue  # already in the declared units
        if not (isinstance(value, (str, dict))):
            continue
        if isinstance(value, str):
            try:  # a bare number written as a string ("1e-3") is in the declared units
                converted[key] = float(value)
                conversions[key] = {"given": value, "value": converted[key], "units": target}
                continue
            except ValueError:
                pass
        if not available():
            raise ImportError(f"Parameter '{key}' is given with units ({value!r}); "
                              "install pint: pip install 'scientific-data-simulator[units]'")
        split = _split_quantity(value)
        if split is None:
            continue  # a plain string option, not a quantity
        magnitude, units = split
        ureg = registry()
        try:
            quantity = ureg.Quantity(magnitude, units).to(target)
        except Exception as e:
            raise ValueError(f"Parameter '{key}': cannot convert {value!r} to '{target}' ({e})") from None
        converted[key] = float(quantity.magnitude)
        conversions[key] = {"given": value, "value": converted[key], "units": target}
    return converted, conversions
