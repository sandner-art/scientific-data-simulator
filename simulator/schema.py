# simulator/schema.py
"""Validation of the engine-level configuration keys.

Model parameters are free-form (whatever the ExperimentLogic reads); only the
keys the engine itself interprets are validated here.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

STATIC_FORMATS = (None, "svg", "pdf", "png", "jpeg", "webp")

# Keys interpreted by the engine rather than by the model. They are excluded
# when the engine infers a model's ground-truth parameters from the config.
ENGINE_KEYS = frozenset({
    "experiment_type", "experiment_description", "n_steps", "seed",
    "generate_plots", "static_plot_format", "save_csv",
    "observation", "law_shift", "storage", "bundle",
})


# Engine keys a law shift must not touch: they are fixed before the model runs,
# so a "shift" of them would be recorded without taking effect. (n_steps is
# allowed: the step loop reads the shifted model config.)
LAW_SHIFT_FORBIDDEN = ENGINE_KEYS - {"n_steps"}


class EngineConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    experiment_type: str
    experiment_description: str = ""
    n_steps: int = Field(1, ge=0)
    seed: Optional[int] = Field(None, ge=0, le=2 ** 63 - 1)
    generate_plots: bool = True
    static_plot_format: Optional[str] = None
    save_csv: bool = False
    observation: Optional[Dict[str, Any]] = None
    law_shift: Optional[Dict[str, Any]] = None
    storage: Optional[Dict[str, Any]] = None
    bundle: Any = None

    @field_validator("static_plot_format", mode="before")
    @classmethod
    def _normalise_format(cls, v: Any) -> Any:
        if isinstance(v, str) and v.lower() in ("null", "none", ""):
            return None
        if v not in STATIC_FORMATS:
            raise ValueError(f"must be one of {STATIC_FORMATS}")
        return v


def validate_engine_config(config: Dict[str, Any]) -> EngineConfig:
    """Validate engine keys; raise ``ValueError`` with a readable message."""
    try:
        return EngineConfig.model_validate(config)
    except ValidationError as e:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '<config>'}: {err['msg']}" for err in e.errors()
        )
        raise ValueError(f"Invalid configuration: {problems}") from None


def parse_law_shift(spec: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Normalise ``law_shift`` to ``{"changes": {...}, "description": str|None}``.

    Accepts either ``{changes: {key: value}, description: "..."}`` or a flat
    mapping of (possibly dotted) config keys to their counterfactual values.
    """
    if not spec:
        return None
    if "changes" in spec:
        extra = set(spec) - {"changes", "description"}
        if extra:
            raise ValueError(f"law_shift: unexpected keys {sorted(extra)}")
        changes, description = spec["changes"], spec.get("description")
    else:
        changes, description = {k: v for k, v in spec.items() if k != "description"}, spec.get("description")
    if not isinstance(changes, dict) or not changes:
        raise ValueError("law_shift.changes must be a non-empty mapping")
    engine_targets = sorted(k for k in changes if k.split(".", 1)[0] in LAW_SHIFT_FORBIDDEN)
    if engine_targets:
        raise ValueError(f"law_shift may only change model parameters, not engine settings: {engine_targets}. "
                         "Change seed/observation/storage directly in the config (or sweep them in a design).")
    return {"changes": dict(changes), "description": description}
