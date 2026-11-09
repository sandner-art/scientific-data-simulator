# simulator/llm.py
"""LLM-assisted model drafting, gated by the automatic model checker.

The simulator does not depend on any LLM provider. Wrap whatever client you use
in a function ``prompt -> text`` and pass it to :class:`CallableLLMClient`::

    import anthropic
    from simulator.llm import CallableLLMClient, draft_model

    sdk = anthropic.Anthropic()
    def ask(prompt):
        msg = sdk.messages.create(model="<model id>", max_tokens=4000,
                                  messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text

    draft = draft_model(CallableLLMClient(ask), description="A pendulum with linear drag ...",
                        config={"n_steps": 2000, "dt": 0.01, "length": 1.0, ...})
    if draft.accepted:
        draft.save("experiments/pendulum/logic.py")

:func:`draft_model` asks for an ``ExperimentLogic`` class, runs
:func:`simulator.checker.check_source` on the answer (static lint first; then
the code runs in a separate, time-limited process), and on failure sends the
checker's findings back for another attempt. Every attempt (prompt hash, code
hash, report) is kept in :attr:`Draft.attempts` for the record.

An accepted draft is *not* reference data: generated models start at fidelity
``unverified`` and need verification tests before anything else.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .base import LLMClient
from .checker import CheckReport, check_source


class CallableLLMClient(LLMClient):
    """Adapter turning a ``prompt -> text`` function into an :class:`LLMClient`."""

    def __init__(self, generate: Callable[[str], str], name: str = "custom"):
        self._generate = generate
        self.name = name
        self.calls = 0
        self.prompt_chars = 0
        self.response_chars = 0

    def generate_text(self, prompt: str, **kwargs: Any) -> str:
        self.calls += 1
        self.prompt_chars += len(prompt)
        text = self._generate(prompt)
        self.response_chars += len(text or "")
        return text

    def generate_structured_output(self, prompt: str, schema: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        text = self.generate_text(prompt + "\n\nRespond with JSON matching this schema:\n" + json.dumps(schema))
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise ValueError("The model did not return JSON")
        return json.loads(match.group(0))

    def get_usage_info(self) -> Dict[str, Any]:
        return {"client": self.name, "calls": self.calls, "prompt_chars": self.prompt_chars,
                "response_chars": self.response_chars}


PROMPT = textwrap.dedent('''
    Write a Python module defining one class `{class_name}` for the Scientific Data Simulator.

    Model to implement:
    {description}

    The class must subclass `simulator.base.ExperimentLogic` and implement
      - `__init__(self, config)`: read parameters from the dict `config` (keys: {keys})
      - `initialize(self, config) -> dict`: initial state
      - `run_step(self, state, step) -> dict`: advance one step (called config["n_steps"] times)
      - `get_results(self) -> dict`: {{name: {{"data": numpy array or float, "descriptor": DataDescriptor(...)}}}}
    Use `from simulator.utils import DataDescriptor, DataType` and give every output
    `DataDescriptor(name, DataType.NDARRAY, units=..., group="time_series", x_axis="time", role=...)`
    with role "coordinate" for the time axis and "latent" for model states (plot_type="line" for states).

    Rules (checked automatically):
      - Draw all random numbers from `self.rng` (a numpy Generator set by the engine); never use
        `np.random.<function>` or the `random` module.
      - No file, network, subprocess, eval/exec or other side effects.
      - Results must be deterministic for a given seed and must not modify `config`.
      - If the model has a time step `dt`, the solution must converge when `dt` is halved.
      - Every parameter listed in `inferable_parameters` must influence the output.
      - Set class attributes: `fidelity_level = "unverified"`, `fidelity_note = "LLM-generated; not verified"`,
        `inferable_parameters = (...)`, and `parameter_units = {{...}}` with pint unit strings (e.g. "N/m").
      - Optionally implement `invariants(self, results) -> dict` with conserved quantities.

    Return only the module source in one ```python code block.
''').strip()

FEEDBACK = textwrap.dedent('''
    The automatic model checker rejected your previous code:
    {feedback}

    Previous code:
    ```python
    {code}
    ```
    Fix every problem and return the complete corrected module in one ```python code block.
''').strip()


def extract_code(text: str) -> str:
    """The first fenced Python code block of an answer (or the whole text if there is none)."""
    match = re.search(r"```(?:python|py)?\s*\n(.*?)```", text or "", re.S)
    return (match.group(1) if match else (text or "")).strip() + "\n"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Draft:
    class_name: str
    source: str
    report: CheckReport
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    usage: Dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.report.accepted

    def save(self, path: str, overwrite: bool = False) -> str:
        """Write the accepted source (plus the check report next to it)."""
        if not self.accepted:
            raise ValueError("Refusing to save a draft that failed the model checker:\n" + self.report.feedback())
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(path)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.source)
        with open(os.path.splitext(path)[0] + "_check.md", "w", encoding="utf-8") as f:
            f.write(self.report.to_markdown())
        init = os.path.join(os.path.dirname(os.path.abspath(path)), "__init__.py")
        if not os.path.exists(init):
            open(init, "w").close()
        return path

    def provenance(self) -> Dict[str, Any]:
        """Summary suitable for ``ExperimentRecord.llm_usage``."""
        return {"class_name": self.class_name, "accepted": self.accepted, "source_sha256": _sha(self.source),
                "attempts": [{k: a[k] for k in ("prompt_sha256", "source_sha256", "accepted", "failures")}
                             for a in self.attempts],
                "usage": self.usage}


def draft_model(client: LLMClient, description: str, config: Dict[str, Any], class_name: str = "GeneratedModel",
                max_attempts: int = 3, timeout: float = 120.0, **check_options: Any) -> Draft:
    """Ask ``client`` for a model, gate it with the checker, retry with feedback."""
    keys = ", ".join(sorted(k for k in config if k not in ("experiment_type", "seed")))
    prompt = PROMPT.format(class_name=class_name, description=description.strip(), keys=keys)
    attempts: List[Dict[str, Any]] = []
    source, report = "", CheckReport(f"<generated>.{class_name}")
    for _ in range(max(1, int(max_attempts))):
        answer = client.generate_text(prompt)
        source = extract_code(answer)
        report = check_source(source, class_name, config, timeout=timeout, **check_options)
        attempts.append({"prompt_sha256": _sha(prompt), "source_sha256": _sha(source),
                         "accepted": report.accepted, "failures": [r.name for r in report.failures],
                         "report": report.to_dict()})
        if report.accepted:
            break
        prompt = FEEDBACK.format(feedback=report.feedback(), code=source)
    usage = client.get_usage_info() if hasattr(client, "get_usage_info") else {}
    return Draft(class_name, source, report, attempts, usage)
