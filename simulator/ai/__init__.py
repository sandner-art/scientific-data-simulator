# simulator/ai/__init__.py
"""Language models in four roles, all through one logged, replayable client.

* :mod:`simulator.ai.clients`  - :class:`LoggedClient` (one JSON line per call: role, model,
  settings, prompt, output), :class:`ReplayClient` (regenerate without the model),
  :func:`llm_provenance` (summary for a release)
* :mod:`simulator.ai.agent`    - :class:`LLMAgent`, a model playing an ``ExperimentEnv``
* :mod:`simulator.ai.attacker` - :func:`llm_attack`, a model answering a release's questions
  with the evidence removed, against the blind baseline
* :mod:`simulator.ai.writer`   - :func:`paraphrase_tasks`, model-rephrased questions accepted
  only if they keep every number and option and add no answer
* :mod:`simulator.ai.solver`   - :func:`solve_tasks`, a model answering bundle questions

The rule throughout: a model may propose, phrase, attempt or attack; answers always come from
the simulator. Model-drafted *models* (:func:`simulator.llm.draft_model`) go through the
checker and start at fidelity ``unverified``.
"""
from .agent import LLMAgent  # noqa: F401
from .attacker import AttackReport, evidence_free_view, llm_attack  # noqa: F401
from .clients import (LoggedClient, ReplayClient, ReplayMiss, extract_json, llm_provenance,  # noqa: F401
                      roles_of, write_llm_provenance)
from .solver import solve_tasks  # noqa: F401
from .writer import answer_vocabulary, check_rewrite, paraphrase_tasks  # noqa: F401
