# scidatasim/ai/__init__.py
"""Language models in four roles, all through one logged, replayable client.

* :mod:`scidatasim.ai.clients`  - :class:`LoggedClient` (one JSON line per call: role, model,
  settings, prompt, output), :class:`ReplayClient` (regenerate without the model),
  :func:`llm_provenance` (summary for a release)
* :mod:`scidatasim.ai.agent`    - :class:`LLMAgent`, a model playing an ``ExperimentEnv``
* :mod:`scidatasim.ai.attacker` - :func:`llm_attack`, a model answering a release's questions
  with the evidence removed, against the blind baseline
* :mod:`scidatasim.ai.writer`   - :func:`paraphrase_tasks`, model-rephrased questions accepted
  only if they keep every number and option and add no answer
* :mod:`scidatasim.ai.solver`   - :func:`solve_tasks`, a model answering bundle questions
* :mod:`scidatasim.ai.local`    - :class:`OllamaClient`, :class:`OpenAICompatibleClient` (models on
  your own GPU or any OpenAI-compatible server) and :func:`shard` (work split over processes)

The rule throughout: a model may propose, phrase, attempt or attack; answers always come from
the simulator. Model-drafted *models* (:func:`scidatasim.llm.draft_model`) go through the
checker and start at fidelity ``unverified``.
"""
from .agent import LLMAgent  # noqa: F401
from .attacker import AttackReport, evidence_free_view, llm_attack  # noqa: F401
from .clients import (LoggedClient, ReplayClient, ReplayMiss, extract_json, llm_provenance,  # noqa: F401
                      roles_of, write_llm_provenance)
from .local import OllamaClient, OpenAICompatibleClient, parse_shard, shard  # noqa: F401
from .solver import solve_tasks  # noqa: F401
from .writer import answer_vocabulary, check_rewrite, paraphrase_tasks  # noqa: F401
