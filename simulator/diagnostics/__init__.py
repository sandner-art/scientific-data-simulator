# simulator/diagnostics/__init__.py
"""Diagnostics for generated benchmarks and for the agents evaluated on them.

The simulator knows every hidden value and can regenerate any run, so it can serve as a
*testbed*: leak channels can be switched on and off, unseen twins made on demand, and agent
behaviour checked against the truth step by step.

* :mod:`simulator.audit`                     - leak audit of public files (exact leaks)
* :mod:`simulator.diagnostics.shortcuts`     - answers predictable without the data (statistical leaks)
* :mod:`simulator.diagnostics.contamination` - fresh twins, canaries, score-gap tests
* :mod:`simulator.diagnostics.episodes`      - agent-fallacy flags for interactive episodes
* :mod:`simulator.diagnostics.quality`       - balance, diversity, duplicates, solvability
* :mod:`simulator.diagnostics.stats`         - permutation, bootstrap and binomial tests
"""
from .contamination import (add_canary, canary_probe, contamination_gap, make_canary, make_twin,  # noqa: F401
                            pair_tasks)
from .episodes import EpisodeLog, diagnose_episode, law_alternatives, run_episodes, summarize  # noqa: F401
from .quality import card_markdown, quality_card  # noqa: F401
from .shortcuts import blind_baseline, feature_shortcuts, public_features  # noqa: F401
from .stats import gap_test  # noqa: F401
