# scidatasim/diagnostics/__init__.py
"""Diagnostics for generated benchmarks and for the agents evaluated on them.

The simulator knows every hidden value and can regenerate any run, so it can serve as a
*testbed*: leak channels can be switched on and off, unseen twins made on demand, and agent
behaviour checked against the truth step by step.

* :mod:`scidatasim.audit`                     - leak audit of public files (exact leaks)
* :mod:`scidatasim.diagnostics.shortcuts`     - answers predictable without the data (statistical leaks),
                                               and scorers a polynomial can satisfy
* :mod:`scidatasim.diagnostics.contamination` - fresh twins, canaries, score-gap tests
* :mod:`scidatasim.diagnostics.episodes`      - agent-fallacy flags for interactive episodes
* :mod:`scidatasim.diagnostics.quality`       - balance, diversity, duplicates, solvability
* :mod:`scidatasim.diagnostics.stats`         - permutation, bootstrap and binomial tests
"""
from .contamination import (add_canary, canary_probe, contamination_gap, make_canary, make_twin,  # noqa: F401
                            pair_tasks)
from .episodes import EpisodeLog, diagnose_episode, law_alternatives, run_episodes, summarize  # noqa: F401
from .quality import accuracy_by, card_markdown, quality_card  # noqa: F401
from .shortcuts import (approximator_baseline, blind_baseline, feature_shortcuts, polynomial_formula,  # noqa: F401
                        public_features)
from .stats import gap_test  # noqa: F401
