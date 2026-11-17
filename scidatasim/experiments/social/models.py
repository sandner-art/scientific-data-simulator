# scidatasim/experiments/social/models.py
"""Three Mesa models with a planted effect: a hidden parameter that drives an emergent outcome.

* :class:`SegregationModel` - Schelling (1971): residents move when fewer than ``homophily`` of
  their neighbours share their group; mild preferences produce strong segregation.
* :class:`EpidemicTrialModel` - an SIR epidemic among agents moving on a lattice, with a
  randomised trial inside it: a random share of agents is treated, and treatment lowers the
  chance of infection per contact by ``effect`` (the planted causal effect).
* :class:`WealthExchangeModel` - kinetic wealth exchange with saving (Chakraborti and
  Chakrabarti 2000): agents keep ``saving`` of their wealth and split the rest at random with a
  partner. Wealth is conserved; inequality falls as saving rises.

They use Mesa's ``Model``, ``Agent`` and its shuffled activation (``agents.shuffle_do``) and keep
the lattice in NumPy for speed. Each takes ``rng`` (the run's seed) and forwards it to Mesa.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from scidatasim.adapters.mesa import _mesa, init_mesa_model

mesa = _mesa()

_MOORE = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# ---------------------------------------------------------------------------- segregation
class Resident(mesa.Agent):
    def __init__(self, model, group: int, pos):
        super().__init__(model)
        self.group = group
        self.pos_rc = pos

    def similar_share(self) -> Optional[float]:
        grid, (r, c) = self.model.grid, self.pos_rc
        h, w = grid.shape
        same = other = 0
        for dr, dc in _MOORE:
            g = grid[(r + dr) % h, (c + dc) % w]
            if g >= 0:
                same += g == self.group
                other += g != self.group
        return None if same + other == 0 else same / (same + other)

    def step(self):
        share = self.similar_share()
        if share is not None and share < self.model.homophily:
            self.model.move(self)


class SegregationModel(mesa.Model):
    """Schelling's segregation model on a torus (Moore neighbourhood)."""

    def __init__(self, width: int = 30, height: int = 30, density: float = 0.85, minority: float = 0.35,
                 homophily: float = 0.5, rng: Optional[int] = None):
        init_mesa_model(self, rng)
        if not (0 < density < 1 and 0 < minority < 1 and 0 <= homophily <= 1):
            raise ValueError("need 0 < density < 1, 0 < minority < 1 and 0 <= homophily <= 1")
        self.homophily = float(homophily)
        self.np_rng = np.random.default_rng(rng)
        self.grid = np.full((height, width), -1, dtype=int)
        cells = height * width
        n = int(round(density * cells))
        chosen = self.np_rng.choice(cells, size=n, replace=False)
        groups = (self.np_rng.random(n) < minority).astype(int)
        for cell, group in zip(chosen, groups):
            pos = divmod(int(cell), width)
            self.grid[pos] = group
            Resident(self, int(group), pos)
        self.moves = 0

    def move(self, agent: Resident) -> None:
        empty = np.flatnonzero(self.grid.ravel() < 0)
        new = divmod(int(self.np_rng.choice(empty)), self.grid.shape[1])
        self.grid[agent.pos_rc] = -1
        self.grid[new] = agent.group
        agent.pos_rc = new
        self.moves += 1

    def step(self):
        self.moves = 0
        self.agents.shuffle_do("step")
        self.running = self.moves > 0

    # reporters
    @property
    def segregation(self) -> float:
        """Mean share of similar neighbours (agents with at least one neighbour)."""
        shares = [s for s in (a.similar_share() for a in self.agents) if s is not None]
        return float(np.mean(shares)) if shares else float("nan")

    @property
    def unhappy(self) -> int:
        return sum(1 for a in self.agents
                   if (s := a.similar_share()) is not None and s < self.homophily)

    @property
    def grid_array(self) -> np.ndarray:
        return self.grid.astype(float)  # -1 empty, 0 majority, 1 minority


# ---------------------------------------------------------------------------- epidemic trial
S, I, R = 0, 1, 2


class Person(mesa.Agent):
    def __init__(self, model, treated: bool, pos):
        super().__init__(model)
        self.treated = treated
        self.state = S
        self.pos_rc = pos

    def step(self):  # move one cell (Moore) or stay
        h, w = self.model.shape
        dr, dc = self.model.random.choice(_MOORE + [(0, 0)])
        self.pos_rc = ((self.pos_rc[0] + dr) % h, (self.pos_rc[1] + dc) % w)


class EpidemicTrialModel(mesa.Model):
    """SIR among moving agents, with a randomised trial: ``treated_share`` of the agents (chosen at
    random) have their chance of infection per infectious contact multiplied by ``1 - effect``.

    Contacts are the infectious agents in the same cell and the eight around it; a susceptible
    agent with k such contacts is infected with probability 1 - (1 - beta (1 - effect if treated)) ^ k.
    Infectious agents recover with probability ``gamma`` per step.
    """

    def __init__(self, n_agents: int = 400, width: int = 20, height: int = 20, beta: float = 0.04,
                 gamma: float = 0.1, initial_infected: int = 5, treated_share: float = 0.5, effect: float = 0.4,
                 rng: Optional[int] = None):
        init_mesa_model(self, rng)
        if not (0 < beta <= 1 and 0 < gamma <= 1 and 0 <= effect <= 1 and 0 <= treated_share <= 1):
            raise ValueError("need 0 < beta <= 1, 0 < gamma <= 1, 0 <= effect <= 1, 0 <= treated_share <= 1")
        if not 0 < initial_infected < n_agents:
            raise ValueError("initial_infected must be between 1 and n_agents - 1")
        self.beta, self.gamma, self.effect = float(beta), float(gamma), float(effect)
        self.shape = (int(height), int(width))
        self.np_rng = np.random.default_rng(rng)
        treated = np.zeros(n_agents, bool)
        treated[self.np_rng.permutation(n_agents)[: int(round(treated_share * n_agents))]] = True  # randomised
        cells = self.np_rng.integers(0, height * width, size=n_agents)
        self.people = [Person(self, bool(t), divmod(int(c), width)) for t, c in zip(treated, cells)]
        for p in self.np_rng.choice(self.people, size=initial_infected, replace=False):
            p.state = I
        self.new_treated = self.new_control = 0

    def _infectious_counts(self) -> np.ndarray:
        counts = np.zeros(self.shape, int)
        for p in self.people:
            if p.state == I:
                counts[p.pos_rc] += 1
        near = np.zeros_like(counts)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                near += np.roll(np.roll(counts, dr, axis=0), dc, axis=1)
        return near

    def step(self):
        self.agents.shuffle_do("step")
        near = self._infectious_counts()
        self.new_treated = self.new_control = 0
        recover = [p for p in self.people if p.state == I and self.np_rng.random() < self.gamma]
        for p in self.people:
            if p.state != S:
                continue
            k = near[p.pos_rc]
            if k == 0:
                continue
            per_contact = self.beta * ((1 - self.effect) if p.treated else 1.0)
            if self.np_rng.random() < 1 - (1 - per_contact) ** k:
                p.state = I
                if p.treated:
                    self.new_treated += 1
                else:
                    self.new_control += 1
        for p in recover:
            p.state = R
        self.running = any(p.state == I for p in self.people)

    # reporters
    def _count(self, state, treated=None) -> int:
        return sum(1 for p in self.people if p.state == state and (treated is None or p.treated == treated))

    @property
    def susceptible(self):
        return self._count(S)

    @property
    def infectious(self):
        return self._count(I)

    @property
    def recovered(self):
        return self._count(R)

    @property
    def susceptible_treated(self):
        return self._count(S, True)

    @property
    def susceptible_control(self):
        return self._count(S, False)


# ---------------------------------------------------------------------------- wealth exchange
class Trader(mesa.Agent):
    def __init__(self, model, wealth: float):
        super().__init__(model)
        self.wealth = wealth


class WealthExchangeModel(mesa.Model):
    """Kinetic wealth exchange with saving propensity ``saving`` (Chakraborti-Chakrabarti).

    Each step every agent is paired at random; a pair keeps ``saving`` of their wealth and splits
    the rest at a uniformly random ratio. Total wealth is conserved exactly.
    """

    def __init__(self, n_agents: int = 500, saving: float = 0.3, rng: Optional[int] = None):
        init_mesa_model(self, rng)
        if not 0 <= saving < 1:
            raise ValueError("saving must be in [0, 1)")
        self.saving = float(saving)
        self.np_rng = np.random.default_rng(rng)
        self.traders = [Trader(self, 1.0) for _ in range(int(n_agents))]

    def step(self):
        order = self.np_rng.permutation(len(self.traders))
        for a, b in zip(order[0::2], order[1::2]):
            ta, tb = self.traders[a], self.traders[b]
            pot = (1 - self.saving) * (ta.wealth + tb.wealth)
            eps = self.np_rng.random()
            ta.wealth = self.saving * ta.wealth + eps * pot
            tb.wealth = self.saving * tb.wealth + (1 - eps) * pot

    # reporters
    @property
    def wealth(self) -> np.ndarray:
        return np.array([t.wealth for t in self.traders])

    @property
    def gini(self) -> float:
        w = np.sort(self.wealth)
        n = w.size
        return float((2 * np.arange(1, n + 1) - n - 1) @ w / (n * w.sum()))

    @property
    def total_wealth(self) -> float:
        return float(self.wealth.sum())

    @property
    def top10_share(self) -> float:
        w = np.sort(self.wealth)[::-1]
        return float(w[: max(1, w.size // 10)].sum() / w.sum())
