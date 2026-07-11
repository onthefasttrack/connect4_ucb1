"""Upper Confidence Bound (UCB1) helpers used in the first experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def ucb1_score(
    mean: float,
    visits: int,
    total_visits: int,
    exploration_constant: float = 1.414,
) -> float:
    """Compute UCB1; an unseen arm gets an infinite optimism bonus."""

    if visits <= 0:
        return float("inf")
    if total_visits <= 0:
        raise ValueError("total_visits must be positive once an arm has visits")
    if exploration_constant < 0:
        raise ValueError("exploration_constant must be non-negative")
    return float(mean + exploration_constant * np.sqrt(np.log(total_visits) / visits))


@dataclass
class BanditStats:
    counts: np.ndarray
    reward_sums: np.ndarray

    @classmethod
    def empty(cls, arms: int) -> "BanditStats":
        return cls(np.zeros(arms, dtype=np.int64), np.zeros(arms, dtype=np.float64))

    @property
    def total_visits(self) -> int:
        return int(self.counts.sum())

    @property
    def means(self) -> np.ndarray:
        return np.divide(self.reward_sums, self.counts, out=np.zeros_like(self.reward_sums), where=self.counts > 0)

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        self.reward_sums[arm] += reward

    def scores(self, exploration_constant: float = 1.414) -> np.ndarray:
        total = max(self.total_visits, 1)
        return np.array(
            [ucb1_score(float(mean), int(visits), total, exploration_constant) for mean, visits in zip(self.means, self.counts)],
            dtype=np.float64,
        )

    def select(self, exploration_constant: float = 1.414, legal: np.ndarray | None = None) -> int:
        scores = self.scores(exploration_constant)
        if legal is not None:
            scores = np.where(legal, scores, -np.inf)
        return int(np.argmax(scores))


class BanditSession:
    """An extendable, seeded UCB experiment with no fixed attempt horizon."""

    def __init__(self, seed: int = 7, arms: int = 4, exploration_constant: float = 0.8) -> None:
        if arms < 1:
            raise ValueError("arms must be positive")
        self.seed = seed
        self.exploration_constant = exploration_constant
        self.rng = np.random.default_rng(seed)
        self.true_means = np.linspace(3.0, 7.0, arms)
        self.stats = BanditStats.empty(arms)
        self.pulls: list[int] = []
        self.pull_values: list[float] = []
        self.selection_scores: list[np.ndarray] = []

    def add_attempts(self, attempts: int = 1) -> None:
        """Run any positive number of sequential UCB selections."""

        if attempts < 1:
            raise ValueError("attempts must be at least one")
        for _ in range(attempts):
            scores = self.stats.scores(self.exploration_constant)
            arm = int(np.argmax(scores))
            reward = float(self.rng.normal(self.true_means[arm], 1.0))
            self.stats.update(arm, reward)
            self.pulls.append(arm)
            self.pull_values.append(reward)
            self.selection_scores.append(scores.copy())

    def snapshot(self) -> "BanditExperiment":
        return BanditExperiment(
            rewards=np.asarray(self.pull_values, dtype=np.float64),
            true_means=self.true_means.copy(),
            pulls=tuple(self.pulls),
            pull_values=tuple(self.pull_values),
            selection_scores=tuple(self.selection_scores),
            stats=BanditStats(self.stats.counts.copy(), self.stats.reward_sums.copy()),
        )


@dataclass(frozen=True)
class BanditExperiment:
    rewards: np.ndarray
    true_means: np.ndarray
    pulls: tuple[int, ...]
    pull_values: tuple[float, ...]
    selection_scores: tuple[np.ndarray, ...]
    stats: BanditStats


def run_ucb_experiment(seed: int = 7, arms: int = 4, horizon: int = 40) -> BanditExperiment:
    """Generate a reproducible finite snapshot for tests and scripted demos."""

    session = BanditSession(seed=seed, arms=arms)
    session.add_attempts(horizon)
    return session.snapshot()
