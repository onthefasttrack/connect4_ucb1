"""Readable Monte Carlo Tree Search built on the Connect-K environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .env import ConnectK, GameState


PolicyValueFn = Callable[[GameState, int], tuple[np.ndarray, float]]


@dataclass
class TreeNode:
    state: GameState
    player: int
    parent: "TreeNode | None" = None
    action_from_parent: int | None = None
    prior: float = 1.0
    visits: int = 0
    value_sum: float = 0.0
    children: dict[int, "TreeNode"] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(frozen=True)
class SearchResult:
    policy: np.ndarray
    visit_counts: np.ndarray
    child_values: np.ndarray
    selected_action: int
    root_value: float
    tree_rows: tuple[dict[str, float | int | str], ...]
    rollout_trace: tuple[int, ...]
    root: TreeNode = field(compare=False, repr=False)


class MCTS:
    def __init__(self, game: ConnectK, exploration_constant: float = 1.4) -> None:
        self.game = game
        self.exploration_constant = exploration_constant

    def _select_child(self, node: TreeNode, network_guided: bool) -> TreeNode:
        log_parent = np.log(max(node.visits, 1) + 1)

        def score(child: TreeNode) -> float:
            bonus = self.exploration_constant * child.prior * np.sqrt(log_parent / max(child.visits, 1))
            if child.visits == 0:
                return float("inf") if not network_guided else bonus + 1e6
            return child.value + bonus

        return max(node.children.values(), key=lambda child: (score(child), -(child.action_from_parent or 0)))

    def _random_rollout(self, state: GameState, rng: np.random.Generator) -> tuple[float, tuple[int, ...]]:
        trace: list[int] = []
        while not self.game.is_terminal(state):
            action = int(rng.choice(self.game.legal_actions(state)))
            trace.append(action)
            state = self.game.step(state, action)
        return self.game.reward(state, perspective=1), tuple(trace)

    def search(
        self,
        state: GameState,
        player: int | None = None,
        simulations: int = 40,
        seed: int = 7,
        policy_value_fn: PolicyValueFn | None = None,
        temperature: float = 1.0,
    ) -> SearchResult:
        if simulations < 1:
            raise ValueError("simulations must be positive")
        player = state.current_player if player is None else player
        rng = np.random.default_rng(seed)
        root = TreeNode(state=state, player=player)
        last_trace: tuple[int, ...] = ()

        for _ in range(simulations):
            node = root
            path = [node]
            while node.children and not self.game.is_terminal(node.state):
                node = self._select_child(node, network_guided=policy_value_fn is not None)
                path.append(node)

            if self.game.is_terminal(node.state):
                value = self.game.reward(node.state, perspective=player)
                trace = ()
            else:
                legal = self.game.legal_actions(node.state)
                if policy_value_fn is None:
                    priors = np.full(len(legal), 1 / len(legal), dtype=np.float64)
                    value, trace = self._random_rollout(node.state, rng)
                    if player != 1:
                        value = -value
                else:
                    logits, value = policy_value_fn(node.state, node.state.current_player)
                    logits = np.asarray(logits, dtype=np.float64)
                    scaled = logits[list(legal)] - np.max(logits[list(legal)])
                    weights = np.exp(scaled)
                    priors = weights / max(weights.sum(), 1e-12)
                    trace = ()
                    if node.state.current_player != player:
                        value = -value
                for action, prior in zip(legal, priors):
                    child_state = self.game.step(node.state, action)
                    node.children[action] = TreeNode(child_state, child_state.current_player, node, action, float(prior))
                last_trace = tuple(trace)

            for visited in reversed(path):
                visited.visits += 1
                visited.value_sum += float(value)

        counts = np.zeros(self.game.config.cols if self.game.config.gravity else self.game.config.rows * self.game.config.cols, dtype=np.int64)
        values = np.zeros_like(counts, dtype=np.float64)
        for action, child in root.children.items():
            counts[action] = child.visits
            values[action] = child.value
        if temperature <= 0:
            policy = np.zeros_like(values, dtype=np.float64)
            policy[int(np.argmax(counts))] = 1.0
        else:
            softened = counts.astype(np.float64) ** (1.0 / temperature)
            policy = softened / softened.sum() if softened.sum() else softened
        selected = int(np.argmax(counts))
        rows = tuple(
            {"action": action, "visits": int(counts[action]), "value": float(values[action]), "prior": float(child.prior), "label": f"A{action}"}
            for action, child in sorted(root.children.items())
        )
        return SearchResult(policy, counts, values, selected, root.value, rows, last_trace, root)
