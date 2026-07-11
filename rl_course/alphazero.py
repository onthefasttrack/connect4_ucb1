"""A compact, classroom-sized AlphaZero loop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .env import ConnectK, GameConfig, GameState, encode_state
from .mcts import MCTS
from .network import PolicyValueNet, policy_value_loss


@dataclass(frozen=True)
class TrainingConfig:
    game: GameConfig = GameConfig()
    episodes: int = 8
    simulations: int = 18
    epochs: int = 3
    learning_rate: float = 0.01
    seed: int = 7


@dataclass(frozen=True)
class SelfPlaySample:
    state: GameState
    player: int
    policy: np.ndarray
    outcome: float


class AlphaZeroTrainer:
    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()
        torch.manual_seed(self.config.seed)
        self.game = ConnectK(self.config.game)
        self.network = PolicyValueNet(self.game)
        self.searcher = MCTS(self.game, exploration_constant=1.2)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.learning_rate)

    def _policy_value(self, state: GameState, player: int) -> tuple[np.ndarray, float]:
        return self.network.predict(state, player)

    def self_play(self, episodes: int | None = None) -> list[SelfPlaySample]:
        rng = np.random.default_rng(self.config.seed)
        samples: list[tuple[GameState, int, np.ndarray]] = []
        for episode in range(episodes or self.config.episodes):
            state = self.game.initial_state(current_player=1)
            while not self.game.is_terminal(state):
                result = self.searcher.search(
                    state,
                    player=state.current_player,
                    simulations=self.config.simulations,
                    seed=int(rng.integers(0, 2**31 - 1)) + episode,
                    policy_value_fn=self._policy_value,
                    temperature=1.0 if len(samples) < 10 else 0.5,
                )
                samples.append((state, state.current_player, result.policy.copy()))
                action = int(rng.choice(len(result.policy), p=result.policy))
                if action not in self.game.legal_actions(state):
                    action = result.selected_action
                state = self.game.step(state, action)
            winner = self.game.winner(state)
            return_samples: list[SelfPlaySample] = []
            for old_state, player, policy in samples:
                outcome = 0.0 if winner is None else (1.0 if winner == player else -1.0)
                return_samples.append(SelfPlaySample(old_state, player, policy, outcome))
            if episode == 0:
                all_samples = return_samples
            else:
                all_samples.extend(return_samples)
            samples = []
        return all_samples if "all_samples" in locals() else []

    def train(self, samples: list[SelfPlaySample] | None = None) -> dict[str, list[float]]:
        samples = samples or self.self_play()
        if not samples:
            return {"loss": [], "value_loss": [], "policy_loss": []}
        inputs = torch.from_numpy(np.stack([encode_state(s.state, s.player) for s in samples]))
        policies = torch.from_numpy(np.stack([s.policy for s in samples]).astype(np.float32))
        values = torch.tensor([s.outcome for s in samples], dtype=torch.float32)
        history = {"loss": [], "value_loss": [], "policy_loss": []}
        for _ in range(self.config.epochs):
            logits, predicted = self.network(inputs)
            loss = policy_value_loss(logits, predicted, policies, values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            policy_loss = -(policies * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            value_loss = torch.nn.functional.mse_loss(predicted, values)
            history["loss"].append(float(loss.item()))
            history["policy_loss"].append(float(policy_loss.item()))
            history["value_loss"].append(float(value_loss.item()))
        return history

    def choose_action(self, state: GameState, simulations: int | None = None, seed: int = 7) -> int:
        result = self.searcher.search(
            state,
            player=state.current_player,
            simulations=simulations or self.config.simulations,
            seed=seed,
            policy_value_fn=self._policy_value,
            temperature=0,
        )
        return result.selected_action

    def evaluate(self, games: int = 20, seed: int = 7) -> float:
        wins = 0
        for game_ix in range(games):
            state = self.game.initial_state()
            rng = np.random.default_rng(seed + game_ix)
            while not self.game.is_terminal(state):
                if state.current_player == 1:
                    action = self.choose_action(state, seed=seed + game_ix)
                else:
                    action = int(rng.choice(self.game.legal_actions(state)))
                state = self.game.step(state, action)
            wins += int(self.game.winner(state) == 1)
        return wins / max(games, 1)
