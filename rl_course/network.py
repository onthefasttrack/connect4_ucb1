"""Small policy/value model used only after the tabular RL foundations."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .env import ConnectK, GameState, encode_state


class PolicyValueNet(nn.Module):
    def __init__(self, game: ConnectK, hidden: int = 64) -> None:
        super().__init__()
        self.game = game
        features = 3 * game.config.rows * game.config.cols
        actions = game.config.cols if game.config.gravity else game.config.rows * game.config.cols
        self.trunk = nn.Sequential(nn.Flatten(), nn.Linear(features, hidden), nn.Tanh())
        self.policy_head = nn.Linear(hidden, actions)
        self.value_head = nn.Sequential(nn.Linear(hidden, 1), nn.Tanh())

    def forward(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(encoded)
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)

    def predict(self, state: GameState, player: int | None = None) -> tuple[np.ndarray, float]:
        player = state.current_player if player is None else player
        with torch.no_grad():
            inputs = torch.from_numpy(encode_state(state, player)).unsqueeze(0)
            logits, value = self(inputs)
        return logits.squeeze(0).cpu().numpy(), float(value.item())


def policy_value_loss(
    policy_logits: torch.Tensor,
    values: torch.Tensor,
    target_policy: torch.Tensor,
    target_values: torch.Tensor,
) -> torch.Tensor:
    """The two supervised targets AlphaZero creates from self-play."""

    policy_loss = -(target_policy * torch.log_softmax(policy_logits, dim=-1)).sum(dim=-1).mean()
    value_loss = nn.functional.mse_loss(values, target_values)
    return policy_loss + value_loss
