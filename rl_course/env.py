"""A tiny immutable Connect-K environment.

The state is deliberately represented as tuples so that students can reason
about transitions without hidden mutation.  NumPy is used only for encoding
and for the algorithms that consume batches of states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class GameConfig:
    rows: int = 5
    cols: int = 6
    target: int = 4
    gravity: bool = True

    def __post_init__(self) -> None:
        if min(self.rows, self.cols, self.target) < 1:
            raise ValueError("rows, cols, and target must be positive")
        if self.target > max(self.rows, self.cols):
            raise ValueError("target cannot exceed the board dimensions")


Board = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class GameState:
    board: Board
    current_player: int = 1

    @property
    def rows(self) -> int:
        return len(self.board)

    @property
    def cols(self) -> int:
        return len(self.board[0]) if self.board else 0


class ConnectK:
    """Pure game rules for gravity and free-placement k-in-a-row games."""

    def __init__(self, config: GameConfig | None = None) -> None:
        self.config = config or GameConfig()

    def initial_state(self, current_player: int = 1) -> GameState:
        board = tuple(tuple(0 for _ in range(self.config.cols)) for _ in range(self.config.rows))
        return GameState(board, current_player)

    def board_array(self, state: GameState) -> np.ndarray:
        return np.asarray(state.board, dtype=np.int8)

    def legal_actions(self, state: GameState) -> tuple[int, ...]:
        board = self.board_array(state)
        if self.config.gravity:
            return tuple(c for c in range(self.config.cols) if board[0, c] == 0)
        return tuple(
            r * self.config.cols + c
            for r in range(self.config.rows)
            for c in range(self.config.cols)
            if board[r, c] == 0
        )

    def step(self, state: GameState, action: int, player: int | None = None) -> GameState:
        player = state.current_player if player is None else player
        if player not in (1, 2):
            raise ValueError("player must be 1 or 2")
        if action not in self.legal_actions(state):
            raise ValueError(f"action {action} is not legal")

        board = self.board_array(state).copy()
        if self.config.gravity:
            row = next(r for r in range(self.config.rows - 1, -1, -1) if board[r, action] == 0)
            col = action
        else:
            row, col = divmod(action, self.config.cols)
        board[row, col] = player
        frozen = tuple(tuple(int(value) for value in line) for line in board)
        return GameState(frozen, 3 - player)

    def winner(self, state: GameState) -> int | None:
        board = self.board_array(state)
        directions: Iterable[tuple[int, int]] = ((0, 1), (1, 0), (1, 1), (1, -1))
        for row in range(self.config.rows):
            for col in range(self.config.cols):
                player = int(board[row, col])
                if player == 0:
                    continue
                for dr, dc in directions:
                    end_r = row + (self.config.target - 1) * dr
                    end_c = col + (self.config.target - 1) * dc
                    if not (0 <= end_r < self.config.rows and 0 <= end_c < self.config.cols):
                        continue
                    if all(board[row + i * dr, col + i * dc] == player for i in range(self.config.target)):
                        return player
        return None

    def is_terminal(self, state: GameState) -> bool:
        return self.winner(state) is not None or not self.legal_actions(state)

    def reward(self, state: GameState, perspective: int = 1) -> float:
        winner = self.winner(state)
        if winner is None:
            return 0.0
        return 1.0 if winner == perspective else -1.0

    def render(self, state: GameState) -> str:
        symbols = {0: "·", 1: "●", 2: "○"}
        return "\n".join(" ".join(symbols[cell] for cell in row) for row in state.board)


def encode_state(state: GameState, player: int) -> np.ndarray:
    """Return three simple channels: current player, opponent, and empty."""

    board = np.asarray(state.board, dtype=np.float32)
    return np.stack([board == player, board == (3 - player), board == 0]).astype(np.float32)
