import pytest

from rl_course.env import ConnectK, GameConfig


def test_gravity_places_piece_at_bottom():
    game = ConnectK(GameConfig(rows=4, cols=4, target=3))
    state = game.step(game.initial_state(), 1)
    assert state.board[-1][1] == 1


def test_horizontal_winner_and_reward_perspective():
    game = ConnectK(GameConfig(rows=4, cols=4, target=3))
    state = game.initial_state()
    for action, player in [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1)]:
        state = game.step(state, action, player)
    assert game.winner(state) == 1
    assert game.reward(state, 1) == 1
    assert game.reward(state, 2) == -1


def test_invalid_action_is_rejected():
    game = ConnectK(GameConfig(rows=4, cols=4, target=3))
    state = game.initial_state()
    with pytest.raises(ValueError):
        game.step(state, 9)


def test_free_placement_legal_actions():
    game = ConnectK(GameConfig(rows=2, cols=2, target=2, gravity=False))
    assert game.legal_actions(game.initial_state()) == (0, 1, 2, 3)
