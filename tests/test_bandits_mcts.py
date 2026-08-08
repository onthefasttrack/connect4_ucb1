import numpy as np

from rl_course.bandits import BanditSession, BanditStats, run_ucb_experiment, ucb1_score
from rl_course.env import ConnectK, GameConfig
from rl_course.mcts import MCTS, TreeNode


def test_unvisited_ucb_arm_is_optimistic():
    assert np.isinf(ucb1_score(0.0, 0, 3))
    stats = BanditStats.empty(3)
    stats.update(0, 1.0)
    assert stats.select() == 1


def test_bandit_experiment_is_reproducible():
    left = run_ucb_experiment(seed=4)
    right = run_ucb_experiment(seed=4)
    assert left.pulls == right.pulls
    assert left.pull_values == right.pull_values
    np.testing.assert_allclose(left.rewards, right.rewards)
    np.testing.assert_allclose(left.true_means, right.true_means)
    assert len(left.selection_scores) == 40
    assert np.isinf(left.selection_scores[0]).all()


def test_bandit_session_has_no_fixed_attempt_limit_and_is_seeded():
    left = BanditSession(seed=12)
    right = BanditSession(seed=12)
    left.add_attempts(137)
    right.add_attempts(137)
    assert left.stats.total_visits == 137
    assert left.pulls == right.pulls
    assert left.pull_values == right.pull_values


def test_bandit_session_uses_student_selected_true_means():
    means = (1.0, 2.5, -0.5, 9.0)
    session = BanditSession(seed=12, true_means=means, exploration_constant=1.25)
    np.testing.assert_allclose(session.true_means, means)
    assert session.exploration_constant == 1.25


def test_mcts_returns_policy_and_visits():
    game = ConnectK(GameConfig(rows=4, cols=4, target=3))
    result = MCTS(game).search(game.initial_state(), simulations=20, seed=3)
    # The first simulation expands the root; subsequent simulations visit a
    # child and therefore contribute to the root action counts.
    assert result.visit_counts.sum() == 19
    assert np.isclose(result.policy.sum(), 1.0)
    assert result.selected_action in game.legal_actions(game.initial_state())
    assert len(result.simulation_records) == 20
    assert result.simulation_records[0].selection_actions == ()
    assert result.simulation_records[0].credited_root_action is None
    assert result.simulation_records[-1].root_visits_after == 20


def test_mcts_negamax_selection_uses_the_parent_perspective():
    game = ConnectK(GameConfig(rows=4, cols=4, target=3))
    root_state = game.initial_state()
    root = TreeNode(root_state, player=1, visits=10)
    first = TreeNode(game.step(root_state, 0), player=2, parent=root, action_from_parent=0, visits=5, value_sum=3.5)
    second = TreeNode(game.step(root_state, 1), player=2, parent=root, action_from_parent=1, visits=5, value_sum=-3.5)
    root.children = {0: first, 1: second}

    # Values at the children belong to player 2. The parent is player 1, so
    # it should prefer the child that is worst for player 2.
    assert MCTS(game)._select_child(root, network_guided=False) is second
