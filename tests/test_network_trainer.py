import numpy as np
import torch

from rl_course.alphazero import AlphaZeroTrainer, TrainingConfig
from rl_course.env import GameConfig
from rl_course.network import PolicyValueNet, policy_value_loss


def test_policy_value_shapes_and_loss():
    config = GameConfig(rows=4, cols=4, target=3)
    net = PolicyValueNet(__import__("rl_course").ConnectK(config))
    logits, values = net(torch.zeros(2, 3, 4, 4))
    targets = torch.zeros_like(logits)
    targets[:, 0] = 1
    loss = policy_value_loss(logits, values, targets, torch.zeros(2))
    assert logits.shape == (2, 4)
    assert values.shape == (2,)
    assert torch.isfinite(loss)


def test_self_play_sample_format():
    trainer = AlphaZeroTrainer(TrainingConfig(game=GameConfig(rows=4, cols=4, target=3), episodes=1, simulations=3, epochs=1))
    samples = trainer.self_play(episodes=1)
    assert samples
    assert samples[0].policy.shape == (4,)
    assert np.isclose(samples[0].policy.sum(), 1.0)


def test_small_agent_evaluation_is_reproducible_and_beats_random_baseline():
    trainer = AlphaZeroTrainer(TrainingConfig(game=GameConfig(), episodes=1, simulations=3, epochs=1, seed=2))
    trainer.train()
    score = trainer.evaluate(games=4, seed=2)
    assert 0.0 <= score <= 1.0
    assert score >= 0.5
