"""Student-facing lesson copy kept separate from the app layout."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Lesson:
    title: str
    eyebrow: str
    body: str
    code: str
    checkpoint: str | None = None
    answer: str | None = None


LESSONS = [
    Lesson("From predictions to decisions", "01 / ORIENTATION", "A classifier predicts a label. An agent must choose an action, observe what happened, and learn from consequences. The missing ingredient is a loop: state → action → reward → next state.", "state = observation\naction = policy(state)\nnext_state, reward = environment.step(action)", "What is the feedback signal an agent receives after an action?", "A reward — a scalar signal that says how useful the outcome was."),
    Lesson("Multi-armed bandits", "02 / EXPLORE", "A bandit removes the changing state. Each arm is an action with an unknown reward distribution. The learner must spend some trials exploring before exploiting its best estimate.", "for _ in range(horizon):\n    arm = choose_arm(stats)\n    reward = hidden_rewards[arm]\n    stats.update(arm, reward)", "Why is always choosing the arm with the highest current average risky?", "Early averages are noisy. An apparently weak arm may simply be under-sampled."),
    Lesson("UCB1: optimism under uncertainty", "03 / UCB1", "UCB1 adds an uncertainty bonus to the empirical mean. An unvisited arm gets infinite optimism, guaranteeing that every option is tried at least once.", "score = mean + c * sqrt(log(total_visits) / visits)\narm = argmax(score)", "Which arm should be selected when one option has never been visited?", "The unvisited arm: its score is +∞, so the algorithm explores it."),
    Lesson("Games are environments", "04 / ENVIRONMENT", "Connect-K makes the RL vocabulary concrete. A state is the board, an action is a legal column, and a reward arrives only when a game ends.", "state = game.initial_state()\nlegal = game.legal_actions(state)\nnext_state = game.step(state, action)", "If the board is not terminal yet, what is the immediate reward?", "Zero in this sparse-reward game; the useful signal arrives at the terminal state."),
    Lesson("Monte Carlo Tree Search", "05 / MCTS", "MCTS turns many cheap imagined games into a policy. It repeatedly selects a promising branch, expands a new action, rolls out to an ending, and backs the result up the tree.", "for _ in range(simulations):\n    leaf = select_and_expand(root)\n    outcome = rollout(leaf)\n    backup(leaf, outcome)", "After many simulations, what does a large visit count mean?", "The search policy considers that action promising under the current evidence."),
    Lesson("Policy and value networks", "06 / FUNCTION APPROXIMATION", "A neural network can replace expensive random rollouts with two predictions: which actions look promising (policy) and how good the position is (value).", "policy_logits, value = network(encoded_board)\nloss = policy_loss + value_loss", "What two labels does self-play create for training?", "A visit-count policy from search and an eventual game outcome."),
    Lesson("The AlphaZero loop", "07 / SELF-PLAY", "AlphaZero lets the network improve the search, then lets search create better training targets. The loop is self-play → MCTS targets → gradient update → stronger play.", "games = trainer.self_play()\nmetrics = trainer.train(games)\nagent = trainer.choose_action(state)", "Why does self-play need search instead of sampling directly from the network?", "Search turns a rough prediction into a stronger, look-ahead policy that becomes a useful target."),
    Lesson("Play against the agent", "08 / LAB", "Use the small trained agent as a microscope. Change the seed, inspect its search, and notice that an agent can be competent without being perfect or omniscient.", "while not game.is_terminal(state):\n    action = agent.choose_action(state)\n    state = game.step(state, action)", None, None),
]
