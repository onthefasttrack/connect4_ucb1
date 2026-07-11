# RL Course Lab — Product Specification

## 1. Purpose

RL Course Lab is a Python-first interactive lesson for students who already
understand machine learning but have not yet learned reinforcement learning.
It uses a small Connect-K game as a common visual language for states, actions,
rewards, search, and function approximation.

The product is a read-only teaching application. Students interact with
experiments and make predictions, but they do not edit code inside the app.
The implementation remains ordinary Python in the repository so an instructor
can teach from, modify, or extend it.

## 2. Audience and learning outcomes

Students should be comfortable with Python, arrays, functions, and the basic
idea of fitting a model from data. After completing the lesson, a student
should be able to:

- distinguish a prediction target from a reward signal;
- describe the state/action/reward/next-state loop;
- explain exploration versus exploitation;
- calculate and interpret the UCB1 score;
- identify a game environment’s legal actions and terminal rewards;
- explain selection, expansion, rollout, and backup in MCTS;
- distinguish a policy prediction from a value prediction;
- describe how AlphaZero creates policy and value targets through self-play;
- inspect and play against a small trained policy/value agent.

## 3. User journey

The sidebar presents eight chapters and a progress indicator. Each chapter has
one primary teaching job, a short Python excerpt, and an interaction or
prediction checkpoint.

| Chapter | Teaching job | Interactive surface |
| --- | --- | --- |
| 1. From predictions to decisions | Introduce sequential decision loops | Board as a shared visual vocabulary |
| 2. Multi-armed bandits | Establish exploration/exploitation | Pull-by-pull returns, reward chart, and revealed true means |
| 3. UCB1 | Make uncertainty visible | Choose-the-next-arm checkpoint and full parameter calculation |
| 4. Games are environments | Map RL terms to Connect-K | Resettable board and legal-action controls |
| 5. Monte Carlo Tree Search | Show search as imagined experience | Visit-count chart and rollout trace |
| 6. Policy and value networks | Introduce function approximation | Policy probabilities and value estimate |
| 7. The AlphaZero loop | Connect search to training | Seeded self-play and loss curves |
| 8. Play against the agent | Turn concepts into a lab exercise | Human-versus-agent board |

## 4. Core behavior

### 4.1 Game configuration

The default game is a gravity-based 5-row × 6-column board with a 4-in-a-row
target. Players are represented by `1` and `2`, empty cells by `0`, and legal
actions are column indices `0` through `5`. A game returns `+1` for the winner
from the selected perspective, `-1` for the other player, and `0` for a draw or
non-terminal state.

The environment also supports free-placement Connect-K configurations for
experiments and tests, although the app’s controls focus on the gravity game.

### 4.2 Experiment seed

The sidebar seed controls the reproducible bandit sequence, MCTS random
rollouts, and AlphaZero training initialization. Changing the seed is an
intentional experiment action; it does not persist across browser sessions.

### 4.3 Checkpoints

Prediction checkpoints are deliberately low stakes. The app reveals the
reasoning after the student submits a choice; it does not grade or transmit
answers. The expected explanation is shown alongside the result so the student
can compare reasoning, not only correctness.

### 4.4 Bandit evidence and UCB transparency

The bandit chapter records every completed pull: decision number, selected
bandit, that bandit’s visit number, observed return, and the selected arm’s UCB
score immediately before the pull. It also reveals each bandit’s true mean for
teaching comparison; that value is never supplied to UCB.

Students add attempts in batches of one, ten, or a custom positive number. The
experiment has no fixed pull limit and retains every recorded attempt for the
current browser session.

For the next choice, the app displays every UCB parameter per arm: total pulls
`N`, arm pulls `nᵢ`, reward sum, observed mean, `ln(N)`, exploration constant
`c`, exploration bonus, final score, and the substituted formula.

### 4.5 Training

Chapter 7 runs a compact in-process training session. The student selects the
number of self-play games and MCTS simulations per move. The result is held in
the current Streamlit session and is available to the play chapter. Closing or
reloading the session discards the model and metrics.

### 4.6 Play

The student plays as player 1 by selecting a legal column. The agent responds
as player 2 after a search. A reset is available after a terminal result. The
agent is intentionally a small teaching model, not a competitive standard
Connect Four engine.

## 5. Non-functional requirements

- **Python-first:** teaching logic must remain readable Python; UI code should
  not require a separate frontend build.
- **Deterministic:** seeded experiments should reproduce the same results.
- **Local-friendly:** the app should run on a laptop CPU with the documented
  small training settings.
- **Transparent:** visualizations expose averages, visits, values, and policy
  rather than presenting opaque “AI confidence.”
- **Accessible:** controls have labels, keyboard-friendly Streamlit widgets,
  readable contrast, and a fallback text explanation for charts.
- **Session-local:** no database, account, telemetry, or external API is
  required for v1.

## 6. Explicitly out of scope for v1

- editable student code cells or an embedded notebook editor;
- persistent accounts, saved models, classrooms, grades, or analytics;
- competitive 7×6 Connect Four strength;
- distributed or GPU-managed training;
- arbitrary custom board editing in the UI;
- production authentication or multi-tenant isolation.

## 7. Acceptance criteria

The implementation is considered complete when:

1. all eight chapters render without exceptions;
2. a student can change the seed and observe reproducible experiment behavior;
3. UCB, environment, and MCTS controls update without a page-level reset;
4. a compact training run produces policy/value loss metrics;
5. the resulting agent can complete a seeded game against a human;
6. the automated suite covers environment rules, UCB, MCTS, network/training,
   evaluation, and Streamlit chapter rendering.
