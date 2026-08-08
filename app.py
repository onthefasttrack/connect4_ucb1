from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from rl_course.alphazero import AlphaZeroTrainer, TrainingConfig
from rl_course.bandits import BanditSession, BanditStats
from rl_course.content import LESSONS
from rl_course.env import ConnectK, GameConfig, GameState
from rl_course.mcts import MCTS


st.set_page_config(page_title="RL Course Lab", page_icon="◒", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Newsreader:opsz,wght@6..72,400;6..72,600&display=swap');
    :root { --paper:#f7f3ec; --ink:#242322; --muted:#756f67; --line:#ded6ca; --red:#b94a45; --blue:#32688b; --gold:#c38d3d; }
    .stApp { background: var(--paper); color: var(--ink); }
    .block-container { max-width: 1180px; padding-top: 2.5rem; padding-bottom: 4rem; }
    h1,h2,h3 { font-family: 'Newsreader', Georgia, serif !important; letter-spacing: -.02em; color:var(--ink); }
    p,li,label,.stMarkdown { color:var(--ink); }
    code, pre, .stCode { font-family:'DM Mono', monospace !important; }
    [data-testid="stSidebar"] { background:#eee8df; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] h2 { font-family:'Newsreader', Georgia, serif !important; font-size:1.35rem; }
    .eyebrow { color:var(--red); font:500 .72rem 'DM Mono',monospace; letter-spacing:.12em; margin-bottom:.5rem; }
    .hero { border-bottom:1px solid var(--line); padding: 1rem 0 2rem; margin-bottom:2rem; }
    .hero h1 { font-size:clamp(2.5rem, 6vw, 5.4rem); line-height:.96; max-width:720px; margin:.1rem 0 1rem; }
    .hero p { max-width:650px; color:var(--muted); font-size:1.1rem; line-height:1.55; }
    .chapter-note { color:var(--muted); font: .8rem 'DM Mono', monospace; }
    .callout { border-left:3px solid var(--red); padding:.7rem 1rem; background:#efe8de; margin:1rem 0 1.5rem; }
    .board-cell { height:55px; display:flex; align-items:center; justify-content:center; font-size:2.2rem; border-bottom:1px solid #c9c0b4; }
    .board-head { text-align:center; color:var(--muted); font: .7rem 'DM Mono',monospace; padding-bottom:.3rem; }
    .metric-line { border-top:1px solid var(--line); padding:.65rem 0; display:flex; justify-content:space-between; }
    .metric-line span:first-child { color:var(--muted); font: .74rem 'DM Mono',monospace; }
    .metric-line span:last-child { font-weight:600; }
    div[data-testid="stButton"] > button { border-radius:4px; border:1px solid #c9c0b4; background:#fbf8f2; color:var(--ink); }
    div[data-testid="stButton"] > button:hover { border-color:var(--red); color:var(--red); }
    </style>
    """,
    unsafe_allow_html=True,
)


CONFIG = GameConfig(rows=6, cols=7, target=4, gravity=True)
GAME = ConnectK(CONFIG)
CHAPTER_SLUGS = (
    "predictions-to-decisions",
    "multi-armed-bandits",
    "ucb1-optimism",
    "games-are-environments",
    "monte-carlo-tree-search",
    "policy-value-networks",
    "alphazero-loop",
    "play-against-agent",
)


def board_view(state: GameState, legal: tuple[int, ...] = ()) -> None:
    """Render a board as a quiet grid of Streamlit columns."""

    headers = st.columns(CONFIG.cols, gap="small")
    for col, header in enumerate(headers):
        with header:
            st.markdown(f"<div class='board-head'>A{col}</div>", unsafe_allow_html=True)
    for row, line in enumerate(state.board):
        columns = st.columns(CONFIG.cols, gap="small")
        for col, value in enumerate(line):
            symbol = "⬜" if value == 0 else ("🔴" if value == 1 else "🔵")
            with columns[col]:
                st.markdown(f"<div class='board-cell'>{symbol}</div>", unsafe_allow_html=True)


def board_buttons(state: GameState, key_prefix: str, on_action) -> None:
    columns = st.columns(CONFIG.cols, gap="small")
    legal = set(GAME.legal_actions(state))
    for action, column in enumerate(columns):
        with column:
            if st.button(
                f"↓ {action}",
                key=f"{key_prefix}-{action}",
                disabled=action not in legal,
                width="stretch",
            ):
                on_action(action)
                st.rerun()


def metric(label: str, value: str) -> None:
    st.markdown(f"<div class='metric-line'><span>{label}</span><span>{value}</span></div>", unsafe_allow_html=True)


def ucb_chart(stats: BanditStats, exploration_constant: float = 0.8) -> None:
    scores = stats.scores(exploration_constant)
    means = stats.means
    finite_scores = scores[np.isfinite(scores)]
    unseen_marker = (max(float(finite_scores.max()), float(means.max()), 0.0) + 1.0) if finite_scores.size else 1.0
    chart_scores = np.where(
        np.isfinite(scores),
        scores,
        unseen_marker,
    )
    fig = go.Figure()
    fig.add_bar(x=[f"Arm {i}" for i in range(len(means))], y=means, name="average reward", marker_color="#32688b")
    fig.add_scatter(x=[f"Arm {i}" for i in range(len(scores))], y=chart_scores, mode="markers+text", text=[f"{x:.2f}" if np.isfinite(x) else "∞" for x in scores], textposition="top center", name="UCB score", marker=dict(size=12, color="#b94a45"))
    fig.update_layout(height=330, margin=dict(l=10,r=10,t=30,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def ucb_calculation_rows(stats: BanditStats, exploration_constant: float) -> list[dict[str, str]]:
    """Format every value in the next UCB decision for the teaching table."""

    total = max(stats.total_visits, 1)
    rows: list[dict[str, str]] = []
    for arm, (visits, reward_sum, mean, score) in enumerate(
        zip(stats.counts, stats.reward_sums, stats.means, stats.scores(exploration_constant))
    ):
        if visits == 0:
            rows.append(
                {
                    "Arm": f"A{arm}",
                    "nᵢ": "0",
                    "Σ rewards": "0.00",
                    "mean R̄ᵢ": "not observed",
                    "ln(N)": "not needed while unvisited",
                    "exploration term": "unvisited arm",
                    "UCBᵢ": "∞",
                    "full calculation": "UCBᵢ = +∞ until this arm is tried",
                }
            )
            continue
        log_total = float(np.log(total))
        root_term = float(np.sqrt(log_total / visits))
        bonus = exploration_constant * root_term
        rows.append(
            {
                "Arm": f"A{arm}",
                "nᵢ": str(int(visits)),
                "Σ rewards": f"{reward_sum:.2f}",
                "mean R̄ᵢ": f"{mean:.3f}",
                "ln(N)": f"ln({total}) = {log_total:.4f}",
                "exploration term": f"{exploration_constant:.2f} × √({log_total:.4f}/{int(visits)}) = {bonus:.3f}",
                "UCBᵢ": f"{score:.3f}",
                "full calculation": f"{mean:.3f} + {exploration_constant:.2f} × √(ln({total})/{int(visits)}) = {score:.3f}",
            }
        )
    return rows


def render_intro(lesson) -> None:
    st.markdown("<div class='hero'><div class='eyebrow'>RL COURSE LAB / A PYTHON-FIRST FIELD GUIDE</div><h1>Learn to make machines choose.</h1><p>Eight small experiments take you from familiar prediction models to a self-playing game agent. Every idea earns its place by changing what you can see on the board.</p></div>", unsafe_allow_html=True)
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        st.markdown("### The through-line")
        st.write("A reward is not a label. It is a delayed clue. We will build the machinery that turns those clues into better decisions.")
        st.markdown("<div class='callout'><b>Today’s question</b><br>How can an agent improve its next move when the answer is not written in the training data?</div>", unsafe_allow_html=True)
    with right:
        state = st.session_state.demo_state
        board_view(state)
        st.caption("A small board is our shared language for states, actions, rewards, and search.")
    st.markdown("### Start with an experiment")
    st.write("Use the chapter list on the left. Each stop has one idea, one Python excerpt, and one moment where you predict what the agent will do.")


def render_lesson_text(lesson) -> None:
    st.markdown(f"<div class='eyebrow'>{lesson.eyebrow}</div>", unsafe_allow_html=True)
    st.header(lesson.title)
    st.write(lesson.body)
    with st.expander("Read the Python idea", expanded=True):
        st.code(lesson.code, language="python")


def render_bandit() -> None:
    st.markdown("### Choose the experiment parameters")
    st.caption("Set the expected rewards used to generate pulls and UCB’s exploration constant c. UCB does not see the expected rewards; changing any setting starts a fresh experiment.")
    parameter_input, *mean_inputs = st.columns([0.85, 1, 1, 1, 1])
    with parameter_input:
        exploration_constant = float(
            st.number_input(
                "Exploration c",
                min_value=0.0,
                value=0.8,
                step=0.1,
                key="bandit_exploration_constant",
            )
        )
    selected_means: list[float] = []
    for arm, column in enumerate(mean_inputs):
        with column:
            selected_means.append(
                float(
                    st.number_input(
                        f"A{arm} expected reward",
                        value=float(np.linspace(3.0, 7.0, 4)[arm]),
                        step=0.25,
                        key=f"bandit_true_mean_{arm}",
                    )
                )
            )
    true_means = tuple(selected_means)
    session = st.session_state.get("bandit_session")
    if (
        session is None
        or session.seed != st.session_state.seed
        or session.exploration_constant != exploration_constant
        or tuple(session.true_means) != true_means
    ):
        session = BanditSession(
            seed=st.session_state.seed,
            arms=4,
            exploration_constant=exploration_constant,
            true_means=true_means,
        )
        st.session_state.bandit_session = session
    st.markdown("### A hidden world, four choices")
    st.write("The evidence below is what the learner has earned. UCB chooses from observed returns and the exploration value c.")
    add_one, add_ten, attempts_input, custom, reset = st.columns([1, 1, 1.25, 0.7, 1])
    with add_one:
        if st.button("Run UCB ×1", key="bandit_add_one", width="stretch"):
            session.add_attempts(1)
    with add_ten:
        if st.button("Run UCB ×10", key="bandit_add_ten", width="stretch"):
            session.add_attempts(10)
    with attempts_input:
        attempts = st.number_input("Attempts to add", min_value=1, value=25, step=1, key="bandit_custom_attempts")
    with custom:
        st.write("")
        if st.button("Add attempts", key="bandit_add_custom"):
            session.add_attempts(int(attempts))
    with reset:
        if st.button("Reset experiment", key="bandit_reset", width="stretch"):
            st.session_state.bandit_session = BanditSession(
                seed=st.session_state.seed,
                arms=4,
                exploration_constant=exploration_constant,
                true_means=true_means,
            )
            session = st.session_state.bandit_session

    stats = session.stats
    history_stats = BanditStats.empty(len(session.true_means))
    pull_rows: list[dict[str, str | int]] = []
    for round_ix, (arm, reward, scores) in enumerate(
        zip(session.pulls, session.pull_values, session.selection_scores), start=1
    ):
        visit_number = int(history_stats.counts[arm]) + 1
        selected_score = scores[arm]
        history_stats.update(arm, reward)
        pull_rows.append(
            {
                "Pull": round_ix,
                "Chosen bandit": f"A{arm}",
                "This bandit's pull": visit_number,
                "Observed return": f"{reward:.3f}",
                "UCB before pull": "∞" if np.isinf(selected_score) else f"{selected_score:.3f}",
            }
        )

    ucb_chart(stats, exploration_constant)
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Observed pulls", expanded=False):
            st.write(" → ".join(f"A{a}" for a in session.pulls) or "No pulls yet. Run UCB to begin the experiment.")
    with col2:
        metric("Total decisions", str(stats.total_visits))
        metric("Most visited", f"A{int(np.argmax(stats.counts))}" if stats.total_visits else "—")
    with st.expander("Returns observed so far", expanded=False):
        st.caption("One row per actual pull: the selected arm, its pull number, the sampled reward, and UCB score before the choice.")
        st.dataframe(pull_rows, hide_index=True, width="stretch")

    st.markdown("### Next UCB decision: every number exposed")
    st.latex(r"\mathrm{UCB}_i = \bar{R}_i + c\sqrt{\frac{\ln(N)}{n_i}}")
    st.caption(
        f"Parameters for the next choice: c = {exploration_constant:.2f}; "
        f"N = total completed pulls = {stats.total_visits}. "
        "nᵢ is that arm’s completed pulls and R̄ᵢ is its observed average return."
    )
    if stats.total_visits == 0:
        st.info("Before the first pull, every bandit is unvisited, so every UCB score is +∞. Run one or more attempts to reveal the finite calculation.")
    st.dataframe(ucb_calculation_rows(stats, exploration_constant), hide_index=True, width="stretch")


def render_ucb_checkpoint() -> None:
    st.markdown("### Choose the experiment parameters")
    st.caption("Set the expected rewards and exploration constant c for this prediction lab. Changing any setting restarts the seeded warm-up evidence.")
    parameter_input, *mean_inputs = st.columns([0.85, 1, 1, 1, 1])
    with parameter_input:
        exploration_constant = float(
            st.number_input(
                "Exploration c",
                min_value=0.0,
                value=0.8,
                step=0.1,
                key="checkpoint_exploration_constant",
            )
        )
    selected_means: list[float] = []
    for arm, column in enumerate(mean_inputs):
        with column:
            selected_means.append(
                float(
                    st.number_input(
                        f"A{arm} expected reward",
                        value=float(np.linspace(3.0, 7.0, 4)[arm]),
                        step=0.25,
                        key=f"checkpoint_true_mean_{arm}",
                    )
                )
            )
    true_means = tuple(selected_means)
    session = st.session_state.get("ucb_checkpoint_session")
    if (
        session is None
        or session.seed != st.session_state.seed
        or session.exploration_constant != exploration_constant
        or tuple(session.true_means) != true_means
    ):
        session = BanditSession(
            seed=st.session_state.seed,
            arms=4,
            exploration_constant=exploration_constant,
            true_means=true_means,
        )
        session.add_attempts(4)  # One observation per arm makes the first guess informative.
        st.session_state.ucb_checkpoint_session = session
        st.session_state.ucb_guess_history = []

    stats = session.stats
    scores = stats.scores(exploration_constant)
    st.markdown("### Current evidence")
    st.write("UCB first sampled every arm once. Use the complete decision state below to predict its next choice, then repeat as many times as you like.")
    evidence_total, evidence_c, evidence_rewards = st.columns(3)
    with evidence_total:
        metric("Total pulls N", str(stats.total_visits))
    with evidence_c:
        metric("Exploration constant c", f"{exploration_constant:.2f}")
    with evidence_rewards:
        metric("Total observed reward", f"{stats.reward_sums.sum():.2f}")
    st.latex(r"\mathrm{UCB}_i = \bar{R}_i + c\sqrt{\frac{\ln(N)}{n_i}}")
    st.caption(
        "For every arm below: nᵢ is the number of pulls, R̄ᵢ is the observed average return, "
        "and the last two columns show the bonus and final UCB1 score."
    )
    st.dataframe(ucb_calculation_rows(stats, exploration_constant), hide_index=True, width="stretch")
    ucb_chart(stats, exploration_constant)

    st.markdown("### Predict the next arm")
    answer = st.radio("Which arm will UCB1 select next?", ["A0", "A1", "A2", "A3"], horizontal=True, key="ucb_answer")
    guess_col, reset_col = st.columns([2, 1])
    with guess_col:
        submit_guess = st.button(
            "Submit guess & run next UCB pull",
            key="submit_ucb_guess",
            type="primary",
            width="stretch",
        )
    with reset_col:
        reset_lab = st.button("Restart lab", key="reset_ucb_lab", width="stretch")

    if reset_lab:
        session = BanditSession(
            seed=st.session_state.seed,
            arms=4,
            exploration_constant=exploration_constant,
            true_means=true_means,
        )
        session.add_attempts(4)
        st.session_state.ucb_checkpoint_session = session
        st.session_state.ucb_guess_history = []
        st.rerun()

    if submit_guess:
        selected_arm = int(np.argmax(scores))
        reward_before_update = len(session.pull_values)
        session.add_attempts(1)
        observed_return = session.pull_values[reward_before_update]
        st.session_state.ucb_guess_history.append(
            {
                "Guess": len(st.session_state.ucb_guess_history) + 1,
                "Your prediction": answer,
                "UCB selected": f"A{selected_arm}",
                "Correct": "✓" if answer == f"A{selected_arm}" else "—",
                "Observed return": f"{observed_return:.3f}",
            }
        )
        st.rerun()

    history = st.session_state.ucb_guess_history
    if history:
        last = history[-1]
        if last["Correct"] == "✓":
            st.success(f"Correct: UCB chose {last['UCB selected']} and observed a return of {last['Observed return']}. The scores above are now updated for your next guess.")
        else:
            st.info(f"UCB chose {last['UCB selected']} and observed a return of {last['Observed return']}. Read the refreshed values above, then guess again.")
        st.markdown("### Guess history")
        st.dataframe(history, hide_index=True, width="stretch")
    else:
        st.info("Trace the row with the largest UCBᵢ, submit a prediction, then use the refreshed scores for your next guess.")


def render_environment() -> None:
    st.markdown("### Step the environment")
    st.write("The board is a state. A column is an action. The environment applies the action and tells us whose turn is next.")
    if st.button("Reset board", key="reset_env"):
        st.session_state.lesson_state = GAME.initial_state()
    state = st.session_state.lesson_state
    board_view(state)
    board_buttons(state, "env", lambda action: st.session_state.__setitem__("lesson_state", GAME.step(state, action)))
    metric("Current player", f"P{state.current_player}")
    metric("Legal actions", ", ".join(map(str, GAME.legal_actions(state))) or "none")
    if GAME.is_terminal(state):
        st.success("This episode is terminal. Reset and try a different sequence.")


def render_mcts() -> None:
    st.markdown("### Search without a neural network")
    simulation_control, reset_control = st.columns([2, 1])
    with simulation_control:
        simulations = st.slider("Simulations", 5, 8000, 30, 5, key="mcts_sims")
    with reset_control:
        st.write("")
        reset_mcts = st.button("Reset search board", key="reset_mcts")
    if reset_mcts:
        st.session_state.mcts_state = GAME.initial_state()
        st.rerun()

    state = st.session_state.mcts_state
    board_view(state)
    result = MCTS(GAME).search(state, simulations=simulations, seed=st.session_state.seed)
    values = result.visit_counts
    fig = go.Figure(go.Bar(x=[f"A{i}" for i in range(CONFIG.cols)], y=values, marker_color=["#b94a45" if i == result.selected_action else "#32688b" for i in range(CONFIG.cols)]))
    fig.update_layout(height=270, title="Root visit counts become the search policy", margin=dict(l=10,r=10,t=45,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    left, right = st.columns(2)
    with left:
        st.markdown("**Last rollout actions**")
        st.code(" → ".join(f"A{a}" for a in result.rollout_trace) or "terminal leaf", language="text")
    with right:
        metric("Suggested action", f"A{result.selected_action}")
        metric("Root value", f"{result.root_value:+.2f}")
    if st.button("Apply the searched move", key="apply_mcts"):
        st.session_state.mcts_state = GAME.step(state, result.selected_action)
        st.rerun()

    st.markdown("### Full MCTS working")
    st.caption(
        "Each row is one simulation. Selection follows the tree, rollout samples actions until the game ends, "
        "then the terminal result is backed up through the visited path. Each node values the result for the player "
        "whose turn it is there, so the value flips sign at every move on the way to the root. A root action receives "
        "a visit whenever the simulation travelled through that child; the first simulation only expands the root."
    )
    simulation_rows = []
    for record in result.simulation_records:
        simulation_rows.append(
            {
                "Simulation": record.simulation,
                "Selection path": " → ".join(f"A{action}" for action in record.selection_actions) or "root expansion",
                "Rollout": " → ".join(f"A{action}" for action in record.rollout_actions) or "none",
                "Full trajectory": " → ".join(f"A{action}" for action in record.trajectory) or "root only",
                "Conclusion": record.outcome,
                "Backed-up value": f"{record.backed_up_value:+.1f}",
                "Root action credited": "—" if record.credited_root_action is None else f"A{record.credited_root_action}",
                "Root visits after": record.root_visits_after,
                "Root value after": f"{record.root_value_after:+.3f}",
            }
        )
    st.dataframe(simulation_rows, hide_index=True, width="stretch")
    st.markdown("**How the simulations become the next policy**")
    st.dataframe(
        [
            {
                "Root action": row["label"],
                "Visits": row["visits"],
                "Mean backed-up value": f"{float(row['value']):+.3f}",
                "Prior": f"{float(row['prior']):.3f}",
                "Selected": "✓" if int(row["action"]) == result.selected_action else "",
            }
            for row in result.tree_rows
        ],
        hide_index=True,
        width="stretch",
    )


def render_network() -> None:
    st.markdown("### Two predictions, one shared trunk")
    st.write("The policy head proposes actions. The value head estimates the position. They are not magic answers; they are learned guesses that search can improve.")
    trainer = st.session_state.get("trainer")
    if trainer is None:
        trainer = AlphaZeroTrainer(TrainingConfig(game=CONFIG, episodes=2, simulations=6, epochs=1, seed=st.session_state.seed))
        st.session_state.trainer = trainer
    logits, value = trainer.network.predict(st.session_state.mcts_state)
    probs = np.exp(logits - logits.max()); probs /= probs.sum()
    fig = go.Figure(go.Bar(x=[f"A{i}" for i in range(CONFIG.cols)], y=probs, marker_color="#32688b"))
    fig.update_layout(height=270, title="Untrained policy: a starting point, not a strategy", margin=dict(l=10,r=10,t=45,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    metric("Value estimate", f"{value:+.2f}")
    st.caption("At this point the network is intentionally untrained. The next chapter shows how self-play creates its labels.")


def render_training() -> None:
    st.markdown("### Let search teach the network")
    st.write("This is a small classroom run, not a leaderboard run. It is short enough to repeat with a different seed and inspect.")
    episodes = st.slider("Self-play games", 2, 12, 4, key="train_episodes")
    simulations = st.slider("Search simulations per move", 4, 24, 8, 4, key="train_simulations")
    if st.button("Run a training loop", type="primary", key="run_training"):
        with st.spinner("Generating self-play and updating the policy/value heads…"):
            trainer = AlphaZeroTrainer(TrainingConfig(game=CONFIG, episodes=episodes, simulations=simulations, epochs=3, seed=st.session_state.seed))
            started = time.perf_counter()
            metrics = trainer.train()
            elapsed = time.perf_counter() - started
            st.session_state.trainer = trainer
            st.session_state.training_metrics = metrics
            st.session_state.training_elapsed = elapsed
            st.session_state.trained = True
    metrics = st.session_state.get("training_metrics")
    if metrics:
        fig = go.Figure()
        for name, values in metrics.items():
            fig.add_scatter(y=values, mode="lines+markers", name=name.replace("_", " ").title())
        fig.update_layout(height=300, title="One compact training trace", margin=dict(l=10,r=10,t=45,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        metric("Runtime", f"{st.session_state.training_elapsed:.1f}s")
        metric("Next step", "Play against the agent")
    else:
        st.info("Run the loop to create self-play samples, policy targets, and value targets.")


def render_play() -> None:
    st.markdown("### Play against the small agent")
    trainer = st.session_state.get("trainer")
    if trainer is None:
        trainer = AlphaZeroTrainer(TrainingConfig(game=CONFIG, episodes=2, simulations=8, epochs=1, seed=st.session_state.seed))
        st.session_state.trainer = trainer
    state = st.session_state.play_state
    board_view(state)
    if GAME.is_terminal(state):
        winner = GAME.winner(state)
        st.success("Draw." if winner is None else f"Player {winner} wins.")
        if st.button("Play again", key="play_again"):
            st.session_state.play_state = GAME.initial_state()
        return
    if state.current_player == 1:
        st.write("Your turn. Choose a column.")
        board_buttons(state, "play-human", lambda action: st.session_state.__setitem__("play_state", GAME.step(state, action)))
    else:
        if st.button("Let the agent think", type="primary", key="agent_move"):
            action = trainer.choose_action(state, simulations=12, seed=st.session_state.seed)
            st.session_state.play_state = GAME.step(state, action)
        st.caption("The agent is searching from the current state; its move is not a lookup table.")


def main() -> None:
    if "seed" not in st.session_state:
        st.session_state.seed = 7
    st.session_state.seed = int(st.sidebar.number_input("Experiment seed", min_value=0, max_value=9999, value=st.session_state.seed, step=1))
    st.sidebar.markdown("### Chapters")
    names = [lesson.title for lesson in LESSONS]
    requested_slug = str(st.query_params.get("chapter", CHAPTER_SLUGS[0]))
    index = CHAPTER_SLUGS.index(requested_slug) if requested_slug in CHAPTER_SLUGS else 0
    if requested_slug not in CHAPTER_SLUGS:
        st.query_params["chapter"] = CHAPTER_SLUGS[0]
    for chapter_index, chapter_name in enumerate(names):
        if st.sidebar.button(
            chapter_name,
            key=f"chapter-{CHAPTER_SLUGS[chapter_index]}",
            type="primary" if chapter_index == index else "secondary",
            width="stretch",
        ):
            st.query_params["chapter"] = CHAPTER_SLUGS[chapter_index]
            st.rerun()
    st.sidebar.progress((index + 1) / len(names))
    st.sidebar.caption(f"Chapter {index + 1} of {len(names)}")
    if "demo_state" not in st.session_state:
        st.session_state.demo_state = GAME.initial_state()
    if "lesson_state" not in st.session_state:
        st.session_state.lesson_state = GAME.initial_state()
    if "mcts_state" not in st.session_state:
        st.session_state.mcts_state = GAME.initial_state()
    if "play_state" not in st.session_state:
        st.session_state.play_state = GAME.initial_state()

    lesson = LESSONS[index]
    if index == 0:
        render_intro(lesson)
    else:
        render_lesson_text(lesson)
        if index == 1:
            render_bandit()
        elif index == 2:
            render_ucb_checkpoint()
        elif index == 3:
            render_environment()
        elif index == 4:
            render_mcts()
        elif index == 5:
            render_network()
        elif index == 6:
            render_training()
        elif index == 7:
            render_play()


if __name__ == "__main__":
    main()
