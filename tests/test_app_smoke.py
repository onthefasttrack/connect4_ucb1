from streamlit.testing.v1 import AppTest


CHAPTERS = [
    "From predictions to decisions",
    "Multi-armed bandits",
    "UCB1: optimism under uncertainty",
    "Games are environments",
    "Monte Carlo Tree Search",
    "Policy and value networks",
    "The AlphaZero loop",
    "Play against the agent",
]


def test_every_chapter_renders_without_exceptions():
    for chapter in CHAPTERS:
        app = AppTest.from_file("app.py").run(timeout=30)
        next(button for button in app.button if button.label == chapter).click().run(timeout=30)
        assert not app.exception, chapter


def test_environment_control_updates_without_exception():
    app = AppTest.from_file("app.py").run(timeout=30)
    next(button for button in app.button if button.label == "Games are environments").click().run(timeout=30)
    next(button for button in app.button if button.label == "↓ 0").click().run(timeout=30)
    assert not app.exception
    assert app.session_state["lesson_state"].board[-1][0] == 1


def test_mcts_move_updates_the_board_immediately():
    app = AppTest.from_file("app.py").run(timeout=30)
    next(button for button in app.button if button.label == "Monte Carlo Tree Search").click().run(timeout=30)
    next(button for button in app.button if button.label == "Apply the searched move").click().run(timeout=30)
    assert not app.exception
    assert sum(cell != 0 for row in app.session_state["mcts_state"].board for cell in row) == 1


def test_mcts_reset_returns_to_an_empty_board():
    app = AppTest.from_file("app.py").run(timeout=30)
    next(button for button in app.button if button.label == "Monte Carlo Tree Search").click().run(timeout=30)
    next(button for button in app.button if button.label == "Apply the searched move").click().run(timeout=30)
    next(button for button in app.button if button.label == "Reset search board").click().run(timeout=30)
    assert not app.exception
    assert all(cell == 0 for row in app.session_state["mcts_state"].board for cell in row)


def test_ucb_checkpoint_shows_current_values_before_prediction():
    app = AppTest.from_file("app.py").run(timeout=30)
    next(button for button in app.button if button.label == "UCB1: optimism under uncertainty").click().run(timeout=30)
    assert not app.exception
    assert len(app.dataframe) == 1
    labels = {input.label for input in app.number_input}
    assert {"Exploration c", "A0 expected reward", "A1 expected reward", "A2 expected reward", "A3 expected reward"}.issubset(labels)
    assert any(button.label == "Submit guess & run next UCB pull" for button in app.button)


def test_ucb_checkpoint_allows_repeated_guesses_with_updated_state():
    app = AppTest.from_file("app.py").run(timeout=30)
    next(button for button in app.button if button.label == "UCB1: optimism under uncertainty").click().run(timeout=30)
    next(radio for radio in app.radio if radio.label == "Which arm will UCB1 select next?").set_value("A0")
    next(button for button in app.button if button.label == "Submit guess & run next UCB pull").click().run(timeout=30)
    next(button for button in app.button if button.label == "Submit guess & run next UCB pull").click().run(timeout=30)
    assert not app.exception
    assert len(app.dataframe) == 2
