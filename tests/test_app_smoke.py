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
        app.radio[0].set_value(chapter).run(timeout=30)
        assert not app.exception, chapter


def test_environment_control_updates_without_exception():
    app = AppTest.from_file("app.py").run(timeout=30)
    app.radio[0].set_value("Games are environments").run(timeout=30)
    next(button for button in app.button if button.label == "↓ 0").click().run(timeout=30)
    assert not app.exception
