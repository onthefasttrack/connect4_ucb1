# RL Course Lab — Troubleshooting

## Quick diagnosis

Run these from the repository root:

```bash
python --version
python -m pip show streamlit torch numpy plotly
python -m pytest -q
python -m compileall -q app.py rl_course tests
```

If the first command reports Python below 3.10, create a new virtual
environment with a supported interpreter before debugging the app itself.

## Installation problems

### `ModuleNotFoundError: streamlit`, `torch`, or `plotly`

Install the project into the active interpreter, not a different system Python:

```bash
python -m pip install -e '.[dev]'
python -c "import streamlit, torch, numpy, plotly; print('runtime imports ok')"
```

If `pip` is installing into another interpreter, always use `python -m pip`
after activating `.venv`.

### PyTorch installation fails

PyTorch wheels vary by operating system and Python version. Confirm the Python
version, install the CPU build appropriate for the platform, and retry the
project install. The app does not require CUDA or Metal for its default lesson
settings.

## Streamlit problems

### `PermissionError` while binding the port

Another process or the execution environment may own the default port. Try a
different port:

```bash
python -m streamlit run app.py --server.port=8502
```

In a restricted execution environment, local socket binding may require the
host’s normal terminal or an approved local-server permission.

### The browser shows a blank or stale page

Stop the process with `Ctrl-C`, restart Streamlit, and hard-refresh the browser.
Then check the terminal for the first Python traceback. A blank page is usually
a startup exception or a stale websocket, not an RL algorithm failure.

### The app starts but a chapter fails

Use the Streamlit testing harness to isolate the chapter:

```bash
python - <<'PY'
from streamlit.testing.v1 import AppTest

app = AppTest.from_file("app.py").run(timeout=30)
print("exceptions:", [error.value for error in app.exception])
PY
```

Then select the chapter in `tests/test_app_smoke.py` and rerun the test with
`python -m pytest -q tests/test_app_smoke.py -vv`.

## RL behavior problems

### UCB keeps selecting an unexpected arm

Check the seed and the number of observed pulls. An arm with zero visits has an
infinite UCB score by design, so every arm is explored before exploitation
dominates. Also verify that the exploration constant has not been changed.

### MCTS shows one fewer root visit than simulations

This is expected. The first simulation expands the root; root child visit
counts therefore sum to `simulations - 1`. The root itself has received the
initial visit.

### The agent takes a weak or repetitive move

The agent is intentionally a small teaching model. Try increasing self-play
games and simulations, keep the seed fixed while comparing settings, and
remember that a short run is not expected to solve standard Connect Four.

### Training is slow

Reduce self-play games, simulations per move, and epochs. The recommended
starting point is 2–4 games and 4–8 simulations. Close other CPU-heavy
processes, and avoid running several training sessions in parallel.

### Training metrics are empty

Confirm that the training button completed without an exception and that the
session was not refreshed during the run. The trainer and metrics are stored in
`st.session_state`, so a reload clears them.

## Test failures

### Tests cannot import `rl_course`

Run tests from the repository root and use the project interpreter:

```bash
python -m pytest -q
```

The project config includes the repository on pytest’s Python path. If plain
`pytest -q` reports a missing package such as Plotly, check `which pytest` and
use `.venv/bin/python -m pytest -q`; a global/pyenv pytest can run against a
different site-packages directory than the active project environment.

### A seeded evaluation threshold changes

The baseline test uses a deliberately tiny training run and a fixed seed. If
algorithm changes alter the result, first confirm that the change is intended,
then update the seed/settings and the test explanation together. Do not remove
the reproducibility assertion just to make a flaky run pass.

## Reporting a new issue

Include:

1. operating system and Python version;
2. installation command and dependency versions;
3. exact Streamlit command;
4. selected chapter and experiment seed;
5. full traceback from the terminal;
6. whether `python -m pytest -q` passes.

Do not include secrets or private deployment URLs in an issue report.
