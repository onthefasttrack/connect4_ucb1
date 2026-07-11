# RL Course Lab — Deployment Guide

## 1. Prerequisites

- Python 3.10 or newer;
- a CPU with enough memory for PyTorch and the small training run;
- outbound access to install Python packages;
- a browser for the Streamlit UI.

GPU acceleration is not required for the documented settings.

## 2. Local installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Verify the install:

```bash
python -m pytest -q
python -m compileall -q app.py rl_course tests
```

## 3. Run the app locally

```bash
python -m streamlit run app.py
```

Useful options for a shared machine or container:

```bash
python -m streamlit run app.py \\
  --server.address=0.0.0.0 \\
  --server.port=8501 \\
  --server.headless=true
```

Open `http://localhost:8501`. The initial page should show the course title,
chapter navigation, and the small Connect-K board.

## 4. Streamlit Community Cloud

1. Push the repository to a Git provider.
2. Create a new Streamlit app and select the repository’s `app.py` as the main
   file.
3. Use Python 3.10+ for the environment.
4. Deploy and open the generated URL.
5. Confirm that the first chapter loads, then visit the training chapter with
   the smallest settings before trying longer runs.

If the host expects a `requirements.txt` rather than PEP 621 metadata, create
one containing the runtime dependencies from `pyproject.toml`:

```text
numpy>=1.24
plotly>=5.18
streamlit>=1.35
torch>=2.1
```

Do not include the exported Marimo environment; it contains many unrelated
packages and makes cold starts substantially heavier.

## 5. Container deployment

The process must bind to the platform-provided port and all interfaces. A
minimal container command is:

```bash
python -m pip install -e .
python -m streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501} --server.headless=true
```

Configure the platform health check to request `/`. Streamlit’s initial HTML
response confirms that the process is listening; a browser or Streamlit-aware
probe is needed to verify the interactive websocket after that.

## 6. Resource guidance

- Start training with 2–4 games and 4–8 simulations per move.
- Increase settings only after confirming the host’s memory and request-time
  limits.
- The agent and training metrics live in session memory; multiple users each
  create their own trainer.
- For classroom-wide use, pre-training or a persisted model is a future
  optimization, not part of v1.

## 7. Security and operations

The app has no login, authorization, secrets, or external service integration.
Deploy it only where the lack of authentication is acceptable. Do not add API
keys to Streamlit secrets for the current implementation; none are needed.

For a public deployment, set platform-level access controls if the lesson is
intended for a private class and monitor memory consumption during training.

## 8. Release checklist

- `python -m pytest -q` passes;
- all eight chapters render in the Streamlit smoke test;
- the smallest training run completes on the target host;
- `/` returns HTTP 200;
- the play chapter can finish a seeded game;
- README and docs match the deployed command;
- no Marimo export or unrelated dependency lock is included in the runtime.
