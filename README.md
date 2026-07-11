# RL Course Lab

An interactive, Python-first introduction to reinforcement learning. The app
starts with bandits and UCB1, builds a Connect-K environment, visualizes MCTS,
and ends with a small AlphaZero-style agent.

## Run locally

```bash
python -m pip install -e '.[dev]'
python -m streamlit run app.py
```

Run the algorithm tests with `python -m pytest -q`. Using the module form
ensures the test runner comes from the same environment as the installed
dependencies.

## Documentation

- [Product specification](docs/spec.md)
- [Engineering design](docs/engg.md)
- [Deployment guide](docs/deploy.md)
- [Troubleshooting](docs/troubleshoot.md)
