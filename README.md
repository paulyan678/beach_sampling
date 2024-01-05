# Autonomous beach microplastic sampling

Research code for autonomous beach sampling with Bayesian spatial inference and
reinforcement learning. The project develops a reproducible simulator, sampling
environment, planning baselines, and learning agents for information-aware surveys.

## Development setup

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install -e .
pytest
```

## License

MIT
