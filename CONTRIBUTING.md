# Contributing to DB Delay Forecaster

We follow a standard **Git Flow** workflow. All contributions should target the `develop` branch.

## Workflow

1.  **Branching:** Create a feature branch off `develop` (e.g., `feat/my-new-feature`).
2.  **Standards:** We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. Please run `pre-commit install` before making your first commit.
3.  **Testing:** Ensure all scripts run successfully using the provided Parquet sample data.
4.  **Pull Requests:** Target your PR at the `develop` branch. Once reviewed and tested, `develop` will be merged into `main` for releases.

## Development Setup

```bash
# Clone and enter repo
git clone https://github.com/HadiDawoud/db-delay-forecaster.git
cd db-delay-forecaster

# Install dev dependencies
pip install -e ".[dev]"
pre-commit install
```

Thank you for contributing to better train delay predictions!
