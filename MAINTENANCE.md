# Maintenance (lightweight)

Use this as a reminder for **small, real commits** when you touch the repo anyway.  
GitHub’s contribution graph counts **commits** on the default branch (with your GitHub-verified email)—not empty “noise” commits.

## Quick wins (pick any)

- Bump a dependency in `requirements.txt` / `pyproject.toml` after `pip list --outdated` (test with `python main.py` or `MAX_ROWS=...`).
- Add one line to [CHANGELOG.md](CHANGELOG.md) under **Unreleased** when you merge work.
- Fix a typo in README or a docstring; run `pre-commit run --all-files`.
- Refresh a plot caption or a metric in README after a new training run.
- Note a new Hugging Face month in `data/download.py` when you extend the dataset window.
