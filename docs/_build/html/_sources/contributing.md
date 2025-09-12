---
title: Contributing
---

# Contributing Guide

Thanks for your interest in improving PNPL! This guide helps you get set up, make changes, and open high‑quality pull requests.

## How to help
- Report bugs and propose improvements.
- Improve documentation and examples.
- Add tests to cover new behavior and regressions.

## Development setup
```bash
git clone https://github.com/neural-processing-lab/pnpl-public.git
cd pnpl-public
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r docs/requirements.txt  # if you build docs locally
pip install pytest
```

Run tests:
```bash
pytest -q
```

Build docs:
```bash
jupyter-book build docs/
open docs/_build/html/index.html
```

## Pull requests
- Keep changes focused and small where possible.
- Include tests and docs updates for user‑visible changes.
- Add a clear description and motivation.

## Code style
- Follow existing patterns in the codebase.
- Prefer explicit names and small, composable functions.
- Type hints welcome where they clarify usage.

## Issue triage
- Reproduce the issue with a minimal example.
- Label: bug/enhancement/docs where appropriate.

Thanks for contributing!

