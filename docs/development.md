---
title: Development
---

# Development

## Running Tests

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install pytest
pytest -q
```

## Building this documentation locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r docs/requirements.txt
jupyter-book build docs/
open docs/_build/html/index.html
```

