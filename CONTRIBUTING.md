# Contributing

Thanks for your interest in contributing. This document explains how to set up the project for development, run checks, and submit a pull request.

## Prerequisites

- **Python 3.14+** (see [.python-version](.python-version))
- **[uv](https://docs.astral.sh/uv/)** — install with:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
  or `brew install uv` on macOS.

## Development setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/<owner>/ukr_synth_dataset.git
   cd ukr_synth_dataset
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```
   This creates a virtual environment (if needed), pins the Python version from `.python-version`, and installs all dependencies from `pyproject.toml` and `uv.lock`.

3. **Verify the package imports**
   ```bash
   uv run python -c "import ukr_synth; print('Import successful')"
   ```

## Linting and type checking

The project uses **Ruff** (lint + format) and **MyPy** (type checking). Config is in `pyproject.toml`.

Run the same checks as CI:

```bash
# Lint (Ruff)
uv run ruff check .
uv run ruff format --check .

# Type check (MyPy)
uv run mypy src/ukr_synth
```

- **Ruff**: line length 100, excludes `output`, `images`, `fonts`. Fix issues with `ruff check . --fix` and `ruff format .`.
- **MyPy**: checks `src/ukr_synth`. Add type hints and fix reported errors before opening a PR.

## Tests

Automated tests are not yet in the repo. When added, run them with:

```bash
uv run pip install pytest
uv run pytest
```

CI also runs the import check above to ensure the package is importable.

## Pull requests

1. **Fork** the repository and clone your fork (or create a branch if you have write access).

2. **Create a branch** for your change:
   ```bash
   git checkout -b feature/your-feature
   # or: fix/your-fix
   ```

3. **Make your changes.** Keep commits focused and messages clear.

4. **Run lint** (see above) and fix any issues.

5. **Push** to your fork and **open a Pull Request** against `main`.

6. **Ensure CI passes.** The GitHub Action runs on push/PR to `main`: it installs deps with `uv sync --frozen`, runs Ruff (lint + format check) and MyPy, and verifies `import ukr_synth`. All steps must pass.

7. **Address review feedback** if any. Maintainers may request changes before merging.

## Code style

- Follow the existing style in the codebase.
- **Ruff** rules are in `[tool.ruff]` in `pyproject.toml` (line length 100, target Python 3.14).
- **MyPy** is used for type checking; prefer type hints for function signatures and fix any reported issues.

## Questions

Open an issue for bugs, feature ideas, or questions. For small typos or docs, a direct PR is fine.
