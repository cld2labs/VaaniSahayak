# Contributing to Vaani Sahayak

Thank you for your interest in contributing to Vaani Sahayak! This guide will help you get started.

## Quick Setup Checklist

- [ ] Fork and clone the repository
- [ ] Set up your local development environment
- [ ] Create a feature branch
- [ ] Make your changes
- [ ] Submit a pull request

---

## Table of Contents

1. [Getting Help](#getting-help)
2. [Reporting Bugs](#reporting-bugs)
3. [Suggesting Features](#suggesting-features)
4. [Development Setup](#development-setup)
5. [Contributing Code](#contributing-code)
6. [Branching Model](#branching-model)
7. [Commit Conventions](#commit-conventions)
8. [Code Guidelines](#code-guidelines)
9. [Pull Request Checklist](#pull-request-checklist)

---

## Getting Help

- Open a [GitHub Discussion](https://github.com/cld2labs/VaaniSahayak/discussions) for questions
- Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues
- Review existing [Issues](https://github.com/cld2labs/VaaniSahayak/issues) before opening a new one

---

## Reporting Bugs

1. Search existing issues to avoid duplicates
2. Open a new issue with the **Bug Report** template
3. Include:
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (macOS version, Python version, chip type)
   - Relevant logs or error messages

---

## Suggesting Features

1. Open an issue with the **Feature Request** template
2. Describe the use case and expected behavior
3. Explain why this feature would benefit Vaani Sahayak users

---

## Development Setup

### Option 1: Local Development

```bash
# 1. Fork and clone
git clone https://github.com/<your-username>/VaaniSahayak.git
cd VaaniSahayak

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r backend/requirements.txt

# 4. Set up environment
cp .env.example .env
# Edit .env with your HF_TOKEN

# 5. Download data and compute embeddings
python scripts/download_data.py
python scripts/precompute_embeddings.py

# 6. Start servers (see README.md for full instructions)
python servers/server_param1.py --preload --port 8001   # Terminal 1
python servers/server_tts.py --preload --port 8003       # Terminal 2
uvicorn backend.main:app --reload --port 8000    # Terminal 3
cd frontend && npm install && npm run dev         # Terminal 4
```

### Option 2: Docker (backend + frontend only)

```bash
# Model servers must run natively for MPS acceleration
python servers/server_param1.py --preload --port 8001
python servers/server_tts.py --preload --port 8003

# Start backend + frontend
docker compose up --build
```

---

## Contributing Code

1. **Create a branch** from `main` (see [Branching Model](#branching-model))
2. **Make focused changes** — one feature or fix per branch
3. **Test locally** — ensure the app works end-to-end
4. **Commit** using [Conventional Commits](#commit-conventions)
5. **Push** and open a pull request

---

## Branching Model

Use the following branch prefixes:

| Prefix | Purpose | Example |
|--------|---------|---------|
| `feat/` | New feature | `feat/telugu-stt-support` |
| `fix/` | Bug fix | `fix/tts-memory-leak` |
| `docs/` | Documentation | `docs/update-api-reference` |
| `refactor/` | Code refactoring | `refactor/retriever-async` |
| `chore/` | Build, CI, tooling | `chore/update-dependencies` |

---

## Commit Conventions

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:** `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `ci`, `perf`

**Examples:**
```
feat(tts): add Telugu voice support
fix(retrieval): handle empty embedding results gracefully
docs(readme): add Docker deployment instructions
chore(deps): bump transformers to 4.42.0
```

---

## Code Guidelines

- **Backend:** Follow PEP 8. Use type hints for function signatures.
- **Frontend:** Use functional React components with hooks. Follow existing Tailwind patterns.
- **Structure:** Keep files in their designated directories (`backend/models/`, `backend/retrieval/`, etc.)
- **No secrets:** Never commit API keys, tokens, or credentials. Use `.env` for local config.
- **Clear commits:** Each commit should represent a single logical change.

---

## Pull Request Checklist

Before submitting your PR, confirm:

- [ ] I have read the [CONTRIBUTING.md](CONTRIBUTING.md) guidelines
- [ ] My branch is up to date with `main`
- [ ] New environment variables are documented in `.env.example`
- [ ] No secrets or credentials are committed
- [ ] The app runs locally without errors
- [ ] I have tested the affected endpoints / UI components
- [ ] My commit messages follow the Conventional Commits format
- [ ] I have updated documentation if the change affects user-facing behavior

---

Thank you for helping make government welfare schemes more accessible to every citizen!
