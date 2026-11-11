# Contributing to SmartPark AI

Thank you for your interest in contributing to **SmartPark AI**! We welcome contributions to improve model accuracy, spatial algorithms, dashboard features, and infrastructure.

## Code of Conduct
Please be respectful and constructive in all discussions, pull requests, and code reviews.

---

## How to Contribute

### 1. Reporting Bugs & Proposing Features
- Check the [Issue Tracker](https://github.com/Hemanth0707/SmartPark-AI/issues) to ensure your topic hasn't already been discussed.
- Provide a clear, reproducible example if reporting a bug, including hardware environment, OS, and Python version.

### 2. Setting Up Development Environment
1. Clone the repository:
   ```bash
   git clone https://github.com/Hemanth0707/SmartPark-AI.git
   cd SmartPark-AI
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .\.venv\Scripts\activate
   # Linux / macOS:
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Running Automated Tests
Before submitting a pull request, run the pytest suite to ensure zero regressions:
```bash
pytest tests -v
```
All 23 unit and integration tests must pass.

### 4. Code Style & Standards
- Follow **PEP 8** style guidelines.
- Add type hints (`typing`) for public functions and dataclasses.
- Maintain comprehensive docstrings explaining function arguments, return types, and algorithmic reasoning.
- Do not commit large binary checkpoints (>50 MB) or raw dataset directories (`CNRPark-Patches-150x150/`).

---

## Pull Request Guidelines
1. Fork the repo and create your branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Write unit tests for new features under `tests/`.
3. Commit with clear, descriptive commit messages:
   ```bash
   git commit -m "feat(matching): add Hungarian bipartite cost matrix optimization"
   ```
4. Push to your fork and submit a Pull Request targeting `main`.
