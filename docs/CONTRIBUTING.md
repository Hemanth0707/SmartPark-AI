# Contributing to SmartPark AI

Thank you for your interest in contributing to SmartPark AI! We welcome contributions from the community.

## 1. Development Workflow

1. **Fork the Repository:** Create a personal fork on GitHub.
2. **Clone & Branch:**
   ```bash
   git clone https://github.com/<your-username>/SmartPark-AI.git
   cd SmartPark-AI
   git checkout -b feature/your-feature-name
   ```
3. **Set Up Python Environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   pip install pytest
   ```

---

## 2. Code Standards

- **PEP 8:** Follow standard Python formatting conventions.
- **Type Hints:** All new functions and public methods must provide PEP 484 type annotations.
- **Docstrings:** Use Google-style or standard Sphinx docstrings explaining arguments and return values.
- **Logging:** Use `src.utils.logger.get_logger()` instead of raw `print()` statements.

---

## 3. Running Automated Tests

Before submitting any Pull Request, ensure that all test suites pass cleanly:

```bash
pytest tests -v
```

Ensure new features are accompanied by corresponding unit tests in `tests/`.

---

## 4. Pull Request Checklist

- [ ] Code is well-formatted and adheres to type hinting.
- [ ] Pytest passes 100% of test cases.
- [ ] Documentation updated if CLI flags or configurations changed.
- [ ] No temporary files, virtual environments, or large weights committed to git.
