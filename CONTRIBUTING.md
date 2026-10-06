# Contributing to Marketing Mix Modeling Research

Thank you for interest in contributing! This guide explains how to extend and improve the dataset and tools.

## Philosophy

This project focuses on **generating realistic synthetic MMM datasets with known ground truth**. All contributions should align with:

1. **Reproducibility:** Same seed = Same data always
2. **Transparency:** All parameters and formulas documented
3. **Validation:** Changes must pass test suite
4. **Simplicity:** Intentionally simplified for teaching (not production)

## Getting Started

### Installation (Development Mode)

```bash
git clone https://github.com/annabelcasttro/marketing-mix-modeling-research.git
cd marketing-mix-modeling-research

# Install with development dependencies
pip install -e ".[dev]"
# or manually:
pip install -r requirements.txt
```

### Running Tests

```bash
# Run all tests
python tests/test_dataset_generation.py

# Or with pytest (if installed)
pytest tests/ -v
```

All tests must pass before submitting a PR.

## Areas for Contribution

### 1. New Data Generation Scenarios

**Add realistic confounding patterns:**
- External shocks (competitor campaigns, price changes)
- Multiple confounders (seasonality + events + promotions)
- Non-linear channel interactions
- Media budget constraints (budget pools, allocation rules)

**Requirements:**
- Document the scenario in METHODOLOGY.md
- Add test cases validating the scenario
- Add example script using the scenario

**Example:**
```python
# In synthetic_mmm_dataset.py, add new method:
def generate_with_competitor_shock(self, shock_week=10, shock_magnitude=0.2):
    """Generate data with competitor campaign shock"""
    df = self.generate()
    # Add shock to sales at shock_week
    ...
    return df
```

### 2. Alternative Adstock Formulations

**Current:** Geometric decay (closed form)

**Alternatives to implement:**
- Adstock with partial decay (Adstock = current_spend + decay × previous_adstock)
- Polynomial adstock (Adstock(t) = Σ spend(s) × (decay × (t-s))^p)
- Generalized Adstock Stock (GAS) model
- Normalized Adstock (Adstock / max_adstock)

**Requirements:**
- Prove mathematical equivalence to ground truth
- Add test comparing outputs
- Benchmark performance (generate + decompose speed)

### 3. Saturation Curve Variants

**Current:** Logistic saturation

**Alternatives:**
- Multiplicative saturation: effect = strength × 1/(1 + exp(-adstock))
- Hill equation (Michaelis-Menten): effect = strength × adstock^s / (k^s + adstock^s)
- Power law: effect = strength × adstock^p
- Diminishing returns: effect = strength × log(1 + adstock)

**Requirements:**
- Add parameter to `generate(saturation_type="logistic")`
- Validate against ground truth
- Add example in `examples/`

### 4. Advanced Examples

**Current:** Basic linear regression validation

**Contribute examples:**
- Bayesian MMM (PyMC3, Stan)
- Regularized regression (Ridge, Lasso, Elastic Net)
- Hierarchical models (multiple regions)
- Causal inference approaches (causalml, DoWhy)
- Time series models (state space, ARIMA)

**Requirements:**
- Functional example script in `examples/`
- Run it against 5 different random seeds
- Report MAPE on each component
- Compare to baseline regression

### 5. Improved Documentation

**Opportunities:**
- Interactive Jupyter notebook tutorial
- Video walkthrough of methodology
- More detailed use-case documentation
- Comparison to real MMM datasets (if public available)

**Requirements:**
- Clear, beginner-friendly language
- Runnable code examples
- Visual diagrams where helpful

## Code Style & Standards

### Python Style

- Follow PEP 8 (use `black` for formatting)
- Type hints where helpful (optional)
- Docstrings for public methods (Google style)
- Comments for complex logic (avoid obvious comments)

### Example:

```python
def apply_adstock(self, spend: np.ndarray, decay_rate: float) -> np.ndarray:
    """
    Apply geometric adstock decay to spend.

    Args:
        spend: Weekly spend array of shape (n_weeks,)
        decay_rate: Decay parameter ∈ [0, 1]

    Returns:
        Adstocked spend array of same shape
    """
    adstocked = np.zeros_like(spend, dtype=float)
    for week in range(len(spend)):
        for past_week in range(week + 1):
            adstocked[week] += spend[past_week] * (decay_rate ** (week - past_week))
    return adstocked
```

### Testing

- Add unit tests for new functionality
- Use descriptive test names: `test_adstock_geometric_decay_correctness()`
- Test edge cases (zero spend, decay=1, decay=0)
- Verify reproducibility with fixed seeds

### Documentation

- Update METHODOLOGY.md with new parameters/formulas
- Add docstrings to new methods
- Include example usage in docstring

## Submission Process

### Before Submitting

1. **Fork the repository**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes** following code style guide

3. **Add tests**
   ```bash
   python tests/test_dataset_generation.py
   ```

4. **Update documentation**
   - METHODOLOGY.md for new methodology
   - Examples if new use case
   - README if user-facing change

5. **Commit with clear message**
   ```bash
   git commit -m "Add competitor shock scenario to dataset generation

   - Implement generate_with_competitor_shock() method
   - Add validation test for shock week timing
   - Update METHODOLOGY.md with shock formula
   - Add example in examples/competitor_shock.py"
   ```

### Pull Request

Create PR with:
- **Title:** One-line summary (e.g., "Add Hill equation saturation curves")
- **Description:**
  - What problem does this solve?
  - What changed?
  - Why this approach?
  - Any trade-offs?
- **Tests passing:** Include test output
- **Documentation:** Link to updated docs

### Review Process

- Maintainer reviews for:
  - Correctness (tests pass, math sound)
  - Alignment with project philosophy
  - Code quality & style
  - Documentation clarity
- Address feedback in same branch
- Merge once approved

## Development Setup

### Recommended Tools

```bash
# For code formatting
pip install black
black synthetic_mmm_dataset.py

# For linting
pip install flake8
flake8 synthetic_mmm_dataset.py

# For testing
pip install pytest
pytest tests/ -v

# For type checking (optional)
pip install mypy
mypy synthetic_mmm_dataset.py
```

### Project Structure

```
marketing-mix-modeling-research/
├── synthetic_mmm_dataset.py     # Core generator (main file to modify)
├── tests/
│   └── test_dataset_generation.py
├── examples/
│   ├── basic_validation.py
│   └── (your new examples here)
├── METHODOLOGY.md               # Update with new methods
├── README.md
└── setup.py
```

## Questions?

- Open an issue for discussion before starting major work
- Comment on existing issues if you want to help
- Email maintainer for questions

---

## Code of Conduct

- Be respectful and constructive
- Welcome diverse perspectives
- Focus on technical merit, not personal attacks
- Assume good intent

## License

By contributing, you agree your changes are licensed under the MIT License.
