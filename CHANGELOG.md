# Changelog

All notable changes to this project are documented in this file.

## [1.1.0] - 2026-10-07

### Added
- **v1.1 Data Generation** (`SyntheticMMMDataset(level="v1.1", n_weeks=52)`)
  - 52-week realistic dynamics with full-year seasonality patterns
  - 5 channels: TV (0.85 decay), Search (0.30), Social (0.50), Display (0.65), Affiliate (0.20)
  - Multiple seasonality peaks: Valentine's (0.4x), Easter (0.6x), Prime Day (0.8x), Summer (-0.2x), Halloween (0.8x), Black Friday (1.2x), Christmas (1.5x)
  - Trend breaks: +10% at week 21 (launch), -5% at week 36 (constraint), recovery +7% from week 37
  - Heteroscedastic noise scaling with baseline sales
  - 7 campaign patterns overlapping with seasonal events for high confounding

- **v1.1 Example** (`examples/52week_regression.py`)
  - End-to-end workflow for 52-week data
  - Linear regression validation showing identifiability challenges
  - Component decomposition accuracy (MAPE) analysis
  - Demonstrates why more sophisticated models needed for realistic datasets

### Changed
- **Refactored SyntheticMMMDataset** for version support
  - `__init__` now accepts `level` parameter ("v1.0" or "v1.1")
  - `generate()` dispatches to version-specific handlers
  - v1.0 backward compatible with all existing code
  - Extracted `_apply_adstock()` and `_apply_saturation()` as shared methods

### Tested
- v1.0: All 13 existing tests passing
- v1.1: Data generation, shape validation, decomposition accuracy, reproducibility

---

## [1.0.0] - 2026-10-06

### Added
- **Core Dataset Generator** (`synthetic_mmm_dataset.py`)
  - `SyntheticMMMDataset` class for generating synthetic MMM data with known ground truth
  - Geometric adstock decay modeling (TV: 85%, Search: 30%, Social: 50%)
  - Logistic saturation curves for diminishing returns
  - Seasonality/confounding scenarios (Halloween weeks 9-10, Black Friday weeks 15-16)
  - Campaign spend patterns (two campaigns with overlapping calendar effects)
  - Customizable baseline, trend, noise, and channel parameters

- **Sample Data** (`mmm_synthetic_data.csv`, `mmm_ground_truth.json`)
  - 17-week synthetic dataset with ground truth decomposition
  - Observable columns: week, sales, tv_spend, search_spend, social_spend
  - Ground truth columns: true_baseline, true_seasonality, true_*_adstock, true_*_effect, etc.

- **Interactive Visualization** (`seasonality_mmm_animation.html`)
  - Week-by-week adstock tank level visualization
  - Real-time decay rate tracking per channel
  - Holiday/calendar effect highlighting
  - Progress display and playback controls

- **Comprehensive Documentation**
  - `README.md`: Quick start guide, use cases, limitations
  - `METHODOLOGY.md`: Detailed mathematical formulation of data generation process
  - `CONTRIBUTING.md`: Guidelines for extending the dataset and examples

- **Validation & Testing**
  - `tests/test_dataset_generation.py`: 14 unit tests covering:
    - Shape and structure validation
    - Ground truth mathematical correctness
    - Adstock geometric decay formula
    - Saturation bounds
    - Sales decomposition
    - Reproducibility with seeds
    - CSV/JSON save/load functionality

- **Examples**
  - `examples/basic_validation.py`: End-to-end workflow demonstrating:
    - Dataset generation
    - Linear regression fitting
    - Decomposition accuracy measurement
    - Confounding detection

- **Package Configuration**
  - `setup.py`: Python package setup for `pip install -e .`
  - `requirements.txt`: Minimal dependencies (numpy, pandas, scipy)
  - `.gitignore`: Standard Python project ignores
  - `LICENSE`: MIT License (open source)

### Design Principles
- **Reproducibility:** Seed-based deterministic generation
- **Transparency:** All formulas and parameters explicitly documented
- **Validation:** Extensive test suite ensuring correctness
- **Simplicity:** Intentionally simplified (17 weeks, 3 channels) for teaching
- **Realism:** Geometric adstock, logistic saturation, campaign confounding reflect real patterns

### Key Features
1. **Confounding Scenario:** Campaigns overlap with known calendar effects
   - Demonstrates identifiability challenges in MMM
   - Tests whether models can separate calendar from campaign lift
   
2. **Channel Heterogeneity:** Different decay rates per channel
   - TV: Slow decay (brand effect)
   - Search: Fast decay (immediate response)
   - Social: Medium decay
   
3. **Complete Ground Truth:** Every component accessible
   - Baseline + trend
   - Seasonality
   - Adstock per channel
   - Campaign effects
   - Noise

4. **Parametric Flexibility:** All parameters customizable
   - Dataset size
   - Baseline level and growth
   - Seasonality strength
   - Campaign effectiveness
   - Decay rates
   - Noise level

### Tested Scenarios
- ✓ Correct ground truth generation
- ✓ Geometric adstock formula validation
- ✓ Saturation curve bounds
- ✓ Sales decomposition correctness
- ✓ Reproducibility with fixed seeds
- ✓ CSV/JSON persistence
- ✓ Expected MAPE ranges on validation regression

### Known Limitations
- Only 17 weeks (insufficient for real seasonality estimation)
- Only 3 channels (real models need 10+)
- Perfect geometric adstock (reality messier)
- No trend breaks, external shocks, competitive data
- Additive model (no channel interactions)
- Simplified campaign pattern (just 2 campaigns)

### Future Work
- [ ] Alternative adstock formulations (Hill equation, GAS model)
- [ ] External shocks and competitor campaigns
- [ ] Media budget constraints
- [ ] Channel interaction effects
- [ ] Measurement error models
- [ ] Multi-region hierarchical dataset
- [ ] Integration with popular MMM frameworks

---

## Version History Template (for future releases)

### [X.X.X] - YYYY-MM-DD

#### Added
- New features or functionality

#### Changed
- Modifications to existing features

#### Fixed
- Bug fixes

#### Deprecated
- Features to be removed in future versions

#### Removed
- Removed features or deprecations from previous versions

#### Security
- Security fixes or announcements

---

**Note:** Version 1.0 represents the initial public release of the synthetic MMM dataset. The dataset is production-ready for research, education, and model validation purposes. See METHODOLOGY.md for technical details and CONTRIBUTING.md for extension opportunities.
