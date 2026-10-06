# Marketing Mix Modeling — Validation Dataset

A synthetic dataset and interactive visualization for validating Marketing Mix Modeling (MMM) decomposition accuracy.

## Problem

Real-world MMM data has unknown ground truth. How do you know if your model's decomposition is correct?

This repository provides **synthetic data where the true decomposition is known**, enabling rigorous validation of MMM methods against ground truth.

## What's Included

### Core Files
- **`synthetic_mmm_dataset.py`** — Python generator supporting:
  - **v1.0:** 17-week baseline (3 channels: TV, Search, Social)
  - **v1.1:** 52-week realistic (5 channels: TV, Search, Social, Display, Affiliate)
- **`mmm_synthetic_data.csv`** — Generated sample dataset (observables + ground truth columns)
- **`mmm_ground_truth.json`** — Complete true decomposition (baseline, seasonality, adstock, campaign effects)
- **`seasonality_mmm_animation.html`** — Interactive 52-week visualization with speed control and progressive chart reveal

## Key Features

### Geometric Adstock Decay
Realistic media carryover modeling:
- **TV:** 85% decay rate per week
- **Search:** 30% decay rate per week
- **Social:** 50% decay rate per week

### Seasonality & Confounding
Built-in challenge scenarios:
- **Halloween Effect:** Weeks 9-10 show 0.8x sales spike (calendar, not campaign)
- **Black Friday Effect:** Weeks 15-16 show 1.2x sales spike (calendar, not campaign)
- **Campaign Timing:** Campaigns run weeks 8-9 and 14-15 — overlapping with holiday spikes

This creates a realistic identifiability challenge: can your model distinguish calendar-driven sales from campaign-driven sales when they move together?

### Saturation Curves
Logistic saturation modeling diminishing returns:
```
effect = strength * (adstocked / (half_saturation + adstocked))
```

## Quick Start

### Generate v1.0 Baseline Dataset (17 weeks, 3 channels)

```python
from synthetic_mmm_dataset import SyntheticMMMDataset

# Create generator
dataset = SyntheticMMMDataset(n_weeks=17, level="v1.0", seed=42)

# Generate data with custom parameters
df = dataset.generate(
    baseline=100,
    trend_rate=0.05,
    seasonality_strength=1.0,
    campaign_strength=0.8
)

# Save outputs
dataset.save_to_csv('v1_0_data.csv')
dataset.save_ground_truth('v1_0_truth.json')
```

### Generate v1.1 Realistic Dataset (52 weeks, 5 channels)

```python
from synthetic_mmm_dataset import SyntheticMMMDataset

# Create generator for 52-week realistic scenario
dataset = SyntheticMMMDataset(n_weeks=52, level="v1.1", seed=42)

# Generate data (includes trend breaks, multiple seasonality, heteroscedastic noise)
df = dataset.generate(
    baseline=1000,
    seasonality_strength=1.0,
    campaign_strength=1.0,
    noise_heteroscedastic=True
)

# Save outputs
dataset.save_to_csv('v1_1_data.csv')
dataset.save_ground_truth('v1_1_truth.json')

# Returns DataFrame with 5 channels, ground truth for validation
print(df.head())  # Includes tv_spend, search_spend, social_spend, display_spend, affiliate_spend
print(dataset.get_ground_truth().keys())
```

### Validate Your MMM Model

```python
import json
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

# Load generated data
df = pd.read_csv('mmm_synthetic_data.csv')

# Load ground truth
with open('mmm_ground_truth.json') as f:
    truth = json.load(f)

# Your model's estimates
my_seasonality = [...]  # Your decomposition
my_campaign_effect = [...]

# Compare
mape = mean_absolute_percentage_error(
    truth['seasonality'],
    my_seasonality
)
print(f"Seasonality MAPE: {mape:.1%}")
```

## Dataset Structure

```
week | sales | tv_spend | search_spend | social_spend | true_baseline | true_seasonality | true_campaign_effect | ...
-----|-------|----------|--------------|--------------|---------------|------------------|----------------------|
0    | 104.9 | 0        | 0            | 0            | 100.0         | 0.0              | 0.0                  |
...
9    | 110.2 | 100      | 80           | 60           | 102.9         | 0.8              | 1.37                 |
15   | 141.5 | 160      | 140          | 100          | 104.4         | 1.2              | 1.46                 |
```

## Use Cases

1. **Method Validation** — Test whether your decomposition technique recovers known ground truth
2. **Assumption Testing** — Explore how much multicollinearity breaks recovery, what saturation shapes matter
3. **Educational** — Understand confounding, identifiability, and the fit-vs-identification gap
4. **Conference/Publication** — Use as benchmark for comparing MMM methodologies
5. **Consulting** — Stress-test client assumptions before field work

## Limitations

This dataset is **intentionally simplified** for clarity:
- Only 17 weeks (real models need 52+ for seasonality estimation)
- Only 3 channels (realistic models have 10+)
- Perfect geometric adstock (reality is messier)
- No trend breaks, external shocks, or competitive data
- No measurement error (budget vs. actual spend)

**But it's perfect for:**
- Understanding confounding mechanisms
- Validating method assumptions
- Testing identifiability under known conditions
- Building intuition on decomposition challenges

## Customization

Modify `SyntheticMMMDataset.generate()` parameters:

```python
dataset.generate(
    baseline=150,              # Starting sales level
    trend_rate=0.08,           # Growth per week
    seasonality_strength=1.5,  # Amplitude of calendar effects
    campaign_strength=1.2,     # Campaign effectiveness multiplier
    noise_level=15,            # Standard deviation of random noise
    tv_decay=0.80,             # TV carryover rate
    search_decay=0.25,         # Search carryover rate
    social_decay=0.45          # Social carryover rate
)
```

## Interactive Visualization

Open `seasonality_mmm_animation.html` in a browser to:
- Watch week-by-week tank level visualization
- See adstock decay in real-time as campaigns run
- Identify calendar effects (holidays) vs. campaign-driven spikes
- Understand why R² can mask misattribution

## Data Output Format

### CSV (Observable + Ground Truth)
Includes both what you'd observe (sales, spend) and ground truth (true effects):
- `sales` — Observable sales (contains signal + noise)
- `true_baseline` — Baseline sales level
- `true_seasonality` — Calendar effect contribution
- `true_*_adstock` — Adstocked spend per channel
- `true_*_effect` — True campaign contribution per channel
- `true_campaign_effect` — Total campaign lift

### JSON (Complete Decomposition)
Structured for easy comparison:
```json
{
  "baseline": [100.0, 100.3, ...],
  "seasonality": [0.0, 0.0, ..., 0.8, 1.2, ...],
  "tv_adstock": [...geometric decay...],
  "tv_effect": [...saturation applied...],
  "campaign_effect": [...sum of all channels...],
  "noise": [...random component...]
}
```

## References

**Key Concepts:**
- Geometric adstock: Adstock(t) = Spend(t) + decay × Spend(t-1) + decay² × Spend(t-2) + ...
- Saturation: Effect = Strength × (Adstock / (Half-saturation + Adstock))
- Confounding: When two variables move together, regression cannot distinguish their effects

**Applications:**
- Validating media mix models before deployment
- Testing causal inference assumptions
- Benchmarking attribution algorithms
- Teaching MMM decomposition concepts

## License

Public domain — use freely for research, education, and commercial applications.

## Questions?

This dataset was designed to support rigorous validation of marketing attribution methods. Fork, modify, and test your own decomposition approaches.

---

*Last updated: 2026-10-06*
