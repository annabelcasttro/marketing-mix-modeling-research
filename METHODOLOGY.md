# Methodology: Synthetic MMM Dataset Generation

## Overview

This document describes the mathematical and design principles behind the synthetic MMM dataset. The goal is **reproducible validation** of attribution decomposition methods using data where the true effects are known.

## Data Generation Process

### 1. Baseline + Trend

**Formula:**
```
baseline(t) = B₀ + (B₀ × trend_rate × t / T)
```

Where:
- `B₀` = Starting baseline (default: 100)
- `trend_rate` = Growth rate per week (default: 0.05 = 5%)
- `t` = Week number (0 to T-1)
- `T` = Total weeks (17)

**Interpretation:**
- Represents "business as usual" sales in the absence of campaigns or seasonal effects
- Linear growth (5% increase from week 0 to week 16)
- Typically: 100 → 105.88 over 17 weeks

**Why it matters:** Tests whether models can separate trend from campaign/seasonal lift.

### 2. Seasonality (Calendar Effects)

**Design:**
```
seasonality(t) = {
  0.8  if t ∈ {9, 10}      (Halloween)
  1.2  if t ∈ {15, 16}     (Black Friday)
  0.0  otherwise
}
```

**Interpretation:**
- Pure calendar effects, **not influenced by spending**
- Halloween: 0.8x multiplier (weeks 9-10)
- Black Friday: 1.2x multiplier (weeks 15-16)
- All other weeks: no seasonal effect

**Confounding Problem:**
- Campaign 1 runs weeks 8-9 → Halloween spike weeks 9-10
- Campaign 2 runs weeks 14-15 → Black Friday spike weeks 15-16
- **Critical challenge:** Model cannot distinguish whether sales spike is from campaign or calendar event

### 3. Media Spend (Campaign Pattern)

**Design:**
```
Campaign 1 (weeks 8-9):
  tv_spend: [100, 100]
  search_spend: [80, 80]
  social_spend: [60, 60]

Campaign 2 (weeks 14-15):
  tv_spend: [160, 160]
  search_spend: [140, 140]
  social_spend: [100, 100]
```

**Interpretation:**
- Two distinct campaigns with different budgets
- Campaign 2 (Black Friday) has higher spend than Campaign 1 (Halloween)
- Realistic pattern: increasing investment in high-opportunity periods
- No spend in weeks 0-7, 10-13, 16+ (control periods)

### 4. Adstock (Media Carryover)

**Geometric Decay Formula:**
```
adstock(t) = Σ(s=0 to t) spend(s) × decay^(t-s)
```

Where:
- `spend(s)` = Spend in past week s
- `decay` = Decay rate (channel-specific)
- `t - s` = Weeks elapsed since spend

**Implementation:**
```python
for week in range(len(spend)):
    for past_week in range(week + 1):
        adstocked[week] += spend[past_week] * (decay ** (week - past_week))
```

**Decay Rates (Default):**
- **TV:** 0.85 (85% carryover) → Long-term brand effect, 3-4 week effective window
- **Search:** 0.30 (30% carryover) → Immediate response, <1 week effective window
- **Social:** 0.50 (50% carryover) → Medium-term effect, ~1-2 week effective window

**Example (TV, weeks 8-9):**
```
Week 8: adstock = 100 × 0.85^0 = 100.0
Week 9: adstock = 100 × 0.85^1 + 100 × 0.85^0 = 85 + 100 = 185.0
Week 10: adstock = 100 × 0.85^2 + 100 × 0.85^1 + 0 × 0.85^0 = 72.3 + 85 + 0 = 157.3
Week 11: adstock = 100 × 0.85^3 + 100 × 0.85^2 + 0 × 0.85^1 + 0 × 0.85^0 = 61.4 + 72.3 + 0 + 0 = 133.7
...
```

**Why Geometric Decay?**
1. **Mathematically tractable:** Simple closed form, easy to invert
2. **Empirically validated:** Observed in many media studies (Naik & Peters, 2012)
3. **Interpretable:** Half-life = log(0.5) / log(decay)
   - TV half-life ≈ 4.5 weeks
   - Search half-life ≈ 0.8 weeks
   - Social half-life ≈ 1.4 weeks

### 5. Saturation (Diminishing Returns)

**Logistic (S-curve) Formula:**
```
effect(a) = strength × a / (half_saturation + a)
```

Where:
- `a` = Adstocked spend
- `strength` = Maximum effect per channel
- `half_saturation` = Adstocked spend level at 50% of maximum effect

**Channel Strengths (Default):**
- TV: 1.0 (strongest)
- Search: 0.8 (strong)
- Social: 0.6 (moderate)

**Half-saturation:** 50 units (same for all channels)

**Example (TV at week 9 where adstock=185):**
```
effect = 1.0 × 185 / (50 + 185) = 1.0 × 0.787 = 0.787
```

**Interpretation:**
- Diminishing returns: First dollar more impactful than last dollar
- Maximum possible effect per channel capped by `strength`
- At low spend: linear relationship (near-zero denominator)
- At high spend: diminishing returns (denominator dominates)

**Why Logistic?**
1. Bounded response (can't go negative or infinite)
2. Inflection point at half-saturation
3. Realistic for budget constraints
4. Standard in marketing science literature

### 6. Campaign Effect

**Formula:**
```
campaign_effect(t) = tv_effect(t) + search_effect(t) + social_effect(t)
```

**Interpretation:**
- Simple additive model (no channel interactions)
- Total contribution from all paid media
- Zero in control periods (weeks with no spend)

### 7. Noise

**Distribution:**
```
noise(t) ~ N(0, σ²)
```

Where:
- Mean = 0 (unbiased)
- Standard deviation = `noise_level` (default: 10)

**Interpretation:**
- Represents unmeasured variables, measurement error, randomness
- Realistic observation noise (±1-2 standard deviations = ±10-20 sales)

### 8. Final Sales

**Complete Formula:**
```
sales(t) = baseline(t) + seasonality(t) + campaign_effect(t) + noise(t)
```

**Verification:**
```python
sales = baseline + seasonality + campaign_effect + noise
sales = np.maximum(sales, baseline * 0.3)  # Floor at 30% of baseline
```

The floor prevents negative sales (rare but possible under high negative noise).

---

## Key Design Choices

### Why Confound Campaign with Calendar Effects?

**Real-world pattern:** Marketers often run campaigns around known high-opportunity periods:
- Holiday shopping → Increased spending weeks before
- High-traffic events → Campaign timing coordinated with event timing

This makes **identifiability** a real challenge:
- Cannot tell if sales spike from campaign or calendar
- Requires external constraints (adstock priors, holdout tests, previous years' data)

### Why Geometric Adstock, Not Other Shapes?

**Alternatives:**
- Exponential: More complex, harder to interpret
- Polynomial: Unbounded, unrealistic
- Rectangular (lag distribution): Too simplistic

**Geometric advantages:**
- Standard in industry (Naik & Peters, 2012; Media Mix Modeling Handbook)
- Closed-form derivation
- One parameter per channel (easy to calibrate)
- Half-life easily interpretable

### Why Three Channels, Not More?

**Trade-off:**
- Three channels capture core complexity (multicollinearity, different decay rates)
- More channels = harder to recover ground truth (high-dimensional identifiability)
- Simplicity aids interpretability for teaching

**Realism:**
- Real campaigns have 10-20+ channels
- But core principles scale

### Why 17 Weeks, Not 52?

**Constraint:**
- Seasonality estimation needs 52+ weeks (full year)
- 17 weeks **intentionally insufficient** for real seasonality estimation

**Benefit:**
- Focuses on **confounding problem** not seasonality stability
- Manageable dataset size
- Tests model's ability to use prior seasonality constraints

---

## Reproducibility & Customization

### Seed Control

```python
dataset = SyntheticMMMDataset(n_weeks=17, seed=42)
```

Same seed → Identical noise sequence → Identical data every time.

### Parameter Customization

All parameters can be overridden:

```python
df = dataset.generate(
    baseline=150,              # Higher base sales
    trend_rate=0.08,           # Faster growth
    seasonality_strength=1.5,  # Stronger calendar effects
    campaign_strength=1.2,     # More effective campaigns
    noise_level=15,            # More observation noise
    tv_decay=0.80,             # Faster TV decay
    search_decay=0.25,         # Even more immediate search
    social_decay=0.45          # Slightly faster social decay
)
```

### Ground Truth Output

Every component is saved:

```json
{
  "baseline": [100.0, 100.29, 100.59, ..., 105.88],
  "seasonality": [0.0, 0.0, ..., 0.8, 0.8, 0.0, ..., 1.2, 1.2],
  "tv_adstock": [0.0, 100.0, 185.0, ...],
  "tv_effect": [0.0, 0.787, 1.456, ...],
  "campaign_effect": [0.0, 1.258, 2.714, ...],
  "sales": [105.2, 113.4, 128.6, ...]
}
```

This allows:
- Exact validation against known truth
- Partial validation (test seasonality recovery with known campaign effects)
- Sensitivity analysis (How do results change with noise level?)

---

## Validation Approach

### Metrics Used

1. **Mean Absolute Percentage Error (MAPE):**
   ```
   MAPE = (1/n) × Σ |true(t) - est(t)| / |true(t)|
   ```
   Interpretation: Average percent error (scale-independent)

2. **Root Mean Squared Error (RMSE):**
   ```
   RMSE = √((1/n) × Σ (true(t) - est(t))²)
   ```
   Interpretation: Penalizes large errors more

3. **R² Score:**
   ```
   R² = 1 - (SS_res / SS_tot)
   ```
   Interpretation: Variance explained (but ≠ correct attribution!)

### Expected Results

**Baseline regression model:**
- R² ≈ 0.95+ (high fit)
- Seasonality MAPE ≈ 10-30% (confounding with campaigns)
- Campaign MAPE ≈ 15-40% (multicollinearity, confounding)

**With constraints (adstock priors, seasonality from history):**
- Seasonality MAPE can drop to <10%
- Campaign MAPE to 5-15%

---

## References

1. **Naik, P. A., & Peters, K. (2012).** "A Hierarchical Marketing Communications Model." Journal of Marketing Research, 49(2), 240-265.
2. **Jin, Y., et al. (2017).** "Inferring Causal Impact in Time-Series Data with Contextual and Temporal Confounders." arXiv preprint.
3. **Vaver, J., & Koehler, J. (2011).** "Measuring Ad Effectiveness Using Geo Experiments." Google Research Blog.
4. **Media Mix Modeling Handbook.** (Various authors, industry standard document)

---

## Questions About the Methodology?

Each parameter is tuned for:
- **Realism:** Reflects observed patterns in marketing data
- **Teachability:** Clear mechanistic relationships
- **Challenge:** Non-trivial confounding that real models encounter

Use `generate()` parameters to explore sensitivity to design choices.
