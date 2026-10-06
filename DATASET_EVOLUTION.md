# Dataset Evolution Roadmap

## Overview

Progressive complexity levels for synthetic MMM datasets. Each level adds realistic challenges to stress-test attribution decomposition methods.

**Goal:** Enable practitioners to validate models across increasing complexity before deploying to real data.

---

## Version Roadmap

### ✅ Level 1: BASELINE (v1.0) — Current

**Focus:** Foundational confounding and identifiability

**Specs:**
- **Duration:** 17 weeks
- **Channels:** 3 (TV, Search, Social)
- **Adstock:** Geometric decay (TV 85%, Search 30%, Social 50%)
- **Saturation:** Logistic curves
- **Seasonality:** 2 holidays (Halloween weeks 9-10, Black Friday weeks 15-16)
- **Confounding:** Campaign overlaps with holidays
- **Trend:** Linear 5% growth
- **Noise:** N(0, 10)

**Key Challenge:** Can model distinguish calendar-driven from campaign-driven lift?

**Validation Test:** Basic linear regression MAPE:
- Seasonality: 10-30%
- Campaign effect: 15-40%
- R²: >0.95 (but decomposition wrong!)

**Files Generated:**
- `mmm_synthetic_data.csv` (17 rows)
- `mmm_ground_truth.json`
- `seasonality_mmm_animation.html`

**Use Case:** Teaching confounding and identifiability concepts

---

### 🔄 Level 2: REALISTIC DYNAMICS (v1.1) — Next Priority

**Focus:** Full-year complexity with trend breaks and multiple seasonality patterns

**Specs:**
- **Duration:** 52 weeks (full year)
- **Channels:** 5
  - TV (decay 0.85, strength 1.0)
  - Search (decay 0.30, strength 0.8)
  - Social (decay 0.50, strength 0.6)
  - Display (decay 0.65, strength 0.7)
  - Affiliate (decay 0.20, strength 0.5)

**Seasonality:** Multiple peaks
```
Week 5-6:   Valentine's Day (0.4x)
Week 13-14: Easter (0.6x)
Week 22:    Prime Day (0.8x)
Week 28-29: Summer (0.2x negative)
Week 42-43: Halloween (0.8x)
Week 48-50: Black Friday/Cyber Monday (1.2x)
Week 51-52: Christmas/New Year (1.5x)
```

**Campaign Pattern:**
```
Campaign 1: Weeks 4-6 (Valentine's)    [TV, Search high]
Campaign 2: Weeks 12-14 (Easter)       [Social heavy]
Campaign 3: Weeks 21-23 (Prime Day)    [All channels peak]
Campaign 4: Weeks 27-29 (Summer)       [Display heavy]
Campaign 5: Weeks 41-43 (Halloween)    [TV, Search peak]
Campaign 6: Weeks 47-50 (Black Friday) [All channels peak]
Campaign 7: Weeks 51-52 (Christmas)    [Affiliate dominant]
```

**Trend Breaks:**
```
Weeks 0-20:  Linear +5% growth
Weeks 21:    Step change +10% (new product launch)
Weeks 22-35: Linear +3% growth
Weeks 36:    Step change -5% (supply constraint)
Weeks 37-52: Linear +7% growth (recovery)
```

**Additional Dynamics:**
- **Weekday Effect:** Search peaks mid-week, Social peaks weekends
  - Pseudo-seasonality on 7-day cycle
- **Lag Structure:** Different adstock per channel (modeled)
- **Noise:** Heteroscedastic (higher during high-sales periods)
  - Noise = N(0, 0.05 × baseline_t)

**Confounding Matrix:**
```
                        Campaign Overlap    Identifiability Challenge
Valentine (Week 5-6)    Week 4-6 campaign   Moderate
Easter (Week 13-14)     Week 12-14 campaign High
Prime Day (Week 22)     Week 21-23 campaign High
Halloween (Week 42-43)  Week 41-43 campaign High
Black Friday (Week 48-50) Week 47-50 campaign Very High
Christmas (Week 51-52)  Week 51-52 campaign Very High
```

**Files Generated:**
- `mmm_synthetic_data_52w.csv` (52 rows)
- `mmm_ground_truth_52w.json`
- Seasonality visualization (HTML)
- Trend break indicators (JSON)

**Validation Test:** Regression MAPE:
- Seasonality: 20-40% (hard to separate from trends)
- Campaign effect: 25-50% (multicollinearity, confounding)
- Each channel: 30-60% individual recovery

**Key Learning:** Full-year data doesn't solve identifiability without external constraints

**Implementation Plan:**
```python
# New method in SyntheticMMMDataset
def generate_52week_realistic(
    self,
    seasonality_pattern="full_year",
    trend_breaks=[21, 36],  # Week numbers
    noise_heteroscedastic=True,
    weekday_effect=True
):
    """Generate 52-week realistic MMM data"""
    ...
```

---

### 📋 Level 3: ADVANCED CONFOUNDING (v1.2) — After v1.1

**Focus:** Budget constraints, channel interactions, external shocks

**Specs (extends v1.1):**

**Budget Constraints:**
```python
# Total marketing budget by week (varies)
budget_by_week = [...]  # 52 values

# Allocation rules
tv_budget_pct = 0.40     # Fixed 40% to TV
search_budget_pct = 0.30 # Fixed 30% to Search
remaining = 0.30         # Dynamic allocation: Social + Display + Affiliate

# Channel competition: allocating to one channel reduces another
```

**Channel Interactions:**
- **Synergy:** TV + Search together > sum of parts
  - `interaction_effect = tv_adstock × search_adstock × 0.15`
- **Cannibalization:** Display competes with Search
  - `search_effect × (1 - 0.10 × display_adstock)`

**External Shocks:**
```
Week 15: Competitor campaign launch
  - Reduces organic baseline by 8%
  - Lasts 4 weeks (persistent)

Week 30: PR crisis
  - Sudden -15% sales shock
  - Recovery over 6 weeks (exponential)

Week 45: Influencer partnership
  - Amplifies Social channel by 2x
  - Interacts with organic growth
```

**Measurement Error:**
- Budget vs. actual spend variance (±10%)
- Channel misattribution (5% of spend misclassified)
- Pricing changes not captured in spend

**Noise Pattern:** Correlated errors
- Weekly noise components
- Residual autocorrelation (AR(1) component)

**Files Generated:**
- `mmm_data_advanced.csv`
- `mmm_ground_truth_advanced.json`
- `interaction_matrix.json`
- `shock_timeline.json`

**Validation Test:** MAPE targets
- Baseline: 15-30%
- Seasonality: 25-50%
- Campaign: 40-70%
- Interactions: 50-80% (very hard!)

---

### 🏢 Level 4: HIERARCHICAL MULTI-REGION (v2.0) — Later Priority

**Focus:** Scaling to realistic multi-market structure

**Specs:**
- **Regions:** 10 (e.g., US regions, or countries)
- **Duration:** 2 years (104 weeks)
- **Channel Types:**
  - **Global:** TV, Search (same effect across regions)
  - **Regional:** Social, Local Display (region-specific)
  - **Interaction:** Local overrides global with coefficient

**Hierarchy Structure:**
```
Global Baseline (shared across regions)
└── Regional Multiplier (1.0-2.0x)
    ├── Local seasonality (region-specific weeks)
    ├── Regional campaigns (subset of channels)
    └── Regional shocks (weather, local events)
```

**Regional Variations:**
- Baseline varies 0.5x to 2.5x (rural vs urban)
- Seasonality shifted ±2 weeks (hemisphere differences)
- Channel effectiveness varies (e.g., TV higher in older regions)
- Price sensitivity per region

**Correlated Errors:**
- Regions share macro shock component
- But region-specific micro errors

**Files Generated:**
- `mmm_data_regions_104w.csv` (1040 rows: 10 regions × 104 weeks)
- `region_hierarchy.json`
- Per-region ground truth files

---

### 🔮 Level 5: PRODUCTION-LIKE (v2.1) — Future

**Focus:** Real-world complexity (SKU-level, inventory, pricing)

**Specs:**
- **SKUs:** 50-500
- **Duration:** 3 years (156 weeks)
- **Aggregation Levels:**
  - SKU-level (lowest)
  - Category (medium)
  - Total brand (highest)
- **Inventory Effects:**
  - Stockouts reduce sales (random, 1-2% of weeks)
  - High inventory encourages promotion
- **Pricing Dynamics:**
  - Price elasticity: -1.5 (1% price increase → 1.5% volume decrease)
  - Promotional pricing (periodic)
  - Competitor pricing response
- **Cannibalization:**
  - SKU interactions (cross-elasticity)
- **Real Noise:**
  - Non-normal (skewed, fat tails)
  - Outliers (returns, refunds)
  - Seasonal heteroscedasticity

---

## Implementation Priority

### Phase 1 (This Month)
- [ ] Implement v1.1 (52-week realistic)
- [ ] Add 5-channel structure
- [ ] Add multiple seasonality peaks
- [ ] Add trend breaks
- [ ] Tests for each component
- [ ] Example: "52-week validation"

### Phase 2 (Next Month)
- [ ] Implement v1.2 (budget constraints)
- [ ] Add channel interactions
- [ ] Add external shocks
- [ ] Add measurement error
- [ ] Tests for confounding matrix
- [ ] Example: "interaction effects recovery"

### Phase 3 (Later)
- [ ] Implement v2.0 (hierarchical)
- [ ] Implement v2.1 (production-like)
- [ ] Optimization (generate 500+ SKUs fast)

---

## Framework Design

### Modular Generation

```python
class SyntheticMMMDataset:
    def generate(self, level="v1.0", **kwargs):
        """Generate dataset at specified level"""
        
        # Core components (reusable)
        baseline = self._generate_baseline(kwargs)
        seasonality = self._generate_seasonality(level, kwargs)
        adstock = self._generate_adstock(kwargs)
        campaigns = self._generate_campaigns(level, kwargs)
        noise = self._generate_noise(kwargs)
        
        # Level-specific composition
        if level == "v1.0":
            return self._compose_v1_0(baseline, seasonality, adstock, campaigns, noise)
        elif level == "v1.1":
            trend_breaks = self._generate_trend_breaks(kwargs)
            return self._compose_v1_1(baseline, seasonality, adstock, campaigns, trend_breaks, noise)
        elif level == "v1.2":
            interactions = self._generate_interactions(kwargs)
            shocks = self._generate_shocks(kwargs)
            return self._compose_v1_2(..., interactions, shocks, ...)
        # etc.
```

### Test Suite Evolution

```
tests/
├── test_v1_0_baseline.py        # Current
├── test_v1_1_realistic.py       # New
├── test_v1_2_advanced.py        # New
├── test_v2_0_hierarchical.py    # New
├── test_confounding_matrix.py   # New
└── test_cross_level_validation.py # New
```

### Examples Evolution

```
examples/
├── 01_basic_validation.py        # Current v1.0
├── 02_52week_regression.py       # v1.1
├── 03_interaction_recovery.py    # v1.2
├── 04_hierarchical_bayesian.py   # v2.0
└── 05_production_skus.py         # v2.1
```

---

## Success Criteria

### v1.1 Completion
- [ ] 52 weeks of realistic data generated correctly
- [ ] All tests passing (seasonality, trend breaks, 5 channels)
- [ ] Example notebook showing MAPE expectations
- [ ] Documentation updated (METHODOLOGY.md)
- [ ] Performance: Generate in <1 second

### v1.2 Completion
- [ ] Budget constraints + allocations working
- [ ] Channel interactions validated
- [ ] External shocks integration
- [ ] Measurement error models
- [ ] Confounding matrix analysis
- [ ] All tests passing

### v2.0 Completion
- [ ] Hierarchical generation with 10 regions
- [ ] Regional variation tests
- [ ] Shared vs. regional effect separation
- [ ] Example with hierarchical Bayesian model

---

## References & Prior Art

**Comparable Systems:**
- Google Media Mix Modeling library (GitHub)
- Facebook's Robyn (open source)
- Uber's PyMC-Marketing

**Academic Foundation:**
- Naik & Peters (2012) - Hierarchical MMM
- Hitsch & Misra (2018) - Identifiability in marketing
- Broadbent & Fry (1994) - Adstock models

---

## Questions to Resolve

1. **v1.1 weekday effect:** Separate component or folded into seasonality?
2. **Noise model:** Heteroscedastic or constant variance?
3. **Trend break magnitude:** Random or user-specified?
4. **v1.2 shock duration:** Fixed or probabilistic?
5. **v2.0 regional scaling:** Linear multiplier or non-linear?

---

**Last Updated:** 2026-10-06
**Owner:** Annabel Castro
**Status:** Planning Phase → Implementing v1.1
