#!/usr/bin/env python3
"""
52-Week Regression Example: Validating channel attribution on realistic data.

Demonstrates:
- Generating v1.1 (52-week realistic) dataset with 5 channels
- Trend breaks, multiple seasonal peaks, heteroscedastic noise
- Linear regression validation and decomposition accuracy
- Channel-level MAPE analysis
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from synthetic_mmm_dataset import SyntheticMMMDataset

from scipy.linalg import lstsq


def mean_absolute_percentage_error(y_true, y_pred):
    """Compute MAPE ignoring zero divisions."""
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-6))) * 100


print("=" * 80)
print("52-WEEK REGRESSION VALIDATION EXAMPLE (v1.1)")
print("=" * 80)

# 1. Generate realistic 52-week dataset
print("\n1. Generating 52-week dataset with realistic dynamics...")
dataset = SyntheticMMMDataset(n_weeks=52, level="v1.1", seed=42)
df = dataset.generate(
    baseline=1000,
    seasonality_strength=1.0,
    campaign_strength=1.0,
    noise_heteroscedastic=True
)
truth = dataset.get_ground_truth()

print(f"   Shape: {df.shape}")
print(f"   Date range: Week 0-51")
print(f"   Channels: TV, Search, Social, Display, Affiliate")
print(f"   Features: 5 channels × 3 (spend, adstock, effect) = 15 features")

# 2. Prepare features for regression
print("\n2. Preparing regression features...")
channels = ['tv', 'search', 'social', 'display', 'affiliate']
adstock_cols = [f'{ch}_adstock' for ch in channels]

X = df[adstock_cols].values
y = df['sales'].values

print(f"   Feature matrix shape: {X.shape}")
print(f"   Target shape: {y.shape}")

# 3. Fit linear regression using scipy
print("\n3. Fitting linear regression model...")
X_with_const = np.column_stack([np.ones(len(X)), X])
coeffs, _, _, _ = lstsq(X_with_const, y)
intercept = coeffs[0]
coefs = coeffs[1:]
y_pred = intercept + X @ coefs

ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
r2 = 1 - (ss_res / ss_tot)
rmse = np.sqrt(ss_res / len(y))

print(f"   R² Score: {r2:.4f}")
print(f"   RMSE: {rmse:.2f}")

# 4. Analyze coefficient accuracy
print("\n4. Channel coefficient analysis...")
true_effects = {
    'tv': np.array(truth['tv_effect']),
    'search': np.array(truth['search_effect']),
    'social': np.array(truth['social_effect']),
    'display': np.array(truth['display_effect']),
    'affiliate': np.array(truth['affiliate_effect']),
}

true_strengths = {
    'tv': 1.0,
    'search': 0.8,
    'social': 0.6,
    'display': 0.7,
    'affiliate': 0.5,
}

print("\nChannel            Est Coef  Actual Strength  Error %")
print("-" * 55)
for i, channel in enumerate(channels):
    est_coef = coefs[i]
    true_strength = true_strengths[channel]
    error_pct = np.abs(est_coef - true_strength) / (true_strength + 1e-6) * 100
    print(f"{channel:15s}  {est_coef:8.4f}  {true_strength:8.4f}       {error_pct:6.1f}%")

# 5. Decompose predicted vs actual
print("\n5. Decomposition accuracy...")
baseline = np.array(truth['baseline'])
seasonality = np.array(truth['seasonality'])
true_campaign = np.array(truth['campaign_effect'])

# Compare components
print("\nComponent         Predicted  Actual  MAPE %")
print("-" * 50)

# For baseline, use intercept as estimate
baseline_est = intercept / len(y)
baseline_mape = mean_absolute_percentage_error(baseline, np.full_like(baseline, intercept / len(y)))

print(f"Baseline          {baseline_est:8.2f}  {baseline.mean():8.2f}  {baseline_mape:6.1f}")

# Campaign effect decomposition
campaign_mape = mean_absolute_percentage_error(true_campaign, y_pred - intercept - seasonality)
print(f"Campaign Effect   {y_pred.mean() - baseline_est:8.2f}  {true_campaign.mean():8.2f}  {campaign_mape:6.1f}")

# Seasonality is hard to recover from additive model without separation
seasonality_mape = mean_absolute_percentage_error(
    seasonality[seasonality != 0],
    (y[seasonality != 0] - baseline[seasonality != 0] - true_campaign[seasonality != 0])
)
print(f"Seasonality       {'N/A':>8s}  {seasonality.mean():8.2f}  {'N/A':>6s}")

# 6. Key observations
print("\n6. Key Learning Points:")
print("   • With 52 weeks and 5 channels, model has more signal/noise data")
print("   • Trend breaks create identifiability challenges (weeks 21, 36)")
print("   • Multiple overlapping seasonality peaks increase confounding")
print("   • Heteroscedastic noise mirrors realistic sales variance")
print("   • Channel interactions (if present) cannot be captured by linear model")
print("   • Adstock lags mean current-week spend doesn't capture full effect")

print("\n" + "=" * 80)
