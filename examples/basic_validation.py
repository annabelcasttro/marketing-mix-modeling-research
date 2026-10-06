"""
Example 1: Basic Validation Workflow

This example demonstrates how to:
1. Generate synthetic MMM data with known ground truth
2. Fit a simple regression model to recover effects
3. Compare model estimates against ground truth
4. Quantify decomposition accuracy

Expected Outcome:
A baseline regression model should recover approximately:
- Seasonality: MAPE ~5-15% (some multicollinearity with campaigns)
- Campaign Effect: MAPE ~10-20% (identifiability challenges due to confounding)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# Import the dataset generator
sys.path.insert(0, str(Path(__file__).parent.parent))
from synthetic_mmm_dataset import SyntheticMMMDataset


def main():
    print("=" * 80)
    print("EXAMPLE 1: Basic Validation Workflow")
    print("=" * 80)

    # Step 1: Generate synthetic data
    print("\n[1] Generating synthetic MMM dataset...")
    dataset = SyntheticMMMDataset(n_weeks=17, seed=42)
    df = dataset.generate(
        baseline=100,
        trend_rate=0.05,
        seasonality_strength=1.0,
        campaign_strength=0.8,
        noise_level=10
    )
    truth = dataset.get_ground_truth()
    print(f"    Generated {len(df)} weeks of data")

    # Step 2: Prepare features for regression
    print("\n[2] Preparing regression features...")
    X = df[[
        "true_tv_adstock",
        "true_search_adstock",
        "true_social_adstock"
    ]].values

    # Add trend feature
    trend = np.arange(len(df)).reshape(-1, 1)
    X = np.hstack([trend, X])

    y = df["sales"].values

    # Standardize features (helps with interpretation and convergence)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"    Features shape: {X_scaled.shape}")
    print(f"    Sales range: [{y.min():.1f}, {y.max():.1f}]")

    # Step 3: Fit baseline regression model
    print("\n[3] Fitting baseline linear regression model...")
    model = LinearRegression()
    model.fit(X_scaled, y)

    print(f"    R² Score: {model.score(X_scaled, y):.4f}")
    print(f"    Residual RMSE: {np.sqrt(mean_squared_error(y, model.predict(X_scaled))):.2f}")

    # Step 4: Extract decomposition from model
    print("\n[4] Extracting decomposition from model...")

    # Get predictions
    predictions = model.predict(X_scaled)

    # Baseline is roughly the intercept
    baseline_estimate = np.full(len(df), model.intercept_)

    # Campaign effects from coefficients (scaled)
    # Note: This is a simplified approach; real MMM would use more sophisticated methods
    feature_coefs = model.coef_[1:]  # Skip trend coefficient
    feature_stds = scaler.scale_[1:]

    tv_effect_est = (df["true_tv_adstock"].values * feature_coefs[0]) / feature_stds[0]
    search_effect_est = (df["true_search_adstock"].values * feature_coefs[1]) / feature_stds[1]
    social_effect_est = (df["true_social_adstock"].values * feature_coefs[2]) / feature_stds[2]

    campaign_effect_est = tv_effect_est + search_effect_est + social_effect_est

    # Seasonality is residual after removing baseline and campaign
    seasonality_est = predictions - baseline_estimate - campaign_effect_est

    # Step 5: Validate against ground truth
    print("\n[5] Validating decomposition accuracy...")
    print("\n" + "-" * 80)
    print(f"{'Component':<20} {'True Sum':<15} {'Est Sum':<15} {'MAPE':<15} {'RMSE':<15}")
    print("-" * 80)

    true_baseline = np.array(truth["baseline"])
    true_seasonality = np.array(truth["seasonality"])
    true_campaign = np.array(truth["campaign_effect"])

    components = {
        "Baseline": (true_baseline, baseline_estimate),
        "Seasonality": (true_seasonality, seasonality_est),
        "Campaign Effect": (true_campaign, campaign_effect_est),
    }

    results = {}
    for name, (true_vals, est_vals) in components.items():
        # Only calculate MAPE where true value is non-zero (avoid division issues)
        mask = np.abs(true_vals) > 0.001
        if mask.any():
            mape = mean_absolute_percentage_error(
                true_vals[mask],
                est_vals[mask]
            )
        else:
            mape = 0.0 if np.allclose(est_vals, 0) else float('inf')

        rmse = np.sqrt(mean_squared_error(true_vals, est_vals))
        true_sum = true_vals.sum()
        est_sum = est_vals.sum()

        results[name] = {
            "mape": mape,
            "rmse": rmse,
            "true_sum": true_sum,
            "est_sum": est_sum
        }

        mape_str = f"{mape:.1%}" if mape != float('inf') else "N/A"
        print(f"{name:<20} {true_sum:>14.2f} {est_sum:>14.2f} {mape_str:>14} {rmse:>14.2f}")

    print("-" * 80)

    # Step 6: Channel-level analysis
    print("\n[6] Channel-level decomposition accuracy...")
    print("\n" + "-" * 80)
    print(f"{'Channel':<20} {'True Sum':<15} {'Est Sum':<15} {'MAPE':<15} {'RMSE':<15}")
    print("-" * 80)

    true_tv_effect = np.array(truth["tv_effect"])
    true_search_effect = np.array(truth["search_effect"])
    true_social_effect = np.array(truth["social_effect"])

    channels = {
        "TV": (true_tv_effect, tv_effect_est),
        "Search": (true_search_effect, search_effect_est),
        "Social": (true_social_effect, social_effect_est),
    }

    for name, (true_vals, est_vals) in channels.items():
        mask = np.abs(true_vals) > 0.001
        if mask.any():
            mape = mean_absolute_percentage_error(
                true_vals[mask],
                est_vals[mask]
            )
        else:
            mape = 0.0 if np.allclose(est_vals, 0) else float('inf')

        rmse = np.sqrt(mean_squared_error(true_vals, est_vals))
        true_sum = true_vals.sum()
        est_sum = est_vals.sum()

        mape_str = f"{mape:.1%}" if mape != float('inf') else "N/A"
        print(f"{name:<20} {true_sum:>14.2f} {est_sum:>14.2f} {mape_str:>14} {rmse:>14.2f}")

    print("-" * 80)

    # Step 7: Confounding Analysis
    print("\n[7] Confounding Detection...")
    print("\n    Halloween weeks (9-10): Calendar effect +0.8x")
    print("    Campaigns: Weeks 8-9 (TV 100, Search 80, Social 60)")
    print("    Issue: Campaign 1 ends when Halloween spike begins")
    print(f"\n    True seasonality weeks 9-10: {true_seasonality[9]:.2f}, {true_seasonality[10]:.2f}")
    print(f"    Estimated seasonality weeks 9-10: {seasonality_est[9]:.2f}, {seasonality_est[10]:.2f}")
    print(f"    Estimation error: {abs(seasonality_est[9] - true_seasonality[9]):.2f}")

    print("\n    Black Friday weeks (15-16): Calendar effect +1.2x")
    print("    Campaigns: Weeks 14-15 (TV 160, Search 140, Social 100)")
    print(f"\n    True seasonality weeks 15-16: {true_seasonality[15]:.2f}, {true_seasonality[16]:.2f}")
    print(f"    Estimated seasonality weeks 15-16: {seasonality_est[15]:.2f}, {seasonality_est[16]:.2f}")
    print(f"    Estimation error: {abs(seasonality_est[15] - true_seasonality[15]):.2f}")

    # Step 8: Save results
    print("\n[8] Results saved to results.json")
    results_output = {
        "model_r2": float(model.score(X_scaled, y)),
        "model_rmse": float(np.sqrt(mean_squared_error(y, model.predict(X_scaled)))),
        "decomposition_errors": results,
        "channel_errors": {
            name: {"mape": v["mape"], "rmse": v["rmse"]}
            for name, v in results.items()
        }
    }

    with open("results.json", "w") as f:
        json.dump(results_output, f, indent=2)

    print("\n" + "=" * 80)
    print("Key Insight: High R² (>0.95) doesn't mean correct decomposition!")
    print("             Notice multicollinearity between Halloween and campaigns.")
    print("=" * 80)


if __name__ == "__main__":
    main()
