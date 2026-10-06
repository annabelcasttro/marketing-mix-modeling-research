"""
Synthetic MMM Dataset with Ground Truth
========================================

Generates 17 weeks of synthetic sales data with:
- True adstock decay (geometric): TV 85%, Search 30%, Social 50%
- True seasonality: Halloween (weeks 9-10), Black Friday (weeks 15-16)
- True campaign spend with saturation curves
- Baseline + trend
- Controlled noise

Perfect for validating MMM decomposition accuracy.
Author: Annabel Castro | MMM Practitioner
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path


class SyntheticMMMDataset:
    def __init__(self, seed=42, n_weeks=17):
        np.random.seed(seed)
        self.n_weeks = n_weeks
        self.ground_truth = {}

    def generate(self,
                 baseline=100,
                 trend_rate=0.05,
                 seasonality_strength=1.0,
                 campaign_strength=0.8,
                 noise_level=10,
                 tv_decay=0.85,
                 search_decay=0.3,
                 social_decay=0.5):
        """
        Generate synthetic MMM data with ground truth.

        Args:
            baseline: Starting sales level
            trend_rate: Linear trend growth per week
            seasonality_strength: Multiplier for seasonal effects
            campaign_strength: Multiplier for campaign effectiveness
            noise_level: Standard deviation of random noise
            tv_decay: Geometric adstock decay rate for TV
            search_decay: Geometric adstock decay rate for Search
            social_decay: Geometric adstock decay rate for Social
        """

        weeks = np.arange(self.n_weeks)

        # Baseline + Trend
        self.ground_truth['baseline'] = baseline + (baseline * trend_rate * weeks / self.n_weeks)

        # Seasonality (calendar effects - NOT influenced by spending)
        seasonality = np.zeros(self.n_weeks)
        seasonality[9] = seasonality_strength * 0.8   # Halloween
        seasonality[10] = seasonality_strength * 0.8  # Halloween
        seasonality[15] = seasonality_strength * 1.2  # Black Friday
        seasonality[16] = seasonality_strength * 1.2  # Black Friday
        self.ground_truth['seasonality'] = seasonality

        # Campaign Spend (Media Mix)
        tv_spend = np.zeros(self.n_weeks)
        search_spend = np.zeros(self.n_weeks)
        social_spend = np.zeros(self.n_weeks)

        # Campaign 1: Weeks 8-9 (confounded with Halloween in week 9-10)
        tv_spend[8:10] = 100
        search_spend[8:10] = 80
        social_spend[8:10] = 60

        # Campaign 2: Weeks 14-15 (confounded with Black Friday in week 15-16)
        tv_spend[14:16] = 160
        search_spend[14:16] = 140
        social_spend[14:16] = 100

        # Adstock (Geometric Decay)
        def apply_adstock(spend, decay_rate):
            adstocked = np.zeros_like(spend, dtype=float)
            for week in range(len(spend)):
                for past_week in range(week + 1):
                    adstocked[week] += spend[past_week] * (decay_rate ** (week - past_week))
            return adstocked

        tv_adstock = apply_adstock(tv_spend, tv_decay)
        search_adstock = apply_adstock(search_spend, search_decay)
        social_adstock = apply_adstock(social_spend, social_decay)

        # Saturation (Simple logistic curve)
        def apply_saturation(adstocked, strength, half_saturation=50):
            return strength * (adstocked / (half_saturation + adstocked))

        tv_effect = apply_saturation(tv_adstock, campaign_strength * 1.0)
        search_effect = apply_saturation(search_adstock, campaign_strength * 0.8)
        social_effect = apply_saturation(social_adstock, campaign_strength * 0.6)

        campaign_effect = tv_effect + search_effect + social_effect

        # Noise
        noise = np.random.normal(0, noise_level, self.n_weeks)

        # Total Sales
        sales = self.ground_truth['baseline'] + seasonality + campaign_effect + noise
        sales = np.maximum(sales, self.ground_truth['baseline'] * 0.3)  # Floor at 30% of baseline

        # Store ground truth
        self.ground_truth['tv_spend'] = tv_spend.tolist()
        self.ground_truth['search_spend'] = search_spend.tolist()
        self.ground_truth['social_spend'] = social_spend.tolist()
        self.ground_truth['tv_adstock'] = tv_adstock.tolist()
        self.ground_truth['search_adstock'] = search_adstock.tolist()
        self.ground_truth['social_adstock'] = social_adstock.tolist()
        self.ground_truth['tv_effect'] = tv_effect.tolist()
        self.ground_truth['search_effect'] = search_effect.tolist()
        self.ground_truth['social_effect'] = social_effect.tolist()
        self.ground_truth['campaign_effect'] = campaign_effect.tolist()
        self.ground_truth['noise'] = noise.tolist()
        self.ground_truth['sales'] = sales.tolist()

        # Create DataFrame
        self.df = pd.DataFrame({
            'week': weeks,
            'sales': sales,
            'tv_spend': tv_spend,
            'search_spend': search_spend,
            'social_spend': social_spend,
            # Ground truth (normally hidden in real MMM)
            'true_baseline': self.ground_truth['baseline'],
            'true_seasonality': seasonality,
            'true_tv_adstock': tv_adstock,
            'true_search_adstock': search_adstock,
            'true_social_adstock': social_adstock,
            'true_campaign_effect': campaign_effect,
            'true_tv_effect': tv_effect,
            'true_search_effect': search_effect,
            'true_social_effect': social_effect,
        })

        return self.df

    def get_ground_truth(self):
        """Return the ground truth (what a perfect MMM model should recover)"""
        return self.ground_truth

    def save_to_csv(self, filename='mmm_synthetic_data.csv'):
        """Save dataset to CSV"""
        self.df.to_csv(filename, index=False)
        print(f"Dataset saved to {filename}")

    def save_ground_truth(self, filename='mmm_ground_truth.json'):
        """Save ground truth to JSON"""
        ground_truth_safe = {k: v if isinstance(v, (int, float, str)) else
                            [float(x) for x in v] for k, v in self.ground_truth.items()}
        with open(filename, 'w') as f:
            json.dump(ground_truth_safe, f, indent=2)
        print(f"Ground truth saved to {filename}")


if __name__ == '__main__':
    # Generate default dataset
    dataset = SyntheticMMMDataset(n_weeks=17)
    df = dataset.generate(
        baseline=100,
        trend_rate=0.05,
        seasonality_strength=1.0,
        campaign_strength=0.8,
        noise_level=10
    )

    # Save files
    dataset.save_to_csv()
    dataset.save_ground_truth()

    print("\n=== Dataset Summary ===")
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    print(f"\nTotal Sales: {df['sales'].sum():.2f}")
    print(f"True Campaign Effect Sum: {dataset.get_ground_truth()['campaign_effect']}")
    print(f"True Seasonality Sum: {dataset.get_ground_truth()['seasonality'].sum():.2f}")
