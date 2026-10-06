"""
Synthetic MMM Dataset with Ground Truth
========================================

Supports multiple complexity levels:
- v1.0: 17 weeks, 3 channels (TV, Search, Social)
- v1.1: 52 weeks, 5 channels (TV, Search, Social, Display, Affiliate)
  with trend breaks, multiple seasonality peaks, and realistic patterns

Perfect for validating MMM decomposition accuracy.
Author: Annabel Castro | MMM Practitioner
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path


class SyntheticMMMDataset:
    def __init__(self, seed=42, n_weeks=17, level="v1.0"):
        """
        Args:
            seed: Random seed for reproducibility
            n_weeks: Number of weeks (default 17 for v1.0, can be 52 for v1.1)
            level: Dataset complexity level ("v1.0" or "v1.1")
        """
        np.random.seed(seed)
        self.n_weeks = n_weeks
        self.level = level
        self.ground_truth = {}

    def generate(self, **kwargs):
        """
        Generate synthetic MMM data with ground truth.
        Dispatches to appropriate version handler.
        """
        if self.level == "v1.0":
            return self._generate_v1_0(**kwargs)
        elif self.level == "v1.1":
            return self._generate_v1_1(**kwargs)
        else:
            raise ValueError(f"Unknown level: {self.level}")

    def _generate_v1_0(self,
                       baseline=100,
                       trend_rate=0.05,
                       seasonality_strength=1.0,
                       campaign_strength=0.8,
                       noise_level=10,
                       tv_decay=0.85,
                       search_decay=0.3,
                       social_decay=0.5):
        """
        Generate v1.0: 17 weeks, 3 channels, basic confounding.

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
        tv_adstock = self._apply_adstock(tv_spend, tv_decay)
        search_adstock = self._apply_adstock(search_spend, search_decay)
        social_adstock = self._apply_adstock(social_spend, social_decay)

        # Saturation (Simple logistic curve)
        tv_effect = self._apply_saturation(tv_adstock, campaign_strength * 1.0)
        search_effect = self._apply_saturation(search_adstock, campaign_strength * 0.8)
        social_effect = self._apply_saturation(social_adstock, campaign_strength * 0.6)

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

    def _generate_v1_1(self,
                       baseline=1000,
                       trend_breaks=None,
                       seasonality_strength=1.0,
                       campaign_strength=1.0,
                       noise_heteroscedastic=True):
        """
        Generate v1.1: 52 weeks, 5 channels, trend breaks, multiple seasonality.

        Args:
            baseline: Starting sales level
            trend_breaks: List of (week, magnitude) tuples for trend changes
            seasonality_strength: Multiplier for seasonal effects
            campaign_strength: Multiplier for campaign effectiveness
            noise_heteroscedastic: If True, noise scales with sales level
        """
        if self.n_weeks != 52:
            raise ValueError("v1.1 requires n_weeks=52")

        if trend_breaks is None:
            trend_breaks = [(21, 0.10), (36, -0.05)]

        weeks = np.arange(self.n_weeks)

        # Baseline + Trend with breaks
        baseline_array = np.zeros(self.n_weeks)
        baseline_array[0] = baseline

        current_trend = 0.05  # Initial +5% growth
        for w in range(1, self.n_weeks):
            # Check for trend breaks
            for break_week, magnitude in trend_breaks:
                if w == break_week:
                    if magnitude > 0:
                        current_trend = magnitude  # Step increase
                    else:
                        current_trend = magnitude  # Step decrease
                elif w == break_week + 1 and magnitude < 0:
                    current_trend = 0.07  # Recovery trend after decrease

            growth = baseline_array[w - 1] * (1 + current_trend / 52)
            baseline_array[w] = growth

        self.ground_truth['baseline'] = baseline_array

        # Seasonality: Multiple peaks throughout year
        seasonality = np.zeros(self.n_weeks)
        peaks = [
            (5, 6, 0.4),      # Valentine's Day
            (13, 14, 0.6),    # Easter
            (22, 22, 0.8),    # Prime Day
            (28, 29, -0.2),   # Summer (negative)
            (42, 43, 0.8),    # Halloween
            (48, 50, 1.2),    # Black Friday/Cyber Monday
            (51, 51, 1.5),    # Christmas
        ]

        for start, end, strength in peaks:
            seasonality[start:end + 1] = seasonality_strength * strength

        self.ground_truth['seasonality'] = seasonality

        # Campaign spend: 5 channels with specific patterns
        channels = {
            'tv': np.zeros(self.n_weeks),
            'search': np.zeros(self.n_weeks),
            'social': np.zeros(self.n_weeks),
            'display': np.zeros(self.n_weeks),
            'affiliate': np.zeros(self.n_weeks),
        }

        # Campaign patterns (week_start, week_end, tv, search, social, display, affiliate)
        campaign_patterns = [
            (4, 6, 80, 60, 40, 30, 20),      # Valentine's
            (12, 14, 40, 30, 120, 50, 20),   # Easter (Social heavy)
            (21, 23, 200, 150, 100, 100, 80), # Prime Day (All peak)
            (27, 29, 50, 40, 50, 150, 30),   # Summer (Display heavy)
            (41, 43, 120, 100, 60, 40, 20),  # Halloween
            (47, 50, 180, 150, 120, 100, 70), # Black Friday
            (51, 52, 60, 50, 40, 30, 100),   # Christmas (Affiliate)
        ]

        for start, end, tv, search, social, display, affiliate in campaign_patterns:
            channels['tv'][start:end + 1] = tv
            channels['search'][start:end + 1] = search
            channels['social'][start:end + 1] = social
            channels['display'][start:end + 1] = display
            channels['affiliate'][start:end + 1] = affiliate

        # Adstock with channel-specific decay rates
        decay_rates = {
            'tv': 0.85,
            'search': 0.30,
            'social': 0.50,
            'display': 0.65,
            'affiliate': 0.20,
        }

        adstock = {}
        for channel, spend in channels.items():
            adstock[channel] = self._apply_adstock(spend, decay_rates[channel])

        # Saturation with channel-specific strengths
        strengths = {
            'tv': 1.0,
            'search': 0.8,
            'social': 0.6,
            'display': 0.7,
            'affiliate': 0.5,
        }

        effects = {}
        for channel in channels.keys():
            effects[channel] = self._apply_saturation(
                adstock[channel],
                campaign_strength * strengths[channel]
            )

        campaign_effect = sum(effects.values())

        # Heteroscedastic noise
        if noise_heteroscedastic:
            noise_std = 0.05 * baseline_array
            noise = np.random.normal(0, 1, self.n_weeks) * noise_std
        else:
            noise = np.random.normal(0, baseline * 0.05, self.n_weeks)

        # Total Sales
        sales = baseline_array + seasonality + campaign_effect + noise
        sales = np.maximum(sales, baseline_array * 0.5)  # Floor at 50% of baseline

        # Store ground truth
        for channel, spend in channels.items():
            self.ground_truth[f'{channel}_spend'] = spend.tolist()
            self.ground_truth[f'{channel}_adstock'] = adstock[channel].tolist()
            self.ground_truth[f'{channel}_effect'] = effects[channel].tolist()

        self.ground_truth['campaign_effect'] = campaign_effect.tolist()
        self.ground_truth['noise'] = noise.tolist()
        self.ground_truth['sales'] = sales.tolist()

        # Create DataFrame
        df_dict = {
            'week': weeks,
            'sales': sales,
            'true_baseline': baseline_array,
            'true_seasonality': seasonality,
            'true_campaign_effect': campaign_effect,
        }

        for channel in channels.keys():
            df_dict[f'{channel}_spend'] = channels[channel]
            df_dict[f'{channel}_adstock'] = adstock[channel]
            df_dict[f'{channel}_effect'] = effects[channel]

        self.df = pd.DataFrame(df_dict)
        return self.df

    def _apply_adstock(self, spend, decay_rate):
        """Apply geometric adstock decay to spend array."""
        adstocked = np.zeros_like(spend, dtype=float)
        for week in range(len(spend)):
            for past_week in range(week + 1):
                adstocked[week] += spend[past_week] * (decay_rate ** (week - past_week))
        return adstocked

    def _apply_saturation(self, adstocked, strength, half_saturation=50):
        """Apply logistic saturation curve to adstock."""
        return strength * (adstocked / (half_saturation + adstocked))

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
