"""
Unit tests for synthetic MMM dataset generation.

Validates that:
1. Generated data has correct shape and types
2. Ground truth decomposition is mathematically correct
3. Adstock calculations follow geometric decay formula
4. Saturation curves are applied correctly
5. Seasonality patterns match specification
6. Reproducibility with seed control
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from synthetic_mmm_dataset import SyntheticMMMDataset


class TestDatasetGeneration:
    """Test suite for SyntheticMMMDataset"""

    def setup_method(self):
        """Create a fresh dataset for each test"""
        self.dataset = SyntheticMMMDataset(n_weeks=17, seed=42)
        self.df = self.dataset.generate()
        self.truth = self.dataset.get_ground_truth()

    def test_dataframe_shape(self):
        """Test that generated DataFrame has correct dimensions"""
        assert self.df.shape[0] == 17, "Should have 17 weeks"
        assert "sales" in self.df.columns, "Should have sales column"
        assert "week" in self.df.columns, "Should have week column"
        assert "tv_spend" in self.df.columns, "Should have tv_spend"
        assert "search_spend" in self.df.columns, "Should have search_spend"
        assert "social_spend" in self.df.columns, "Should have social_spend"

    def test_ground_truth_shape(self):
        """Test that ground truth has all required components"""
        required_keys = {
            "baseline", "seasonality", "tv_spend", "search_spend", "social_spend",
            "tv_adstock", "search_adstock", "social_adstock",
            "tv_effect", "search_effect", "social_effect",
            "campaign_effect", "noise", "sales"
        }
        assert set(self.truth.keys()) == required_keys, "Missing ground truth components"

    def test_baseline_monotonic_growth(self):
        """Test that baseline increases monotonically (positive trend)"""
        baseline = np.array(self.truth["baseline"])
        diffs = np.diff(baseline)
        assert np.all(diffs >= 0), "Baseline should be monotonically increasing"
        assert diffs.mean() > 0, "Baseline should grow on average"

    def test_seasonality_pattern(self):
        """Test that seasonality matches specification"""
        seasonality = np.array(self.truth["seasonality"])

        # Halloween weeks (9-10): +0.8x
        assert seasonality[9] == 0.8, f"Week 9 should be 0.8, got {seasonality[9]}"
        assert seasonality[10] == 0.8, f"Week 10 should be 0.8, got {seasonality[10]}"

        # Black Friday weeks (15-16): +1.2x
        assert seasonality[15] == 1.2, f"Week 15 should be 1.2, got {seasonality[15]}"
        assert seasonality[16] == 1.2, f"Week 16 should be 1.2, got {seasonality[16]}"

        # All other weeks: 0
        for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14]:
            assert seasonality[i] == 0.0, f"Week {i} should be 0, got {seasonality[i]}"

    def test_adstock_geometric_decay(self):
        """Test that adstock follows geometric decay formula"""
        spend = np.array(self.truth["tv_spend"])
        adstock = np.array(self.truth["tv_adstock"])
        decay = 0.85

        for week in range(len(spend)):
            expected_adstock = 0.0
            for past_week in range(week + 1):
                expected_adstock += spend[past_week] * (decay ** (week - past_week))
            assert np.isclose(adstock[week], expected_adstock, rtol=1e-5), \
                f"Adstock mismatch at week {week}"

    def test_campaign_spend_pattern(self):
        """Test that campaign spend follows specification"""
        tv_spend = np.array(self.truth["tv_spend"])
        search_spend = np.array(self.truth["search_spend"])
        social_spend = np.array(self.truth["social_spend"])

        # Campaign 1: weeks 8-9
        assert tv_spend[8] == 100, "Campaign 1 TV spend week 8 should be 100"
        assert tv_spend[9] == 100, "Campaign 1 TV spend week 9 should be 100"
        assert search_spend[8] == 80, "Campaign 1 Search spend week 8 should be 80"
        assert social_spend[8] == 60, "Campaign 1 Social spend week 8 should be 60"

        # Campaign 2: weeks 14-15
        assert tv_spend[14] == 160, "Campaign 2 TV spend week 14 should be 160"
        assert tv_spend[15] == 160, "Campaign 2 TV spend week 15 should be 160"
        assert search_spend[14] == 140, "Campaign 2 Search spend week 14 should be 140"
        assert social_spend[14] == 100, "Campaign 2 Social spend week 14 should be 100"

    def test_saturation_bounds(self):
        """Test that saturation effects are bounded correctly"""
        tv_effect = np.array(self.truth["tv_effect"])
        search_effect = np.array(self.truth["search_effect"])
        social_effect = np.array(self.truth["social_effect"])

        # All effects should be non-negative
        assert np.all(tv_effect >= 0), "TV effects should be non-negative"
        assert np.all(search_effect >= 0), "Search effects should be non-negative"
        assert np.all(social_effect >= 0), "Social effects should be non-negative"

        # Effects should be bounded by saturation strength
        assert np.all(tv_effect <= 1.0), "TV effects should be <= 1.0 (strength=1.0)"
        assert np.all(search_effect <= 0.8), "Search effects should be <= 0.8 (strength=0.8)"
        assert np.all(social_effect <= 0.6), "Social effects should be <= 0.6 (strength=0.6)"

    def test_campaign_effect_composition(self):
        """Test that total campaign effect is sum of channel effects"""
        tv = np.array(self.truth["tv_effect"])
        search = np.array(self.truth["search_effect"])
        social = np.array(self.truth["social_effect"])
        total = np.array(self.truth["campaign_effect"])

        expected_total = tv + search + social
        assert np.allclose(total, expected_total, rtol=1e-5), \
            "Campaign effect should be sum of channel effects"

    def test_sales_decomposition(self):
        """Test that sales = baseline + seasonality + campaign_effect + noise"""
        baseline = np.array(self.truth["baseline"])
        seasonality = np.array(self.truth["seasonality"])
        campaign = np.array(self.truth["campaign_effect"])
        noise = np.array(self.truth["noise"])
        sales = np.array(self.truth["sales"])

        expected_sales = baseline + seasonality + campaign + noise
        assert np.allclose(sales, expected_sales, rtol=1e-5), \
            "Sales decomposition formula failed"

    def test_reproducibility(self):
        """Test that same seed produces identical results"""
        dataset1 = SyntheticMMMDataset(n_weeks=17, seed=123)
        df1 = dataset1.generate()
        truth1 = dataset1.get_ground_truth()

        dataset2 = SyntheticMMMDataset(n_weeks=17, seed=123)
        df2 = dataset2.generate()
        truth2 = dataset2.get_ground_truth()

        # DataFrames should be identical
        pd.testing.assert_frame_equal(df1, df2)

        # Ground truth should be identical
        for key in truth1:
            val1 = truth1[key]
            val2 = truth2[key]
            if isinstance(val1, (list, np.ndarray)):
                assert np.allclose(val1, val2), f"Mismatch in {key}"
            else:
                assert val1 == val2, f"Mismatch in {key}"

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results"""
        dataset1 = SyntheticMMMDataset(n_weeks=17, seed=42)
        df1 = dataset1.generate()

        dataset2 = SyntheticMMMDataset(n_weeks=17, seed=99)
        df2 = dataset2.generate()

        # Noise should be different
        assert not np.allclose(df1["sales"].values, df2["sales"].values), \
            "Different seeds should produce different noise"

    def test_csv_save_load(self):
        """Test that saved CSV can be loaded and matches original"""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test_data.csv")
            self.dataset.save_to_csv(csv_path)

            # Load and verify
            loaded_df = pd.read_csv(csv_path)
            pd.testing.assert_frame_equal(self.df, loaded_df)

    def test_json_save_load(self):
        """Test that saved JSON can be loaded and matches original"""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "test_truth.json")
            self.dataset.save_ground_truth(json_path)

            # Load and verify
            with open(json_path) as f:
                loaded_truth = json.load(f)

            for key in self.truth:
                if isinstance(self.truth[key], list):
                    assert np.allclose(self.truth[key], loaded_truth[key]), \
                        f"Mismatch in {key} after JSON load"


def run_tests():
    """Run all tests and report results"""
    test_suite = TestDatasetGeneration()
    test_methods = [m for m in dir(test_suite) if m.startswith("test_")]

    print("=" * 70)
    print("MMM SYNTHETIC DATASET — VALIDATION TESTS")
    print("=" * 70)

    passed = 0
    failed = 0

    for test_method in test_methods:
        test_suite.setup_method()
        try:
            getattr(test_suite, test_method)()
            print(f"✓ {test_method}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_method}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_method}: {type(e).__name__}: {e}")
            failed += 1

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
