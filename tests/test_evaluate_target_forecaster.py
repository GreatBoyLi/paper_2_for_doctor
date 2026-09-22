from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


SCRIPT_FILE = Path(__file__).resolve().parents[1] / "evaluation" / "evaluate_target_forecaster.py"


class EvaluateTargetForecasterTests(unittest.TestCase):
    def test_main_uses_validation_data_and_non_fold_outputs(self):
        text = SCRIPT_FILE.read_text(encoding="utf-8")
        self.assertNotIn("FOLD_ID", text)
        self.assertNotIn("fold_", text)
        self.assertIn('TARGET_DATASET_DIR / "val.npz"', text)
        self.assertIn('CHECKPOINT_DIR / "stage2_best.pt"', text)
        self.assertIn('EVALUATION_DIR / "target_validation"', text)

    def test_denormalize_power_uses_each_station_q99(self):
        self.assertTrue(SCRIPT_FILE.is_file(), f"Target评估脚本尚不存在：{SCRIPT_FILE}")
        from evaluation.evaluate_target_forecaster import denormalize_power

        normalized = np.array([[[1.0, 2.0], [0.5, 3.0]]], dtype=np.float32)
        q99 = np.array([10.0, 20.0], dtype=np.float32)

        actual = denormalize_power(normalized, q99)

        expected = np.array([[[10.0, 40.0], [5.0, 60.0]]], dtype=np.float32)
        np.testing.assert_allclose(actual, expected)

    def test_aggregate_metrics_follow_paper_formulas(self):
        self.assertTrue(SCRIPT_FILE.is_file(), f"Target评估脚本尚不存在：{SCRIPT_FILE}")
        from evaluation.evaluate_target_forecaster import calculate_aggregate_metrics

        targets = np.array([
            [[2.0, 3.0], [4.0, 6.0]],
            [[4.0, 6.0], [8.0, 12.0]],
        ])
        predictions = np.array([
            [[1.0, 2.0], [3.0, 5.0]],
            [[3.0, 5.0], [6.0, 10.0]],
        ])

        metrics = calculate_aggregate_metrics(predictions, targets, [1, 2])

        self.assertEqual(metrics["horizon_step"].tolist(), [1, 2])
        self.assertEqual(metrics["horizon_minutes"].tolist(), [15, 30])
        self.assertAlmostEqual(metrics.loc[0, "p_max"], 20.0)
        self.assertAlmostEqual(metrics.loc[0, "nmae"], 0.1)
        self.assertAlmostEqual(metrics.loc[0, "nrmse"], 0.1)
        self.assertAlmostEqual(metrics.loc[0, "mae"], 2.0)
        self.assertAlmostEqual(metrics.loc[0, "rmse"], 2.0)
        self.assertAlmostEqual(metrics.loc[0, "mape_percent"], 30.0)
        self.assertAlmostEqual(metrics.loc[0, "correlation_percent"], 100.0)
        self.assertEqual(metrics.loc[0, "mape_sample_count"], 2)
        self.assertAlmostEqual(metrics.loc[1, "nmae"], 0.15)
        self.assertAlmostEqual(metrics.loc[1, "nrmse"], np.sqrt(10.0) / 20.0)
        self.assertAlmostEqual(metrics.loc[1, "mae"], 3.0)
        self.assertAlmostEqual(metrics.loc[1, "rmse"], np.sqrt(10.0))
        self.assertAlmostEqual(metrics.loc[1, "mape_percent"], 20.0)
        self.assertAlmostEqual(metrics.loc[1, "correlation_percent"], 100.0)

    def test_mape_excludes_power_below_one_percent_of_pmax(self):
        self.assertTrue(SCRIPT_FILE.is_file(), f"Target评估脚本尚不存在：{SCRIPT_FILE}")
        from evaluation.evaluate_target_forecaster import calculate_aggregate_metrics

        targets = np.array([[[0.0]], [[0.5]], [[1.0]], [[100.0]]])
        predictions = np.array([[[5.0]], [[5.0]], [[2.0]], [[90.0]]])

        metrics = calculate_aggregate_metrics(predictions, targets, [1])

        self.assertEqual(metrics.loc[0, "mape_sample_count"], 2)
        self.assertAlmostEqual(metrics.loc[0, "mape_percent"], 55.0)

    def test_target_times_advance_by_fifteen_minutes(self):
        from evaluation.evaluate_target_forecaster import build_target_times

        starts = np.array(["2021-03-10T04:00:00"], dtype="datetime64[ns]")

        target_times = build_target_times(starts, output_steps=4)

        expected = np.array([[
            "2021-03-10T04:00:00", "2021-03-10T04:15:00",
            "2021-03-10T04:30:00", "2021-03-10T04:45:00",
        ]], dtype="datetime64[ns]")
        np.testing.assert_array_equal(target_times, expected)

    def test_physical_constraints_zero_night_and_negative_predictions(self):
        from evaluation.evaluate_target_forecaster import apply_pv_physical_constraints

        predictions = np.array([[[5.0], [-3.0]], [[7.0], [8.0]]], dtype=np.float32)
        target_times = np.array([
            ["2021-03-10T02:00:00", "2021-03-10T12:00:00"],
            ["2021-03-10T12:00:00", "2021-03-10T23:00:00"],
        ], dtype="datetime64[ns]")

        constrained, night_mask = apply_pv_physical_constraints(
            predictions, target_times, np.array([-37.718287]), np.array([145.050975]),
            "Australia/Melbourne",
        )

        np.testing.assert_allclose(constrained, [[[0.0], [0.0]], [[7.0], [0.0]]])
        np.testing.assert_array_equal(night_mask, [[[True], [False]], [[False], [True]]])

    def test_station_metrics_use_each_station_maximum(self):
        self.assertTrue(SCRIPT_FILE.is_file(), f"Target评估脚本尚不存在：{SCRIPT_FILE}")
        from evaluation.evaluate_target_forecaster import calculate_station_metrics

        targets = np.array([
            [[2.0, 5.0]],
            [[4.0, 10.0]],
        ])
        predictions = np.array([
            [[1.0, 3.0]],
            [[3.0, 8.0]],
        ])

        metrics = calculate_station_metrics(predictions, targets, ["A", "B"], [1])

        self.assertEqual(metrics["station"].tolist(), ["A", "B"])
        self.assertEqual(metrics["p_max"].tolist(), [4.0, 10.0])
        self.assertEqual(metrics["nmae"].tolist(), [0.25, 0.2])
        self.assertEqual(metrics["mae"].tolist(), [1.0, 2.0])
        self.assertEqual(metrics["rmse"].tolist(), [1.0, 2.0])
        np.testing.assert_allclose(metrics["mape_percent"], [37.5, 30.0])
        np.testing.assert_allclose(metrics["correlation_percent"], [100.0, 100.0])
        self.assertEqual(metrics["mape_sample_count"].tolist(), [2, 2])

    def test_save_results_writes_predictions_and_metric_tables(self):
        self.assertTrue(SCRIPT_FILE.is_file(), f"Target评估脚本尚不存在：{SCRIPT_FILE}")
        from evaluation.evaluate_target_forecaster import save_results

        predictions = np.ones((2, 2, 2), dtype=np.float32)
        targets = np.zeros((2, 2, 2), dtype=np.float32)
        q99 = np.array([10.0, 20.0], dtype=np.float32)
        aggregate_metrics = pd.DataFrame({"horizon_step": [1], "nmae": [0.1]})
        station_metrics = pd.DataFrame({"station": ["A"], "nmae": [0.2]})

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            target_times = np.array([
                ["2021-03-10T04:00:00", "2021-03-10T04:15:00"],
                ["2021-03-10T04:15:00", "2021-03-10T04:30:00"],
            ], dtype="datetime64[ns]")
            night_mask = np.zeros((2, 2, 2), dtype=bool)
            save_results(
                output_dir, predictions, targets, predictions * q99, targets * q99,
                q99, ["A", "B"], aggregate_metrics, station_metrics,
                predictions_normalized_before_constraints=predictions + 1,
                predictions_raw_before_constraints=(predictions + 1) * q99,
                target_times=target_times, night_mask=night_mask,
                metrics_before_constraints=aggregate_metrics,
            )
            saved = np.load(output_dir / "predictions.npz")

            np.testing.assert_allclose(saved["predictions_normalized"], predictions)
            np.testing.assert_allclose(
                saved["predictions_normalized_before_constraints"], predictions + 1
            )
            np.testing.assert_array_equal(saved["target_times"], target_times)
            np.testing.assert_array_equal(saved["night_mask"], night_mask)
            self.assertEqual(saved["station_names"].astype(str).tolist(), ["A", "B"])
            self.assertTrue((output_dir / "metrics_by_horizon.csv").is_file())
            self.assertTrue((output_dir / "metrics_by_station.csv").is_file())
            self.assertTrue((output_dir / "metrics_before_constraints.csv").is_file())


if __name__ == "__main__":
    unittest.main()
