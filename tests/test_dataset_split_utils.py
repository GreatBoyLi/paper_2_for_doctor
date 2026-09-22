import unittest

import numpy as np
import pandas as pd

from dataset.split_utils import build_window_batch, split_source_months


class DatasetSplitUtilsTests(unittest.TestCase):
    def test_source_months_use_first_floor_eighty_percent_days_for_training(self):
        index = pd.date_range("2014-01-01", "2014-02-28 23:45", freq="15min")
        data = pd.DataFrame({"A": np.arange(len(index))}, index=index)

        train_blocks, val_blocks, summary = split_source_months(data, 0.8)

        self.assertEqual(len(train_blocks), 2)
        self.assertEqual(len(val_blocks), 2)
        self.assertEqual(train_blocks[0].index.max(), pd.Timestamp("2014-01-24 23:45"))
        self.assertEqual(val_blocks[0].index.min(), pd.Timestamp("2014-01-25 00:00"))
        self.assertEqual(train_blocks[1].index.max(), pd.Timestamp("2014-02-22 23:45"))
        self.assertEqual(val_blocks[1].index.min(), pd.Timestamp("2014-02-23 00:00"))
        self.assertEqual(summary["train_days"].tolist(), [24, 22])
        self.assertEqual(summary["val_days"].tolist(), [7, 6])

    def test_windows_never_cross_a_block_boundary(self):
        first = pd.DataFrame(
            {"A": np.arange(40)},
            index=pd.date_range("2014-01-01", periods=40, freq="15min"),
        )
        second = pd.DataFrame(
            {"A": np.arange(40)},
            index=pd.date_range("2014-02-01", periods=40, freq="15min"),
        )

        batch = build_window_batch([first, second], input_steps=16, output_steps=16)

        self.assertEqual(batch["X"].shape[0], 18)
        self.assertFalse(any(
            pd.Timestamp(start).month != pd.Timestamp(end).month
            for start, end in zip(batch["x_start_times"], batch["y_end_times"])
        ))


if __name__ == "__main__":
    unittest.main()
