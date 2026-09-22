from pathlib import Path
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_FILE = PROJECT_DIR / "dataset/prepare_target_dataset.py"


class PrepareTargetSplitTests(unittest.TestCase):
    def test_target_preparation_uses_five_train_days_and_two_validation_days(self):
        text = SCRIPT_FILE.read_text(encoding="utf-8")

        self.assertNotIn("FOLD_CONFIGS", text)
        self.assertNotIn("fold_dir", text)
        self.assertIn("TRAIN_DAYS = 5", text)
        self.assertIn('"TARGET_SPLIT_SUMMARY.csv"', text)
        self.assertIn('OUTPUT_DIR / "q99_per_station.csv"', text)
        self.assertIn('OUTPUT_DIR / "train.npz"', text)
        self.assertIn('OUTPUT_DIR / "val.npz"', text)


if __name__ == "__main__":
    unittest.main()
