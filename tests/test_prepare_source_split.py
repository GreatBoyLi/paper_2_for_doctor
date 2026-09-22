from pathlib import Path
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_FILE = PROJECT_DIR / "dataset/prepare_source_dataset.py"


class PrepareSourceSplitTests(unittest.TestCase):
    def test_source_preparation_uses_single_root_level_split(self):
        text = SCRIPT_FILE.read_text(encoding="utf-8")

        self.assertNotIn("FOLD_CONFIGS", text)
        self.assertNotIn("fold_dir", text)
        self.assertIn("split_source_months", text)
        self.assertIn('"SOURCE_SPLIT_SUMMARY.csv"', text)
        self.assertIn('OUTPUT_DIR / "train.npz"', text)
        self.assertIn('OUTPUT_DIR / "val.npz"', text)


if __name__ == "__main__":
    unittest.main()
