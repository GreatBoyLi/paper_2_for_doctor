import unittest

from pipeline.run_source_data_pipeline import STEPS as SOURCE_STEPS
from pipeline.run_target_data_pipeline import STEPS as TARGET_STEPS


class DataPipelineEntrypointsTests(unittest.TestCase):
    def test_source_pipeline_has_the_four_required_steps_in_order(self):
        self.assertEqual(
            [step.script.name for step in SOURCE_STEPS],
            ["source_data.py", "prepare_source_2014.py", "prepare_source_dataset.py", "build_source_graph.py"],
        )

    def test_target_pipeline_has_the_four_required_steps_in_order(self):
        self.assertEqual(
            [step.script.name for step in TARGET_STEPS],
            ["prepare_target_data.py", "extract_target_period.py", "prepare_target_dataset.py", "build_target_graph.py"],
        )

    def test_every_configured_script_exists(self):
        for step in SOURCE_STEPS + TARGET_STEPS:
            self.assertTrue(step.script.is_file(), str(step.script))
