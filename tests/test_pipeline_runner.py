from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from pipeline.runner import PipelineStep, run_pipeline


class PipelineRunnerTests(unittest.TestCase):
    def test_steps_use_current_interpreter_and_script_directory_in_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "one/first.py"
            second = root / "two/second.py"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_text("print('first')", encoding="utf-8")
            second.write_text("print('second')", encoding="utf-8")
            steps = [PipelineStep("第一步", first), PipelineStep("第二步", second)]

            with patch("pipeline.runner.subprocess.run") as run:
                run_pipeline("测试流程", steps, python_executable="test-python")

            self.assertEqual(run.call_count, 2)
            self.assertEqual(run.call_args_list[0].args[0], ["test-python", str(first)])
            self.assertEqual(run.call_args_list[0].kwargs, {"cwd": first.parent, "check": True})
            self.assertEqual(run.call_args_list[1].args[0], ["test-python", str(second)])

    def test_failure_stops_later_steps_and_keeps_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = [root / f"{name}.py" for name in ("one", "two", "three")]
            for script in scripts:
                script.write_text("", encoding="utf-8")
            steps = [PipelineStep(str(index), script) for index, script in enumerate(scripts)]

            failure = subprocess.CalledProcessError(7, ["python", str(scripts[1])])
            with patch("pipeline.runner.subprocess.run", side_effect=[None, failure]) as run:
                with self.assertRaises(SystemExit) as result:
                    run_pipeline("测试流程", steps)

            self.assertEqual(result.exception.code, 7)
            self.assertEqual(run.call_count, 2)
