import json
import sys
import unittest
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "Code"
sys.path.insert(0, str(CODE_DIR))

from train import get_run_list, get_task2_model_name, load_config, make_result_row


def load_json_without_duplicate_keys(path):
    def reject_duplicates(pairs):
        counts = Counter(key for key, _ in pairs)
        duplicates = [key for key, count in counts.items() if count > 1]
        if duplicates:
            raise ValueError(f"Duplicate JSON keys: {duplicates}")
        return dict(pairs)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f, object_pairs_hook=reject_duplicates)


class ConfigAndRunListTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def test_config_has_no_duplicate_keys(self):
        config = load_json_without_duplicate_keys(CODE_DIR / "config.json")
        self.assertIn("DATASETS", config)
        self.assertIn("task2", config)
        self.assertIn("task3", config)

    def test_task1_single_run_uses_selected_dataset_and_model(self):
        runs = get_run_list(self.config, "task1", run_all=False)
        self.assertEqual(runs, [(self.config["DATA"], self.config["MODEL"], None)])

    def test_task1_run_all_covers_main_benchmark_matrix(self):
        runs = get_run_list(self.config, "task1", run_all=True)
        expected = len(self.config["DATASETS"]) * len(self.config["MODELS"])
        self.assertEqual(len(runs), expected)
        self.assertIn(("cells", "AlexNet", None), runs)
        self.assertIn(("orgs", "ResNet18", None), runs)

    def test_task2_run_all_covers_baseline_and_light_models(self):
        runs = get_run_list(self.config, "task2", run_all=True)
        modes = {mode for _, _, mode in runs}
        expected = (
            len(self.config["DATASETS"])
            * len(self.config["task2"]["MODELS"])
            * len(self.config["task2"]["MODES"])
        )
        self.assertEqual(len(runs), expected)
        self.assertEqual(modes, {"Baseline", "Light"})
        self.assertIn(("cells", "AlexNet", "Baseline"), runs)
        self.assertIn(("cells", "AlexNet", "Light"), runs)

    def test_task2_single_run_is_baseline_debug_run(self):
        runs = get_run_list(self.config, "task2", run_all=False)
        self.assertEqual(runs, [(self.config["DATA"], self.config["MODEL"], "Baseline")])

    def test_task2_model_names_resolve_baseline_and_light(self):
        self.assertEqual(get_task2_model_name("AlexNet", "Baseline"), "AlexNet")
        self.assertEqual(get_task2_model_name("AlexNet", "Light"), "Light_AlexNet")

    def test_result_row_contains_efficiency_fields(self):
        metrics = {
            "num_params": 100,
            "total_params": 120,
            "training_time": 1.5,
            "peak_train_memory": None,
            "train_memory_type": "not_available_on_cpu",
            "accuracy": 0.8,
            "precision": 0.7,
            "recall": 0.6,
            "macro_f1": 0.65,
            "inference_time": 0.2,
            "peak_inference_memory": None,
            "inference_memory_type": "not_available_on_cpu",
            "inference_latency_per_sample": 0.01,
        }
        row = make_result_row("cells", "Light_AlexNet", "Light", self.config, metrics)
        expected_fields = {
            "dataset",
            "model",
            "mode",
            "trainable_parameters",
            "total_parameters",
            "training_time_seconds",
            "peak_training_memory_mb",
            "training_memory_type",
            "inference_time_seconds",
            "peak_inference_memory_mb",
            "inference_memory_type",
            "inference_latency_per_sample",
            "accuracy",
            "precision",
            "recall",
            "macro_f1",
        }
        self.assertTrue(expected_fields.issubset(row.keys()))
        self.assertEqual(row["mode"], "Light")
        self.assertEqual(row["accuracy"], 80.0)

    def test_task3_uses_scarce_organs_target(self):
        runs = get_run_list(self.config, "task3", run_all=False)
        target_names = {data_name for data_name, _, _ in runs}
        self.assertEqual(target_names, {"organs"})
        self.assertEqual({mode for _, _, mode in runs}, set(self.config["task3"]["MODES"]))


if __name__ == "__main__":
    unittest.main()
