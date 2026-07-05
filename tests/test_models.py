import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "Code"
sys.path.insert(0, str(CODE_DIR))

import models
from train import load_config


class ModelShapeTests(unittest.TestCase):
    def test_baseline_models_match_dataset_class_counts(self):
        config = load_config()
        for dataset_name, dataset_config in config["DATASETS"].items():
            with self.subTest(dataset=dataset_name):
                x = torch.randn(1, dataset_config["channels"], 64, 64)
                for model_name in config["MODELS"]:
                    model_class = getattr(models, model_name)
                    model = model_class(
                        in_channels=dataset_config["channels"],
                        num_classes=dataset_config["num_classes"],
                    )
                    model.eval()
                    with torch.no_grad():
                        output = model(x)
                    self.assertEqual(output.shape, (1, dataset_config["num_classes"]))

    def test_light_models_match_dataset_class_counts(self):
        config = load_config()
        for dataset_name, dataset_config in config["DATASETS"].items():
            with self.subTest(dataset=dataset_name):
                x = torch.randn(1, dataset_config["channels"], 64, 64)
                for model_name in config["task2"]["MODELS"]:
                    light_model_name = f"Light_{model_name}"
                    model_class = getattr(models, light_model_name)
                    model = model_class(
                        in_channels=dataset_config["channels"],
                        num_classes=dataset_config["num_classes"],
                    )
                    model.eval()
                    with torch.no_grad():
                        output = model(x)
                    self.assertEqual(output.shape, (1, dataset_config["num_classes"]))


if __name__ == "__main__":
    unittest.main()
