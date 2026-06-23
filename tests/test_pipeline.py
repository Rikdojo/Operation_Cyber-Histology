import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "Code"
sys.path.insert(0, str(CODE_DIR))

from data import get_loaders
from train import build_model, load_config, prepare_experiment
from trainer import Trainer, macro_classification_scores


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CODE_DIR / "config.json")

    def test_config_has_required_datasets_and_models(self):
        self.assertEqual(set(self.config["DATASETS"]), {"cells", "chest", "lesions", "orgs"})
        self.assertEqual(set(self.config["MODELS"]), {"AlexNet", "VGG16", "ResNet18"})

        for dataset_config in self.config["DATASETS"].values():
            self.assertGreater(dataset_config["channels"], 0)
            self.assertGreater(dataset_config["num_classes"], 1)

    def test_models_return_expected_output_shape(self):
        for dataset_name, dataset_config in self.config["DATASETS"].items():
            x = torch.randn(2, dataset_config["channels"], 64, 64)

            for model_name in self.config["MODELS"]:
                model = build_model(model_name, dataset_name, self.config)
                model.eval()
                with torch.no_grad():
                    output = model(x)

                self.assertEqual(output.shape, (2, dataset_config["num_classes"]))

    def test_metric_calculation(self):
        labels = torch.tensor([0, 1, 1, 2])
        predictions = torch.tensor([0, 1, 0, 2])
        scores = macro_classification_scores(labels, predictions, num_classes=3)

        self.assertAlmostEqual(scores["accuracy"], 75.0)
        self.assertGreater(scores["macro_f1"], 0)
        self.assertLessEqual(scores["macro_f1"], 100)

    def test_tiny_training_and_prediction_loop(self):
        device = torch.device("cpu")
        model = build_model("AlexNet", "chest", self.config).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        trainer = Trainer(model, criterion, optimizer, device)

        x = torch.randn(6, 1, 64, 64)
        y = torch.tensor([0, 1, 2, 0, 1, 2])
        loader = DataLoader(TensorDataset(x, y), batch_size=3)

        train_loss, train_acc = trainer.train_one_epoch(loader)
        predictions, labels = trainer.predict(loader)

        self.assertGreater(train_loss, 0)
        self.assertGreaterEqual(train_acc, 0)
        self.assertEqual(predictions.shape, labels.shape)
        self.assertEqual(predictions.numel(), 6)

    def test_data_loader_uses_local_pt_files(self):
        data_dir = PROJECT_ROOT / "data"
        if not (data_dir / "cells.pt").exists():
            self.skipTest("Local data files are not available")

        train_loader, val_loader, test_loader = get_loaders("cells", data_dir, batch_size=4)
        train_images, train_labels = next(iter(train_loader))
        val_images, _ = next(iter(val_loader))
        test_images, _ = next(iter(test_loader))

        self.assertEqual(train_images.shape[1:], (3, 64, 64))
        self.assertEqual(train_labels.ndim, 2)
        self.assertEqual(val_images.shape[1:], (3, 64, 64))
        self.assertEqual(test_images.shape[1:], (3, 64, 64))

    def test_prepare_experiment_builds_all_parts(self):
        data_dir = PROJECT_ROOT / "data"
        if not (data_dir / "chest.pt").exists():
            self.skipTest("Local data files are not available")

        device = torch.device("cpu")
        trainer, train_loader, val_loader, test_loader = prepare_experiment(
            "chest", "AlexNet", self.config, device
        )

        self.assertIsInstance(trainer, Trainer)
        self.assertIsNotNone(next(iter(train_loader)))
        self.assertIsNotNone(next(iter(val_loader)))
        self.assertIsNotNone(next(iter(test_loader)))


if __name__ == "__main__":
    unittest.main()
