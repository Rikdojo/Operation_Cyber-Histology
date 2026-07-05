import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "Code"
sys.path.insert(0, str(CODE_DIR))

from data import get_loaders


class DataLoaderTests(unittest.TestCase):
    def test_get_loaders_splits_and_normalizes_synthetic_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)
            torch.save(
                {
                    "train_images": torch.randn(10, 3, 64, 64),
                    "train_labels": torch.randint(0, 8, (10,)),
                    "test_images": torch.randn(4, 3, 64, 64),
                    "test_labels": torch.randint(0, 8, (4,)),
                },
                data_path / "cells.pt",
            )

            train_loader, val_loader, test_loader = get_loaders(
                data="cells",
                data_path=data_path,
                batch_size=2,
                val_split=0.2,
                seed=42,
            )

            self.assertEqual(len(train_loader.dataset), 8)
            self.assertEqual(len(val_loader.dataset), 2)
            self.assertEqual(len(test_loader.dataset), 4)

            images, labels = next(iter(train_loader))
            self.assertEqual(images.shape, (2, 3, 64, 64))
            self.assertEqual(labels.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
