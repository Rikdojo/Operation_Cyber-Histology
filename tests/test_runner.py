import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "Code"
sys.path.insert(0, str(CODE_DIR))

from runner import get_device, get_memory_usage_mb, start_memory_measurement


class RunnerUtilityTests(unittest.TestCase):
    def test_get_device_returns_supported_torch_device(self):
        device = get_device()
        self.assertIn(device.type, {"cuda", "mps", "cpu"})

    def test_cpu_memory_tracking_falls_back_clearly(self):
        device = torch.device("cpu")
        start_memory_measurement(device)
        memory_mb, memory_type = get_memory_usage_mb(device)
        self.assertIsNone(memory_mb)
        self.assertEqual(memory_type, "not_available_on_cpu")


if __name__ == "__main__":
    unittest.main()
