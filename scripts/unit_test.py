import unittest
import numpy as np
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from glacier_mapping.data.slice import add_index, compute_dems


class TestSliceFunctions(unittest.TestCase):
    def test_add_index(self):
        """Test the add_index function."""
        print("=== Test: add_index Function ===")
        # Create a sample 4-band array
        tiff_np = np.ones((10, 10, 4), dtype=np.float32)
        tiff_np[..., 0] = 2  # Band 1
        tiff_np[..., 1] = 4  # Band 2

        # Test NDVI-like index (B2 - B1) / (B2 + B1)
        result = add_index(tiff_np, index1=1, index2=0)
        expected_index = (4 - 2) / (4 + 2)  # 2 / 6 = 0.333...

        self.assertEqual(result.shape, (10, 10, 5))
        self.assertTrue(np.allclose(result[..., 4], expected_index))
        print("  ✓ Correctly calculates spectral index.")

        # Test division by zero
        tiff_np[..., 0] = -4
        result = add_index(tiff_np, index1=1, index2=0)
        self.assertFalse(np.isnan(result).any())
        print("  ✓ Handles division by zero gracefully.")

    def test_compute_dems(self):
        """Test the compute_dems function."""
        print("=== Test: compute_dems Function ===")
        # Create a sample 2-band DEM array
        dem_np = np.zeros((10, 10, 2), dtype=np.float32)
        dem_np[..., 0] = 1000  # Elevation
        dem_np[..., 1] = 30  # Slope

        result = compute_dems(dem_np)

        self.assertEqual(result.shape, (10, 10, 2))
        self.assertTrue(np.all(result[..., 0] == 1000))
        self.assertTrue(np.all(result[..., 1] == 30))
        print("  ✓ Correctly extracts elevation and slope.")


if __name__ == "__main__":
    unittest.main()
