import tempfile
import unittest
from pathlib import Path

import numpy as np

from tpmslike.boundary import read_boundary, write_boundary


class BoundaryFormatTest(unittest.TestCase):
    def test_round_trip(self):
        curves = [
            np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
            np.asarray([[0, 1, 0], [1, 1, 0]], dtype=np.float32),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "boundary.txt"
            write_boundary(path, curves)
            loaded = read_boundary(path)
        self.assertEqual(loaded.n_curves, 2)
        np.testing.assert_allclose(loaded.curves[1], curves[1])


if __name__ == "__main__":
    unittest.main()
