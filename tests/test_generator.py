import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from hermite.main import _closed_cycles, generate_boundary_file
from tpmslike.boundary import read_boundary
from deepcurrents import _global
from deepcurrents.train import group_bdry_verts_rrr


class HermiteGeneratorTest(unittest.TestCase):
    def test_cycle_preserves_obj_direction(self):
        edges = [
            np.asarray([5, 1]), np.asarray([1, 2]), np.asarray([2, 6]),
            np.asarray([6, 9]), np.asarray([9, 8]), np.asarray([8, 4]),
            np.asarray([4, 3]), np.asarray([3, 0]), np.asarray([0, 5]),
        ]
        np.testing.assert_array_equal(
            _closed_cycles(edges)[0], np.asarray([5, 1, 2, 6, 9, 8, 4, 3, 0])
        )

    def test_topology_one_matches_checked_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "boundary.txt"
            generate_boundary_file(Path("examples/topologies/1.obj"), output)
            boundary = read_boundary(output)
        self.assertEqual(boundary.n_curves, 2)
        self.assertEqual([curve.shape[0] for curve in boundary.curves], [90, 90])

    def test_rrr_face_indices_do_not_pad_short_faces(self):
        boundary = read_boundary("examples/boundaries/topo_1.txt")
        _global.bdry_d = 1
        ids = group_bdry_verts_rrr(torch.from_numpy(boundary.points))
        self.assertEqual([int(face.numel()) for face in ids], [42, 22, 22, 42, 22, 42])


if __name__ == "__main__":
    unittest.main()
