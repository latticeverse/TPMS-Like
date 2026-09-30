import unittest

import numpy as np
import pyvista as pv

from postprocessing.shell import (
    _drop_degenerate,
    _faces,
    _repair_float32_slivers,
    assemble_periodic,
    thicken_surface,
)


class PeriodicAssemblyTest(unittest.TestCase):
    def test_rrr_uses_boundary_plane_reflections(self):
        mesh = pv.Plane(i_size=2.0, j_size=2.0, i_resolution=2, j_resolution=2).triangulate()
        assembled = assemble_periodic(mesh, "rrr")
        np.testing.assert_allclose(assembled.bounds, (-2, 2, -2, 2, -1, 1), atol=1e-6)

    def test_thickening_removes_clamped_degenerate_faces(self):
        mesh = pv.Plane(i_size=2.0, j_size=2.0, i_resolution=4, j_resolution=4).triangulate()
        # Include a repeated cell; the thickener must remove it before offsetting.
        mesh = pv.PolyData(mesh.points, np.concatenate([mesh.faces, mesh.faces[:4]]))
        shell = thicken_surface(mesh, 0.08, clamp_bounds=1.0)
        faces = _faces(shell)
        vectors_a = shell.points[faces[:, 1]] - shell.points[faces[:, 0]]
        vectors_b = shell.points[faces[:, 2]] - shell.points[faces[:, 0]]
        areas = 0.5 * np.linalg.norm(np.cross(vectors_a, vectors_b), axis=1)
        self.assertTrue(np.all(areas > 1e-12))

    def test_float32_boundary_sliver_is_separated(self):
        # This triangle is valid in float64 but loses its area through the
        # float32 arithmetic used by VTK's PLY reader.
        points = np.array([
            [0.99538045, 1.0, -0.03709658],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, -1.403e-9],
        ])
        mesh = pv.PolyData(points, np.array([3, 0, 1, 2], dtype=np.int64))
        repaired = _repair_float32_slivers(mesh, clamp_bounds=1.0)
        faces = _faces(repaired)
        coords = repaired.points.astype(np.float32)
        vectors_a = coords[faces[:, 1]] - coords[faces[:, 0]]
        vectors_b = coords[faces[:, 2]] - coords[faces[:, 0]]
        area = 0.5 * np.linalg.norm(np.cross(vectors_a, vectors_b), axis=1)
        self.assertTrue(np.all(area > 1e-12))


if __name__ == "__main__":
    unittest.main()
