"""Periodic assembly and thickness conversion for TPMS-like surfaces."""

from pathlib import Path
from typing import Iterable
import warnings

import numpy as np
import pyvista as pv

from tpmslike.boundary import read_boundary


def _faces(mesh: pv.PolyData) -> np.ndarray:
    mesh = mesh.triangulate()
    if mesh.n_cells == 0:
        raise ValueError("Surface mesh has no cells")
    raw = np.asarray(mesh.faces, dtype=np.int64)
    cells = raw.reshape(-1, 4)
    if not np.all(cells[:, 0] == 3):
        raise ValueError("Expected a triangulated surface")
    return cells[:, 1:].copy()


def _unique_triangles(mesh: pv.PolyData, area_tolerance: float = 1e-12) -> pv.PolyData:
    """Remove repeated and zero-area triangles without changing coordinates."""
    mesh = mesh.triangulate()
    points = np.asarray(mesh.points, dtype=np.float64)
    if mesh.n_cells == 0:
        return pv.PolyData(points)
    faces = _faces(mesh)
    repeated = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 0] == faces[:, 2])
    a = points[faces[:, 1]] - points[faces[:, 0]]
    b = points[faces[:, 2]] - points[faces[:, 0]]
    areas = 0.5 * np.linalg.norm(np.cross(a, b), axis=1)
    keep = ~repeated & (areas > area_tolerance)
    valid = faces[keep]
    if valid.size:
        canonical = np.sort(valid, axis=1)
        _, first = np.unique(canonical, axis=0, return_index=True)
        valid = valid[np.sort(first)]
    flat = np.column_stack((np.full(len(valid), 3, dtype=np.int64), valid)).reshape(-1)
    return pv.PolyData(points, flat).clean(point_merging=True, tolerance=1e-10)


def thicken_surface(mesh: pv.PolyData, thickness: float, clamp_bounds: float | None = None) -> pv.PolyData:
    """Offset a triangulated surface and connect its open boundary edges."""

    if thickness <= 0:
        raise ValueError("Thickness must be positive")
    mesh = _unique_triangles(mesh).compute_normals(
        point_normals=True, cell_normals=False, consistent_normals=True,
        auto_orient_normals=True, inplace=False
    )
    faces = _faces(mesh)
    normals = np.asarray(mesh.point_data["Normals"], dtype=np.float64)
    half = float(thickness) / 2.0
    outer = mesh.points + half * normals
    inner = mesh.points - half * normals
    if clamp_bounds is not None:
        # Keep the surface normal at box faces.  Zeroing those components makes
        # the two offsets coincide along box edges and leaves slit-like open
        # boundaries after the side-wall triangles are filtered.  Clipping the
        # offsets is sufficient to keep the shell inside the requested cell.
        outer = np.clip(outer, -clamp_bounds, clamp_bounds)
        inner = np.clip(inner, -clamp_bounds, clamp_bounds)
    # A triangle that lies on a box edge can become collinear after clamping
    # both offsets to the cell bounds.  Remove it before finding boundary
    # edges; the newly exposed edge is then closed by the side-wall quads.
    outer_a = outer[faces[:, 1]] - outer[faces[:, 0]]
    outer_b = outer[faces[:, 2]] - outer[faces[:, 0]]
    inner_a = inner[faces[:, 1]] - inner[faces[:, 0]]
    inner_b = inner[faces[:, 2]] - inner[faces[:, 0]]
    outer_area = 0.5 * np.linalg.norm(np.cross(outer_a, outer_b), axis=1)
    inner_area = 0.5 * np.linalg.norm(np.cross(inner_a, inner_b), axis=1)
    valid = (outer_area > 1e-12) & (inner_area > 1e-12)
    faces = faces[valid]
    points = np.vstack([outer, inner])
    n = mesh.n_points
    outer_faces = faces
    inner_faces = faces[:, ::-1] + n

    # Close boundaries when the input is an open patch. Each undirected edge
    # used once belongs to the boundary and receives two triangles.
    edge_counts = {}
    for tri in faces:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = tuple(sorted((int(a), int(b))))
            edge_counts[key] = edge_counts.get(key, 0) + 1
    side_faces = []
    for (a, b), count in edge_counts.items():
        if count == 1:
            side_faces.append([a, b, b + n])
            side_faces.append([a, b + n, a + n])

    cells = [np.r_[3, face] for face in outer_faces]
    cells.extend(np.r_[3, face] for face in inner_faces)
    cells.extend(np.r_[3, face] for face in side_faces)
    flat_cells = np.concatenate([np.asarray(cell, dtype=np.int64) for cell in cells])
    result = _unique_triangles(pv.PolyData(points, flat_cells))
    return _repair_float32_slivers(result, clamp_bounds=clamp_bounds)


def _transform(mesh: pv.PolyData, matrix: np.ndarray, offset: np.ndarray) -> pv.PolyData:
    points = np.asarray(mesh.points) @ matrix.T + offset
    faces = mesh.faces.reshape(-1, 4).copy()
    if np.linalg.det(matrix) < 0:
        faces[:, 1:] = faces[:, 1:][:, ::-1]
    return pv.PolyData(points, faces.reshape(-1))


def _merge(parts: Iterable[pv.PolyData]) -> pv.PolyData:
    """Merge surface parts using the same point weld as the legacy code."""
    parts = list(parts)
    if not parts:
        raise ValueError("At least one mesh part is required")
    points = []
    cells = []
    point_offset = 0
    for part in parts:
        # Reconstruct to discard stale VTK cell arrays retained by filters.
        part = pv.PolyData(np.asarray(part.points), np.asarray(part.faces, dtype=np.int64).copy())
        points.append(np.asarray(part.points))
        faces = np.asarray(part.faces, dtype=np.int64).copy()
        if faces.size:
            cursor = 0
            while cursor < faces.size:
                count = int(faces[cursor])
                cells.extend([count, *(faces[cursor + 1:cursor + 1 + count] + point_offset)])
                cursor += count + 1
        point_offset += part.n_points
    raw = pv.PolyData(np.vstack(points), np.asarray(cells, dtype=np.int64))
    cleaned = raw.clean(point_merging=True, tolerance=1e-8)
    return _unique_triangles(cleaned)


def _boundary_vertices(mesh: pv.PolyData) -> np.ndarray:
    """Return point ids incident to exactly one triangle edge."""
    faces = _faces(mesh)
    edges = np.sort(np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique[counts == 1]
    if boundary_edges.size == 0:
        return np.empty(0, dtype=np.int64)
    return np.unique(boundary_edges.reshape(-1))


def enforce_boundary(mesh: pv.PolyData, boundary: str | Path | None,
                     tolerance: float = 1e-5,
                     max_snap_distance: float | None = None) -> pv.PolyData:
    """Snap extracted boundary vertices to the requested Hermite samples.

    Marching cubes places points on a cube face within floating point error.
    The original postprocessor replaced those points with the exact samples
    from the boundary file before reflecting the patch.  Repeating that step
    prevents small gaps from becoming visible seams after periodic assembly.
    """
    if boundary is None:
        return mesh
    target = read_boundary(boundary).points.astype(np.float64)
    if target.size == 0:
        return mesh
    result = mesh.copy(deep=True)
    ids = _boundary_vertices(result)
    for point_id in ids:
        point = result.points[point_id].astype(np.float64)
        compatible = np.ones(len(target), dtype=bool)
        for axis in range(3):
            on_face = np.isclose(abs(point[axis]), 1.0, atol=tolerance)
            if on_face:
                compatible &= np.isclose(target[:, axis], point[axis], atol=tolerance)
        candidates = np.flatnonzero(compatible)
        if candidates.size:
            distances = np.linalg.norm(target[candidates] - point, axis=1)
            nearest = int(np.argmin(distances))
            if max_snap_distance is None or distances[nearest] <= max_snap_distance:
                result.points[point_id] = target[candidates[nearest]]
    return result


def _clip_cell(mesh: pv.PolyData, half_size: float = 2.0) -> pv.PolyData:
    """Clip the assembled four-unit span to its central periodic cell."""
    bounds = (-half_size, half_size, -half_size, half_size, -half_size, half_size)
    clipped = mesh.clip_box(bounds, invert=False)
    return _unique_triangles(clipped.extract_surface())


def _drop_degenerate(mesh: pv.PolyData, area_tolerance: float = 1e-12) -> pv.PolyData:
    """Remove zero-area triangles introduced where a clipped seam coincides."""
    return _unique_triangles(mesh, area_tolerance=area_tolerance)


def _repair_float32_slivers(mesh: pv.PolyData, clamp_bounds: float | None,
                            area_tolerance: float = 1e-12) -> pv.PolyData:
    """Separate boundary slivers that collapse in PLY's float32 coordinates.

    VTK's PLY writer stores points as float32.  A handful of triangles can be
    non-zero in the in-memory float64 mesh but lose their area through
    cancellation after that conversion.  Move one low-valence boundary point
    inward by a sub-voxel amount so serialization preserves the closed shell.
    """
    if clamp_bounds is None or mesh.n_cells == 0:
        return mesh
    faces = _faces(mesh)
    points = np.asarray(mesh.points, dtype=np.float64).copy()
    points32 = points.astype(np.float32)
    area64 = 0.5 * np.linalg.norm(
        np.cross(points[faces[:, 1]] - points[faces[:, 0]],
                 points[faces[:, 2]] - points[faces[:, 0]]), axis=1
    )
    # Keep the cross-product in float32 as well: cancellation in this step is
    # what the VTK PLY reader will reproduce for very thin boundary slivers.
    area32 = 0.5 * np.linalg.norm(
        np.cross(points32[faces[:, 1]] - points32[faces[:, 0]],
                 points32[faces[:, 2]] - points32[faces[:, 0]]), axis=1
    )
    candidates = np.flatnonzero((area64 > area_tolerance) & (area32 <= area_tolerance))
    if candidates.size == 0:
        return mesh
    valence = np.bincount(faces.reshape(-1), minlength=mesh.n_points)
    for cell_id in candidates:
        tri = faces[cell_id]
        spread = np.ptp(points[tri], axis=0)
        boundary_axes = [
            axis for axis in np.flatnonzero(spread <= 1e-7)
            if np.isclose(abs(float(points[tri, axis].mean())), clamp_bounds, atol=1e-5)
        ]
        axes = boundary_axes or list(np.argsort(spread))
        # Try each vertex/axis pair and keep the smallest inward move that
        # survives float32 cross-product arithmetic.  Choosing a fixed vertex
        # is insufficient when the sliver's two nearly parallel edges share it.
        repaired = False
        for vertex in tri:
            for axis in axes:
                sign = float(np.sign(points[tri, axis].mean())) or 1.0
                trial = np.asarray(points[vertex], dtype=np.float64).copy()
                trial[axis] = np.clip(trial[axis] - sign * 1e-4,
                                      -clamp_bounds, clamp_bounds)
                trial_points = points32[tri].copy()
                trial_points[np.flatnonzero(tri == vertex)[0]] = trial.astype(np.float32)
                trial_a = trial_points[1] - trial_points[0]
                trial_b = trial_points[2] - trial_points[0]
                trial_area = 0.5 * np.linalg.norm(np.cross(trial_a, trial_b))
                if trial_area > area_tolerance:
                    points[vertex] = trial
                    points32[vertex] = trial.astype(np.float32)
                    repaired = True
                    break
            if repaired:
                break
    return pv.PolyData(points, np.asarray(mesh.faces, dtype=np.int64).copy())


def _component_count(mesh: pv.PolyData) -> int:
    """Count cell-connected components without treating point contacts as joins."""
    if mesh.n_cells == 0:
        return 0
    connected = mesh.connectivity()
    labels = connected.cell_data.get("RegionId")
    return int(np.unique(labels).size) if labels is not None else 1


def assemble_periodic(mesh: pv.PolyData, symmetry: str, copies: int = 1) -> pv.PolyData:
    """Assemble a legacy-compatible four-unit periodic surface span.

    Deep Currents predicts the complete ``[-1, 1]^3`` patch.  The paper's
    postprocessor reflects it across the negative x/y faces and the positive z
    face for ``rrr``; ``ttt`` uses translations by one cell.  These planes are
    part of the geometry convention and must not be replaced by reflections
    around the origin.
    """

    if copies < 1:
        raise ValueError("copies must be at least 1")
    mesh = _unique_triangles(mesh.extract_surface())
    parts = [mesh]
    if symmetry == "rrr":
        reflected_x = _transform(mesh, np.diag([-1.0, 1.0, 1.0]), np.array([-2.0, 0.0, 0.0]))
        first = _merge((mesh, reflected_x))
        reflected_y = _transform(first, np.diag([1.0, -1.0, 1.0]), np.array([0.0, -2.0, 0.0]))
        second = _merge((first, reflected_y))
        reflected_z = _transform(second, np.diag([1.0, 1.0, -1.0]), np.array([0.0, 0.0, 2.0]))
        assembled = _merge((second, reflected_z))
        assembled.points += np.array([1.0, 1.0, -1.0])
        parts = [assembled]
    elif symmetry == "ttt":
        first = _merge((mesh, _transform(mesh, np.eye(3), np.array([2.0, 0.0, 0.0]))))
        second = _merge((first, _transform(first, np.eye(3), np.array([0.0, 2.0, 0.0]))))
        assembled = _merge((second, _transform(second, np.eye(3), np.array([0.0, 0.0, 2.0]))))
        assembled.points += np.array([-1.0, -1.0, -1.0])
        parts = [assembled]
    elif symmetry != "none":
        raise ValueError(f"Unknown symmetry: {symmetry}")
    if copies == 1 or symmetry == "none":
        return _merge(parts)
    # The postprocessor used a 3x3x3 neighbourhood before clipping.  A span
    # assembled above has period four in these coordinates, so additional
    # neighbourhoods are translated by four units.
    base = parts[0]
    neighbours = []
    for ix in range(-(copies - 1), copies):
        for iy in range(-(copies - 1), copies):
            for iz in range(-(copies - 1), copies):
                neighbours.append(_transform(base, np.eye(3), np.array([4.0 * ix, 4.0 * iy, 4.0 * iz])))
    return _merge(neighbours)


def build_lattice(surface: str | Path, symmetry: str = "none", thickness: float = 0.08,
                  copies: int = 1, boundary: str | Path | None = None,
                  snap_boundary: bool = False,
                  snap_tolerance: float = 0.08) -> pv.PolyData:
    """Read, assemble, and thicken a surface mesh.

    For ``rrr``/``ttt`` the legacy pipeline clips the assembled span to
    ``[-2, 2]^3``, scales it to ``[-1, 1]^3``, then offsets by half the
    requested shell thickness.  The order is significant at periodic seams.
    """

    if thickness <= 0:
        raise ValueError("Thickness must be positive")
    mesh = pv.read(surface).extract_surface(algorithm="dataset_surface").triangulate().clean()
    # MC boundary vertices are only approximately on the Hermite curves.  An
    # unconditional nearest-point replacement can move a vertex by a large
    # fraction of a cell and collapse triangles (especially for topologies 1
    # and 6).  Boundary snapping is therefore opt-in and distance-limited.
    if snap_boundary and boundary is not None:
        mesh = enforce_boundary(mesh, boundary, max_snap_distance=snap_tolerance)
    if symmetry in {"rrr", "ttt"}:
        mesh = _clip_cell(assemble_periodic(mesh, symmetry=symmetry, copies=copies))
        if mesh.n_cells:
            components = _component_count(mesh)
            if components > 1:
                warnings.warn(
                    f"Surface contains {components} disconnected components before "
                    "periodic cleanup; inspect the implicit-field level before "
                    "using the largest component.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            mesh = mesh.extract_largest().triangulate().clean()
            mesh = _drop_degenerate(mesh)
        mesh.points *= 0.5
        shell = thicken_surface(mesh, thickness, clamp_bounds=1.0)
        return _drop_degenerate(shell)
    return thicken_surface(mesh, thickness)
