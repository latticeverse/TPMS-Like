"""Generate cubic-Hermite boundary curves from topology graph OBJ files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

from tpmslike.boundary import write_boundary
from tpmslike.obj import read_polyline_obj


def generate_hermite_curve(step: float, p0, p1, d0, d1) -> np.ndarray:
    """Sample one cubic Hermite segment on ``[0, 1)``."""
    p0 = np.asarray(p0, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)
    d0 = np.asarray(d0, dtype=np.float32)
    d1 = np.asarray(d1, dtype=np.float32)
    if np.allclose(p0, p1):
        return np.repeat(p0[None], max(1, int(round(1.0 / step))), axis=0)
    t = np.arange(0.0, 1.0, step, dtype=np.float32)
    t2 = t * t
    t3 = t2 * t
    return (
        (1 - 3 * t2 + 2 * t3)[:, None] * p0
        + (3 * t2 - 2 * t3)[:, None] * p1
        + (t - 2 * t2 + t3)[:, None] * d0
        + (-t2 + t3)[:, None] * d1
    ).astype(np.float32)


def _closed_cycles(edges: Iterable[np.ndarray]) -> list[np.ndarray]:
    """Convert an undirected degree-two graph into ordered closed cycles."""
    adjacency: dict[int, list[int]] = {}
    edge_records = []
    for edge in edges:
        if len(edge) != 2:
            raise ValueError("Topology OBJ line elements must contain two vertices")
        a, b = (int(edge[0]), int(edge[1]))
        edge_id = len(edge_records)
        adjacency.setdefault(a, []).append((b, edge_id))
        adjacency.setdefault(b, []).append((a, edge_id))
        edge_records.append((a, b))
    visited_edges: set[int] = set()
    cycles: list[np.ndarray] = []
    for edge_id, (start, first_next) in enumerate(edge_records):
        if edge_id in visited_edges:
            continue
        cycle = [start]
        visited_edges.add(edge_id)
        previous, current = start, first_next
        while current != start:
            cycle.append(current)
            choices = [(node, candidate_id) for node, candidate_id in adjacency[current]
                       if candidate_id not in visited_edges and node != previous]
            if not choices:
                raise ValueError("Topology graph contains an open boundary")
            next_vertex, next_edge = choices[0]
            visited_edges.add(next_edge)
            previous, current = current, next_vertex
        cycles.append(np.asarray(cycle, dtype=np.int64))
    return cycles


def _node_derivatives(points: np.ndarray, cycles: list[np.ndarray], vzoom: float) -> np.ndarray:
    derivatives = np.zeros((len(points), 2, 3), dtype=np.float32)
    for cycle in cycles:
        for index, left in enumerate(cycle):
            right = cycle[(index + 1) % len(cycle)]
            direction = points[right] - points[left]
            zero_count = int(np.count_nonzero(np.abs(direction) <= 1e-6))
            if zero_count == 2:
                derivatives[left, 0] = direction / 2.0
                derivatives[right, 1] = direction / 2.0
            else:
                if abs(direction[0]) <= 1e-6:
                    if abs(points[left, 1]) == 1:
                        derivatives[left, 0, 1] = direction[1]
                        derivatives[right, 1, 2] = direction[2]
                    else:
                        derivatives[left, 0, 2] = direction[2]
                        derivatives[right, 1, 1] = direction[1]
                if abs(direction[1]) <= 1e-6:
                    if abs(points[left, 0]) == 1:
                        derivatives[left, 0, 0] = direction[0]
                        derivatives[right, 1, 2] = direction[2]
                    else:
                        derivatives[left, 0, 2] = direction[2]
                        derivatives[right, 1, 0] = direction[0]
                if abs(direction[2]) <= 1e-6:
                    if abs(points[left, 0]) == 1:
                        derivatives[left, 0, 0] = direction[0]
                        derivatives[right, 1, 1] = direction[1]
                    else:
                        derivatives[left, 0, 1] = direction[1]
                        derivatives[right, 1, 0] = direction[0]
    return derivatives * float(vzoom)


def _sample_cycles(points: np.ndarray, cycles: list[np.ndarray], derivatives: np.ndarray, topology_id: str) -> list[np.ndarray]:
    curves = []
    topology_id = str(topology_id).removeprefix("topo_")
    for cycle in cycles:
        if topology_id == "1":
            total = 30 if len(cycle) == 3 else 10
        elif topology_id == "12":
            total = 40 if len(cycle) == 3 else 30
        elif topology_id == "16":
            total = 20 if len(cycle) == 3 else 12
        else:
            total = 20 if len(cycle) != 3 else 40
        step = 1.0 / total
        segments = []
        for index, left in enumerate(cycle):
            right = cycle[(index + 1) % len(cycle)]
            segments.append(generate_hermite_curve(
                step, points[left], points[right], derivatives[left, 0], derivatives[right, 1]
            ))
        curves.append(np.concatenate(segments, axis=0))
    return curves


def generate_boundary_file(topology_path: str | Path, output_path: str | Path, vzoom: float = 2.5, samples: int | None = None):
    """Generate a canonical boundary text file from a topology OBJ."""
    topology_path = Path(topology_path)
    output_path = Path(output_path)
    points, edges = read_polyline_obj(topology_path)
    cycles = _closed_cycles(edges)
    if topology_path.stem.removeprefix("topo_") == "11" and cycles:
        # Legacy code flips the topology node order before computing
        # derivatives, not the sampled coordinates after the fact.
        cycles[0] = cycles[0][::-1].copy()
    derivatives = _node_derivatives(points, cycles, vzoom)
    curves = _sample_cycles(points, cycles, derivatives, topology_path.stem)
    if samples is not None:
        if samples < 3:
            raise ValueError("samples must be at least 3")
        uniform = []
        for curve in curves:
            closed = np.vstack([curve, curve[0]])
            distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
            targets = np.linspace(0.0, distance[-1], int(samples), endpoint=False)
            uniform.append(np.column_stack([
                np.interp(targets, distance, closed[:, axis]) for axis in range(3)
            ]).astype(np.float32))
        curves = uniform
    write_boundary(output_path, curves)
    return curves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_path", type=Path, default=Path("examples/topologies"))
    parser.add_argument("--topoID", type=str, default="1")
    parser.add_argument("--Vzoom", type=float, default=2.5)
    parser.add_argument("--output_path", type=Path, default=Path("runs/boundaries"))
    parser.add_argument("--samples", type=int, default=None)
    args = parser.parse_args()
    output = args.output_path / f"{args.topoID}.txt"
    generate_boundary_file(args.input_path / f"{args.topoID}.obj", output, args.Vzoom, args.samples)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
