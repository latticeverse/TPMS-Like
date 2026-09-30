"""Small, dependency-light OBJ readers used for boundary graphs."""

from pathlib import Path
from typing import List, Tuple

import numpy as np


def read_polyline_obj(path: str | Path) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Read vertices and ``l`` line elements from the topology OBJ files."""

    vertices = []
    edges = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        fields = raw.split()
        if not fields:
            continue
        if fields[0] == "v" and len(fields) >= 4:
            vertices.append([float(value) for value in fields[1:4]])
        elif fields[0] in {"l", "line"} and len(fields) >= 3:
            edges.append([int(value.split("/")[0]) - 1 for value in fields[1:]])
    if not vertices or not edges:
        raise ValueError(f"Topology OBJ has no vertices or lines: {path}")
    return np.asarray(vertices, dtype=np.float32), [np.asarray(edge, dtype=np.int64) for edge in edges]
