"""Read and write the boundary format shared by all pipeline stages.

The text format is intentionally simple and stable:

* line 1: number of closed boundary curves;
* remaining lines: one scalar coordinate per line, grouped as ``x, y, z``.

The parser also accepts whitespace separated triples, which makes hand-written
inputs less error prone while remaining compatible with the original files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np


@dataclass(frozen=True)
class Boundary:
    """A collection of closed polygonal curves.

    ``curves`` has shape ``(n_curves, n_points, 3)``.  Curves may have
    different point counts, so the public representation is a list of arrays.
    """

    curves: List[np.ndarray]

    @property
    def n_curves(self) -> int:
        return len(self.curves)

    @property
    def points(self) -> np.ndarray:
        if not self.curves:
            return np.empty((0, 3), dtype=np.float32)
        return np.concatenate(self.curves, axis=0)


def _numeric_tokens(lines: Iterable[str]) -> List[float]:
    values: List[float] = []
    for line_number, raw in enumerate(lines, start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            values.extend(float(token) for token in line.split())
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value on line {line_number}") from exc
    return values


def read_boundary(path: str | Path) -> Boundary:
    """Load a boundary file and validate its declared curve count."""

    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Boundary file is empty: {path}")
    try:
        n_curves = int(lines[0].split("#", 1)[0].strip())
    except ValueError as exc:
        raise ValueError(f"First line must be an integer curve count: {path}") from exc
    if n_curves <= 0:
        raise ValueError("Boundary curve count must be positive")

    values = _numeric_tokens(lines[1:])
    if len(values) % 3:
        raise ValueError(f"Boundary coordinates are not divisible by 3: {path}")
    points = np.asarray(values, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] < n_curves:
        raise ValueError("Boundary file contains fewer points than curves")

    # Original files do not store per-curve lengths.  They use equal lengths;
    # reject ambiguous inputs instead of silently passing malformed data on.
    if points.shape[0] % n_curves:
        raise ValueError(
            "Boundary format requires the same number of points per curve; "
            f"got {points.shape[0]} points for {n_curves} curves"
        )
    curve_size = points.shape[0] // n_curves
    curves = [points[i * curve_size : (i + 1) * curve_size].copy() for i in range(n_curves)]
    return Boundary(curves)


def write_boundary(path: str | Path, curves: Sequence[np.ndarray]) -> None:
    """Write curves using the canonical one-coordinate-per-line format."""

    curves = [np.asarray(curve, dtype=np.float32).reshape(-1, 3) for curve in curves]
    if not curves:
        raise ValueError("At least one boundary curve is required")
    lengths = {curve.shape[0] for curve in curves}
    if len(lengths) != 1:
        raise ValueError("The legacy boundary format requires equal curve lengths")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f"{len(curves)}\n")
        for curve in curves:
            for point in curve:
                stream.write("{:.9g}\n{:.9g}\n{:.9g}\n".format(*point))
