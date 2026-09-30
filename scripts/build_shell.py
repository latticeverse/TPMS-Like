"""Create a periodic thick shell lattice from a surface mesh."""

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from postprocessing.shell import build_lattice


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--symmetry", choices=("rrr", "ttt", "none"), default="none")
    parser.add_argument("--thickness", type=float, default=0.08)
    parser.add_argument("--copies", type=int, default=1)
    parser.add_argument("--boundary", type=Path, default=None,
                        help="Hermite boundary file (used only with --snap-boundary)")
    parser.add_argument("--snap-boundary", action="store_true",
                        help="Snap only nearby MC boundary vertices")
    parser.add_argument("--snap-tolerance", type=float, default=0.08)
    args = parser.parse_args()
    mesh = build_lattice(args.surface, symmetry=args.symmetry, thickness=args.thickness,
                         copies=args.copies, boundary=args.boundary,
                         snap_boundary=args.snap_boundary,
                         snap_tolerance=args.snap_tolerance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(args.output)
    print(f"Wrote {args.output} ({mesh.n_points} vertices, {mesh.n_cells} cells)")


if __name__ == "__main__":
    main()
