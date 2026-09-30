"""Run topology -> boundary -> Deep Currents -> shell as one reproducible job."""

from pathlib import Path
import argparse
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hermite.main import generate_boundary_file
from postprocessing.shell import build_lattice


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--type", choices=("rrr", "ttt", "ttt_a"), default="rrr")
    parser.add_argument("--vzoom", type=float, default=2.5)
    parser.add_argument("--iterations", type=int, default=100000)
    parser.add_argument("--n-samples", type=int, default=4096)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--grid-size", type=float, default=0.04)
    parser.add_argument("--thickness", type=float, default=0.08)
    parser.add_argument("--boundary", type=Path, default=None,
                        help="Optional exact boundary file used to snap MC face vertices")
    parser.add_argument("--snap-boundary", action="store_true",
                        help="Snap only MC boundary vertices within --snap-tolerance")
    parser.add_argument("--snap-tolerance", type=float, default=0.08)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    boundary = args.boundary or (args.run_dir / "boundary.txt")
    if args.boundary is None:
        generate_boundary_file(args.topology, boundary, vzoom=args.vzoom, samples=args.samples)
    elif not boundary.exists():
        raise FileNotFoundError(f"Boundary file does not exist: {boundary}")

    train_cmd = [
        sys.executable, "-m", "deepcurrents.train",
        "--boundary", str(boundary), "--type", args.type,
        "--index", "0", "--out", str(args.run_dir / "surface"),
        "--n_iterations", str(args.iterations), "--grid-size", str(args.grid_size),
        "--n_samples", str(args.n_samples),
    ]
    if args.device:
        train_cmd.extend(["--device", args.device])
    subprocess.run(train_cmd, check=True)

    surface = args.run_dir / "surface" / f"0_{args.type}_or_ms_average.obj"
    symmetry = "rrr" if args.type == "rrr" else "ttt"
    copies = 1 if symmetry == "rrr" else 2
    shell = build_lattice(surface, symmetry=symmetry, thickness=args.thickness,
                          copies=copies, boundary=boundary,
                          snap_boundary=args.snap_boundary,
                          snap_tolerance=args.snap_tolerance)
    shell_path = args.run_dir / "shell.ply"
    shell.save(shell_path)
    print(f"Wrote {shell_path}")


if __name__ == "__main__":
    main()
