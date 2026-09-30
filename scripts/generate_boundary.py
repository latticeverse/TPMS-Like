"""Generate a Hermite boundary from a topology OBJ.

Example:
    python scripts/generate_boundary.py --topology examples/topologies/1.obj \
        --output runs/topo_1/boundary.txt
"""

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hermite.main import generate_boundary_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--vzoom", type=float, default=2.5)
    parser.add_argument("--samples", type=int, default=None)
    args = parser.parse_args()
    generate_boundary_file(args.topology, args.output, vzoom=args.vzoom, samples=args.samples)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
