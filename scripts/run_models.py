"""Run several paper topologies sequentially and record their output paths."""

from pathlib import Path
import argparse
import json
import subprocess
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topologies", nargs="+", type=int, default=[1, 3, 6])
    parser.add_argument("--topology-dir", type=Path, default=None,
                        help="Directory containing <id>.obj files (defaults to examples/topologies)")
    parser.add_argument("--out", type=Path, default=Path("runs/models"))
    parser.add_argument("--iterations", type=int, default=100000)
    parser.add_argument("--n-samples", type=int, default=4096)
    parser.add_argument("--grid-size", type=float, default=0.04)
    parser.add_argument("--thickness", type=float, default=0.08)
    parser.add_argument("--vzoom", type=float, default=2.5,
                        help="Hermite tangent scale for custom topology directories")
    parser.add_argument("--type", choices=("rrr", "ttt", "ttt_a"), default="rrr")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    topology_dir = args.topology_dir or (root / "examples" / "topologies_paper")
    results = []
    for topology_id in args.topologies:
        run_dir = args.out / f"topo_{topology_id}_{args.type}"
        command = [
            sys.executable, str(root / "scripts" / "run_pipeline.py"),
            "--topology", str(topology_dir / f"{topology_id}.obj"),
            "--run-dir", str(run_dir), "--type", args.type,
            "--iterations", str(args.iterations), "--grid-size", str(args.grid_size),
            "--n-samples", str(args.n_samples),
            "--thickness", str(args.thickness), "--device", args.device,
        ]
        if topology_dir.name == "topologies_paper":
            boundary = root / "examples" / "boundaries" / f"paper_topo_{topology_id}.txt"
            if boundary.exists():
                command.extend(["--boundary", str(boundary)])
        else:
            command.extend(["--vzoom", str(args.vzoom)])
        started = time.time()
        record = {"topology": topology_id, "command": command, "status": "running"}
        try:
            subprocess.run(command, check=True, cwd=root)
            record.update({"status": "ok", "shell": str(run_dir / "shell.ply")})
        except subprocess.CalledProcessError as exc:
            record.update({"status": "failed", "returncode": exc.returncode})
        record["seconds"] = round(time.time() - started, 2)
        results.append(record)
        (args.out / "summary.json").parent.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
