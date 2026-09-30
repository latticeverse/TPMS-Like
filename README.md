# TPMS-like shell lattices

[中文说明](README.zh-CN.md)

This repository accompanies **New families of triply periodic minimal surface-like shell lattices** (*Additive Manufacturing* 77, 103779, 2023). It implements the complete workflow from a topology graph to a three-dimensional periodic shell lattice:

1. generate periodic boundary curves and cubic Hermite splines from a topology OBJ;
2. fit a minimal surface with the Deep Currents neural implicit-surface model;
3. assemble periodic copies, clip the result, and thicken it into a shell mesh.

## Repository layout

```text
tpmslike/             Boundary-file and OBJ I/O utilities
hermite/              Hermite boundary generator
deepcurrents/         Deep Currents model and training program
postprocessing/       Periodic assembly, mesh cleanup, and shell thickening
scripts/              Stage-by-stage and end-to-end commands
examples/topologies/  Legacy topology graphs
environment.yml       Conda environment specification
```

Inverse homogenization and mechanical homogenization experiments are not included in this open-source release. No undocumented `gen_shape.py` command is provided.

## Installation

Python 3.10 and Conda are recommended:

```bash
conda env create -f environment.yml
conda activate tpmslike
```

The environment includes PyTorch, NumPy, SciPy, PyVista, VTK, scikit-image, TensorBoard, tqdm, and PyMCubes. CUDA is optional; use `--device cuda:0` when a suitable GPU is available, or run on the CPU otherwise.

## End-to-end run

Run a short test optimization for topology 1:

```bash
python scripts/run_pipeline.py \
  --topology examples/topologies_paper/1.obj \
  --boundary examples/boundaries/paper_topo_1.txt \
  --run-dir runs/topo_1_rrr \
  --type rrr \
  --iterations 2000 \
  --n-samples 2048 \
  --grid-size 0.08 \
  --thickness 0.08
```

For paper-quality results, use `--iterations 100000`, `--n-samples 4096`, and `--grid-size 0.04`. The run produces:

- `boundary.txt`: the Hermite boundary;
- `surface/0_rrr_or_ms_average.obj`: the fitted implicit surface;
- `shell.ply`: the final periodic thick shell.

## Run individual stages

### 1. Generate a boundary

```bash
python scripts/generate_boundary.py \
  --topology examples/topologies_paper/1.obj \
  --output runs/topo_1/boundary.txt \
  --vzoom 2.0
```

### 2. Train Deep Currents

```bash
python -m deepcurrents.train \
  --boundary runs/topo_1/boundary.txt \
  --type rrr \
  --out runs/topo_1/surface \
  --n_iterations 100000 \
  --n_samples 4096 \
  --grid-size 0.04 \
  --seed 1 \
  --device cuda:0
```

### 3. Build a shell

```bash
python scripts/build_shell.py \
  --surface runs/topo_1/surface/0_rrr_or_ms_average.obj \
  --output runs/topo_1/shell.ply \
  --symmetry rrr \
  --thickness 0.08
```

## Data formats

Topology files use OBJ vertices and line segments to describe the graph. Boundary files are read and written through `tpmslike.boundary`: the first line gives the number of closed curves, followed by three-dimensional coordinates in curve order, with one scalar per line. Blank lines and `#` comments are accepted.

```text
1
-1.0
0.0
1.0
...
```

The modules communicate through explicit files:

```text
topology .obj -> boundary .txt -> implicit surface .obj -> shell .ply
```

## Periodic assembly and mesh postprocessing

For `rrr`, the implementation reflects the surface across the planes specified by the paper, assembles the copies, clips the central periodic cell, scales it to `[-1, 1]^3`, and then offsets it to create a shell. `ttt` uses translated copies. Use `--symmetry none` when the input is already a complete periodic unit.

## License and citation

This project is released under the MIT License. Please read [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The `deepcurrents/` package is derived from [DeepCurrents](https://github.com/dmsm/DeepCurrents), whose upstream license notice must be retained.

```bibtex
@article{xu2023tpmslike,
  title   = {New families of triply periodic minimal surface-like shell lattices},
  author  = {Xu, Yonglai and Pan, Hao and Wang, Ruonan and Du, Qiang and Lu, Lin},
  journal = {Additive Manufacturing},
  volume  = {77},
  pages   = {103779},
  year    = {2023},
  doi     = {10.1016/j.addma.2023.103779}
}
```
