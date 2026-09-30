# TPMS-like 壳晶格：中文版说明

本项目是论文 **New families of triply periodic minimal surface-like shell
lattices**（*Additive Manufacturing* 77, 103779, 2023）的配套代码。项目实现从拓扑图到三维周期壳晶格的完整流程：

1. 从拓扑 OBJ 生成周期边界曲线和 Hermite 样条；
2. 使用 Deep Currents 神经隐式曲面模型拟合最小曲面；
3. 进行周期拼接、裁剪和加厚，输出可视化及后续制造使用的壳体网格。

英文说明见 [README.md](README.md)。

## 项目结构

```text
tpmslike/                    边界文件和 OBJ 读写工具
hermite/                     Hermite 边界生成器
deepcurrents/                Deep Currents 模型和训练程序
postprocessing/              周期拼接、网格清理和壳体加厚
scripts/                     分阶段及一键运行脚本
examples/topologies/         20 个 legacy 拓扑图
environment.yml              Conda 环境
```

逆均匀化（IH）和力学均匀化实验不包含在当前开源目录中，因此本项目不提供未验证的 `gen_shape.py` 等命令。

## 安装环境

推荐使用 Python 3.10 和 Conda：

```bash
conda env create -f environment.yml
conda activate tpmslike
```

环境包含 PyTorch、NumPy、SciPy、PyVista、VTK、scikit-image、TensorBoard、tqdm 和 PyMCubes。CUDA 是可选的；有可用 GPU 时建议显式指定 `--device cuda:0`，否则可以使用 CPU。


## 一键运行

用较低预算测试拓扑 1：

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

论文质量的运行建议使用 `--iterations 100000`、`--n-samples 4096` 和 `--grid-size 0.04`。输出包括：

- `boundary.txt`：Hermite 边界；
- `surface/0_rrr_or_ms_average.obj`：训练得到的平均隐式曲面；
- `shell.ply`：周期加厚后的最终壳体。


## 分阶段运行

### 1. 生成边界

```bash
python scripts/generate_boundary.py \
  --topology examples/topologies_paper/1.obj \
  --output runs/topo_1/boundary.txt \
  --vzoom 2.0
```

### 2. 训练 Deep Currents

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

### 3. 生成壳体

```bash
python scripts/build_shell.py \
  --surface runs/topo_1/surface/0_rrr_or_ms_average.obj \
  --output runs/topo_1/shell.ply \
  --symmetry rrr \
  --thickness 0.08
```

## 数据格式

拓扑文件使用 OBJ 的顶点和线段表示图结构。边界文件由 `tpmslike.boundary` 统一读写：第一行是闭合曲线数量，之后按曲线顺序写入三维坐标，每个标量占一行；允许空行和 `#` 注释。

示例：

```text
1
-1.0
0.0
1.0
...
```

模块之间通过明确的文件传输数据：拓扑 `.obj` → 边界 `.txt` → 隐式曲面 `.obj` → 壳体 `.ply`。

## 周期拼接和网格后处理

对于 `rrr`，程序按照论文约定在指定盒面反射曲面，拼接后裁剪到中心周期单元，再缩放到 `[-1, 1]^3`，最后进行厚度偏移。`ttt` 使用平移复制；如果输入本身已经是完整周期单元，则使用 `--symmetry none`。


## 许可和引用

本项目采用 MIT License。请同时阅读 [LICENSE](LICENSE) 和
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)；`deepcurrents/` 源自
[DeepCurrents](https://github.com/dmsm/DeepCurrents)，需要保留其上游许可说明。
论文引用：

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
