# MACE-LAMMPS 编译与运行教程: Graphene-water interface

本目录用于运行论文案例 D: graphene-water interface 的 fine-tuned MACE 模型。
当前采用 MACE 官方的原生 LAMMPS 接口，即 `pair_style mace` / `PKG_ML-MACE`。

目录中的关键文件:

- `graphene-water.data`: 372 原子 graphene-water LAMMPS data 文件。
- `mace-ft-tutorial-main-3_compiled_f32.model-lammps.pt`: 已转换好的 LAMMPS MACE 模型。
- `mace-lmp.in`: 可直接用于本目录模型的 LAMMPS 输入文件。
- `run.sh`: Slurm/模块环境示例脚本，按你的集群模块名修改。

> 原子类型顺序很重要。本目录 data 文件中 type 1/2/3 分别是 `C/O/H`，
> 因此 `pair_coeff * * ... C O H` 必须保持这个顺序。

## 0. 激活 MACE Python 环境

在仓库根目录激活已构建好的环境:

```bash
cd /Users/shiqian/Documents/github/MLIP/finetuning/mace-ft-tutorial
source scripts/activate_mace_ft.sh
```

检查模型转换命令是否可用:

```bash
mace_create_lammps_model --help
```

## 1. 将 fine-tuned MACE 模型转换为 LAMMPS 模型

如果你已经有 fine-tuned 的 `.model` 文件，例如:

```text
examples/graphene-water/results/graphene-water-mace-mpa0/graphene-water-mace-mpa0_stagetwo.model
```

用下面命令生成 LAMMPS 可读的 TorchScript 模型:

```bash
cd /Users/shiqian/Documents/github/MLIP/finetuning/mace-ft-tutorial

mace_create_lammps_model \
  examples/graphene-water/results/graphene-water-mace-mpa0/graphene-water-mace-mpa0_stagetwo.model \
  --dtype float32
```

通常会生成类似文件:

```text
examples/graphene-water/results/graphene-water-mace-mpa0/graphene-water-mace-mpa0_stagetwo.model-lammps.pt
```

复制到本目录:

```bash
cp examples/graphene-water/results/graphene-water-mace-mpa0/*-lammps.pt lmp-mace-f32/
```

然后在 `mace-lmp.in` 中改成你的模型文件名:

```lammps
pair_style mace no_domain_decomposition
pair_coeff * * your_finetuned_model-lammps.pt C O H
```

本目录已经自带一个可用的 f32 LAMMPS 模型:

```lammps
pair_coeff * * mace-ft-tutorial-main-3_compiled_f32.model-lammps.pt C O H
```

## 2. CPU 版 LAMMPS + ML-MACE 编译

适合先做功能验证。CPU 可以跑，但 MACE-LAMMPS 在 GPU 上通常快很多。

```bash
mkdir -p $HOME/software
cd $HOME/software

# 下载 CPU libtorch。版本建议与你用于转换模型的 PyTorch 大版本尽量一致。
curl -L -o libtorch-cpu.zip \
  https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.2.0%2Bcpu.zip
unzip libtorch-cpu.zip
mv libtorch libtorch-cpu

# 使用 ACEsuit/lammps 的 mace 分支，包含 PKG_ML-MACE。
git clone --branch mace --depth=1 https://github.com/ACEsuit/lammps.git lammps-mace
cd lammps-mace
mkdir build-cpu
cd build-cpu

cmake \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX=$(pwd) \
  -D CMAKE_CXX_STANDARD=17 \
  -D CMAKE_CXX_STANDARD_REQUIRED=ON \
  -D BUILD_MPI=ON \
  -D BUILD_OMP=ON \
  -D PKG_OPENMP=ON \
  -D PKG_ML-MACE=ON \
  -D CMAKE_PREFIX_PATH=$HOME/software/libtorch-cpu \
  ../cmake

make -j 8
make install
```

编译完成后可执行文件通常在:

```text
$HOME/software/lammps-mace/build-cpu/lmp
```

运行:

```bash
cd /Users/shiqian/Documents/github/MLIP/finetuning/mace-ft-tutorial/lmp-mace-f32
export OMP_NUM_THREADS=8
mpirun -np 1 $HOME/software/lammps-mace/build-cpu/lmp -in mace-lmp.in
```

## 3. GPU 版 LAMMPS + ML-MACE 编译

GPU 推荐使用 Kokkos。下面以 CUDA 12.1 + A100 为例，其他 GPU 需要替换
`Kokkos_ARCH_*`:

- A100: `Kokkos_ARCH_AMPERE80=ON` 或按集群/源码支持情况使用 `AMPERE100`
- V100: `Kokkos_ARCH_VOLTA70=ON`
- H100: `Kokkos_ARCH_HOPPER90=ON`

```bash
mkdir -p $HOME/software
cd $HOME/software

curl -L -o libtorch-gpu.zip \
  https://download.pytorch.org/libtorch/cu121/libtorch-shared-with-deps-2.2.0%2Bcu121.zip
unzip libtorch-gpu.zip
mv libtorch libtorch-gpu

git clone --branch mace --depth=1 https://github.com/ACEsuit/lammps.git lammps-mace
cd lammps-mace
mkdir build-gpu
cd build-gpu

cmake \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX=$(pwd) \
  -D CMAKE_CXX_STANDARD=17 \
  -D CMAKE_CXX_STANDARD_REQUIRED=ON \
  -D BUILD_MPI=ON \
  -D BUILD_SHARED_LIBS=ON \
  -D PKG_KOKKOS=ON \
  -D Kokkos_ENABLE_CUDA=ON \
  -D CMAKE_CXX_COMPILER=$(pwd)/../lib/kokkos/bin/nvcc_wrapper \
  -D Kokkos_ARCH_AMPERE80=ON \
  -D CMAKE_PREFIX_PATH=$HOME/software/libtorch-gpu \
  -D PKG_ML-MACE=ON \
  ../cmake

make -j 20
make install
```

运行单 GPU:

```bash
cd /Users/shiqian/Documents/github/MLIP/finetuning/mace-ft-tutorial/lmp-mace-f32
$HOME/software/lammps-mace/build-gpu/lmp -k on g 1 -sf kk -in mace-lmp.in
```

Slurm 示例:

```bash
#!/bin/bash
#SBATCH -J gw-mace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

module purge
module load cuda/12.1
module load gcc/9
module load openmpi

cd /path/to/mace-ft-tutorial/lmp-mace-f32
/path/to/lammps-mace/build-gpu/lmp -k on g 1 -sf kk -in mace-lmp.in
```

## 4. 运行本目录 graphene-water 示例

确认 `mace-lmp.in` 中模型路径为本目录相对路径:

```lammps
read_data   graphene-water.data
pair_style  mace no_domain_decomposition
pair_coeff  * * mace-ft-tutorial-main-3_compiled_f32.model-lammps.pt C O H
```

然后运行:

```bash
cd /Users/shiqian/Documents/github/MLIP/finetuning/mace-ft-tutorial/lmp-mace-f32
lmp -k on g 1 -sf kk -in mace-lmp.in
```

如果是 CPU 编译版:

```bash
export OMP_NUM_THREADS=8
mpirun -np 1 /path/to/lmp -in mace-lmp.in
```

## 5. 常见问题

### `Unknown pair style mace`

说明 LAMMPS 没有打开 `PKG_ML-MACE`，或者你运行的不是刚编译的 `lmp`。
重新检查:

```bash
/path/to/lmp -h | grep -i mace
```

### 找不到 `libtorch.so` / `libtorch_cpu.dylib`

运行前设置动态库路径:

```bash
# Linux
export LD_LIBRARY_PATH=$HOME/software/libtorch-gpu/lib:$LD_LIBRARY_PATH

# macOS
export DYLD_LIBRARY_PATH=$HOME/software/libtorch-cpu/lib:$DYLD_LIBRARY_PATH
```

### `pair_coeff` 后元素顺序不对

`graphene-water.data` 的 Masses 部分是:

```text
1 C
2 O
3 H
```

因此必须写:

```lammps
pair_coeff * * model-lammps.pt C O H
```

### 多 GPU 怎么办

原生 `pair_style mace no_domain_decomposition` 当前更推荐单 GPU。若要多 GPU，
建议评估 MACE 的 ML-IAP 接口:

```bash
mace_create_lammps_model your.model --format=mliap
```

对应 LAMMPS 输入会变成:

```lammps
pair_style mliap unified your.model-mliap_lammps.pt 0
pair_coeff * * C O H
```

ML-IAP 需要 LAMMPS `PKG_ML-IAP`、`PKG_PYTHON`、`MLIAP_ENABLE_PYTHON=ON`
和 Kokkos GPU 构建；这条路线更适合后续做多 GPU 性能测试。

## 参考

- MACE 官方 LAMMPS 文档: https://mace-docs.readthedocs.io/en/latest/guide/lammps.html
- MACE ML-IAP 文档: https://mace-docs.readthedocs.io/en/latest/guide/lammps_mliap.html
