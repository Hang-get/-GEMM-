# CUDA SGEMM Optimization

从朴素 CUDA 矩阵乘法出发，逐步引入共享内存分块、线程级 Tile、寄存器复用、`float4` 向量化访存、双缓冲和共享内存 Bank Conflict 优化，以 cuBLAS SGEMM 为性能基线。

这个仓库保留各阶段的独立实现，方便对照代码、运行测试和理解每一步优化的收益与代价。

**原笔记中的结果：** 在 `M = N = K = 4096` 的测试中，避免共享内存 Bank Conflict 的版本达到 cuBLAS 吞吐的 **99.5%**，约为 Naive 版本的 **9.4 倍**。这是原笔记报告的历史结果，不代表本仓库在任意 GPU 上都能达到相同性能，也不是本次 README 整理时重新测量的数据。

> 本 README 根据 [CUDA SGEMM 优化笔记](http://linn-ylz.com/Computer-Science/CUDA/CUDA-SGEMM-optimization-notes/) 整理，并对照仓库源码补充运行方式、Kernel 编号和边界限制。实现沿用原项目，来源与许可证见文末。

## 1. 计算目标

实现单精度通用矩阵乘法：

$$
C \leftarrow \alpha AB + \beta C
$$

其中：

| 参数 | 含义 |
| --- | --- |
| `A` | `M × K`，FP32，行主序 |
| `B` | `K × N`，FP32，行主序 |
| `C` | `M × N`，FP32，行主序 |
| `alpha`、`beta` | 缩放系数；当前测试程序固定为 `1.0`、`0.0` |

行主序元素地址为 `row * stride + col`。Naive 版本中，每个线程负责一个输出元素，沿 K 维累加乘积。

cuBLAS 默认按列主序解释矩阵。本项目利用 `Cᵀ = BᵀAᵀ`，交换 A、B 以及 M、N 的传入顺序，完成行主序数据的计算：

```cpp
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, &alpha, B, N, A, K, &beta, C, N);
```

## 2. 项目结构与 Kernel 编号

```text
.
├── CMakeLists.txt
├── main.cu                  # Kernel 选择、cuBLAS 校验、CUDA Event 计时
├── run.sh                   # 对 Kernel 10 扫描方阵尺寸
├── LICENSE
└── src
    ├── utils.cu             # 数据生成、结果校验、设备信息与错误检查
    ├── utils.cuh
    └── sgemm
        ├── sgemm.cuh        # Kernel 接口与辅助宏
        ├── sgemm.cu         # cuBLAS 封装与 Naive 实现
        ├── shared_memory.cu # 共享内存分块
        ├── tile.cu          # 一维/二维 Tile 与寄存器缓存
        ├── float4.cu        # 向量化访存
        ├── double_buffering.cu
        └── split_tile.cu    # 交错 Tile 与 Bank Conflict 优化变体
```

Kernel 编号由 `main.cu` 中的 `switch` 定义：

| 编号 | 实现 | 源码文件 |
| --- | --- | --- |
| `0` | cuBLAS 基线 | `src/sgemm/sgemm.cu` |
| `1` | Naive | `src/sgemm/sgemm.cu` |
| `2` | Shared Memory | `src/sgemm/shared_memory.cu` |
| `3` | 1D Tile | `src/sgemm/tile.cu` |
| `4` | 2D Tile | `src/sgemm/tile.cu` |
| `5` | 2D Tile + Register Cache | `src/sgemm/tile.cu` |
| `6` | 2D Tile + `float4` | `src/sgemm/float4.cu` |
| `7` | `float4` + Double Buffering | `src/sgemm/double_buffering.cu` |
| `8` | Bank Conflict 优化版本 | `src/sgemm/split_tile.cu` |
| `9` | 1D Split Tile 变体 | `src/sgemm/split_tile.cu` |
| `10` | 2D Split Tile 变体 | `src/sgemm/split_tile.cu` |

请使用 `0–10` 中的合法编号；当前程序对其他编号也会落入 cuBLAS 分支，不会提示编号无效。

## 3. 编译与运行

### 环境要求

- 支持 CUDA 的 NVIDIA GPU 与匹配的驱动。
- CUDA Toolkit，包括 `nvcc` 和 cuBLAS。
- CMake **3.22 或以上**。
- 与所用 CUDA Toolkit 兼容的主机 C++ 编译器；项目设置 CUDA C++20。

**macOS 注意：** 普通 MacBook 不能运行本项目的 NVIDIA CUDA Kernel。可以在 Mac 上编辑代码和管理仓库，但编译、正确性验证和性能测试需要在具备 NVIDIA GPU 的 Linux/Windows 环境进行。

### 编译

```bash
git clone https://github.com/Hang-get/-GEMM-.git
cd ./-GEMM-
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

当前 `CMakeLists.txt` 设置了 `-O3 -lineinfo`，目标架构固定为 `CUDA_ARCHITECTURES "86"`。请根据 GPU 的 Compute Capability 修改这一设置后再编译，不要把 `86` 当成所有 GPU 的通用配置。可参考 [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda-gpus)。

如果 CMake 无法找到 `nvcc`，可显式指定实际安装路径，例如：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

可执行文件输出到**项目根目录**，名称为 `sgemm`，不是 `build/sgemm`。

### 运行

```bash
./sgemm
./sgemm 0 4096
./sgemm 1 4096
./sgemm 8 4096
./sgemm 8 2048 4096 1024
```

依次表示：打印设备信息、运行 cuBLAS、运行 Naive、运行 Bank Conflict 优化版本，以及用 `M=2048, N=4096, K=1024` 运行优化版本。

参数形式为：

```text
./sgemm <kernel_id> <size>
./sgemm <kernel_id> <M> <N> <K>
```

正常运行时先与 cuBLAS 结果对比，通过后输出三个字段：

```text
<平均耗时（秒）> <吞吐（GFLOPS）> <M>
```

最后一列只输出 M；矩形矩阵测试需要额外记录 N、K。吞吐按 `2MNK / 时间 / 10^9` 计算，未额外计入 alpha、beta 的运算。

运行已有的尺寸扫描脚本：

```bash
bash run.sh
```

该脚本**只测试 Kernel 10**，将 `M=N=K` 从 256 增加到 6400，步长为 256。它不是所有 Kernel 的完整性能对照脚本。

## 4. 优化路线

主要阅读路径为：

```text
Naive → Shared Memory → 1D Tile → 2D Tile
      → Register Cache → float4 → Double Buffering → Bank Conflict 优化
```

其中 `BM × BN` 是一个 Block 负责的输出区域，`BK` 是每次沿 K 维处理的长度，`TM × TN` 是一个线程负责的输出区域。

### 4.1 Naive：每个线程计算一个输出元素

每个线程沿 K 维读取 A 的一行和 B 的一列，累加得到 `C[row, col]`。实现简单，也包含输出边界判断，但不同线程会反复请求相同的输入数据。

不考虑缓存与合并访存时，每个输出元素需要读取约 `2K` 个 FP32 元素，执行约 `2K` 次浮点运算，对应约 **0.25 FLOP/Byte** 的输入算术强度。这里计算的是算法层面的读取需求，不是实际测得的显存流量。

### 4.2 Shared Memory：在 Block 内复用输入

每轮把 `BM × BK` 的 A 子块和 `BK × BN` 的 B 子块从 Global Memory 搬到 Shared Memory，再由 Block 内线程共享。

一个 Block 完成整个 K 维计算时，忽略边界填充和 C 的读写：

- Naive 输入读取量：`2K × BM × BN` 个元素。
- Shared Memory 分块输入读取量：`K × (BM + BN)` 个元素。
- 比值：`(1/BM + 1/BN) / 2`；当 `BM=BN` 时为 `1/BM`。

同一输出 Tile 的总运算量没有改变，减少的是重复的全局输入读取。不过共享内存访问、同步与指令发射仍有成本，不能仅凭读取量下降就断言性能必然提升。原笔记中，这一版本甚至略慢于 Naive。

### 4.3 1D Tile：每个线程计算多个输出

一个线程负责同一列中 `TM` 个输出元素，以寄存器中的 `val[TM]` 保存累加结果。固定 k 时，读取到的 B 元素可以服务于多个 A 元素的乘加，提高线程内数据复用。

当前配置为 `BM=128, BN=16, BK=16, TM=8`，一个 Block 包含 `16 × 16` 个线程。

### 4.4 2D Tile：在线程内计算一个小矩阵

一个线程负责 `TM × TN` 个输出，使用二维累加数组。每个 k 上，TM 个 A 元素和 TN 个 B 元素可组成一次小型外积，完成 `TM × TN` 次乘加。

当前配置为 `BM=BN=64, BK=16, TM=TN=4`。增大线程 Tile 有利于复用，但同时提高寄存器压力；若发生 Register Spill，临时数据可能落入 Local Memory，抵消收益。

### 4.5 Register Cache：显式复用共享内存中的操作数

先把当前 k 对应的 A、B 元素读入线程私有数组，再执行乘加：

```cpp
for (int m = 0; m < TM; ++m) {
    a_reg[m] = As[thread_row + m][k];
}
for (int n = 0; n < TN; ++n) {
    b_reg[n] = Bs[k][thread_col + n];
}
for (int m = 0; m < TM; ++m) {
    for (int n = 0; n < TN; ++n) {
        accum[m][n] += a_reg[m] * b_reg[n];
    }
}
```

这是概念示例，完整实现见 `tile.cu`。原笔记中的收益不明显：显式缓存不一定改变最终生成的访存指令，也不能自动隐藏 Shared Memory 到寄存器的加载延迟。

### 4.6 float4：减少访存指令与索引计算

一次向量加载/存储处理 4 个连续 FP32 元素，并将 A 的子块转置存入 Shared Memory，以适配后续的访问布局。当前配置为 `BM=BN=128, BK=8, TM=TN=8`。

主要收益是减少访存指令与地址计算，改善指令吞吐。**加载同样的数据时，float4 不会自动把传输字节数变成四分之一**；实际收益取决于对齐、布局、编译器和硬件。

向量访存要求合适的 16 字节对齐、行跨度以及完整 Tile 边界。当前向量化实现没有完整的尾部处理，限制见第 6 节。

### 4.7 Double Buffering：交替使用两份缓冲区

为 Shared Memory 和操作数缓存分配两份空间，通过 `write_idx ^= 1` 交替使用，使下一轮写入不覆盖上一轮仍在读取的数据，从而去掉单缓冲版本中的循环尾部同步。

每轮写入后的 Block 级同步仍然保留。在该循环结构中，下一轮的同步也参与约束缓冲区再次复用前的进度，因此不能任意删除。

**实现边界：** 当前代码是双缓冲与同步减少方案，没有使用 `cp.async` 或显式异步拷贝流水线；不能仅凭 Double Buffering 的名称就宣称已经实现完整的异步预取与计算重叠。两份缓冲区也会增加 Shared Memory 占用。

### 4.8 Bank Conflict 优化：交错分配线程 Tile

即使全局输入复用充分，共享内存访问仍可能因多个线程命中同一 Bank 的不同地址而被串行化。`split_tile.cu` 中的实现将线程负责的区域按 4 个元素一组交错分配，并同步调整读取与结果写回的坐标。

关键是协调**数据布局、线程到数据的映射以及向量访存宽度**。仅仅转置 A 或仅仅使用 `float4`，并不保证所有共享内存访问都没有冲突。

原笔记以这一阶段作为主要优化终点。寄存器 Bank Conflict 与 SASS 级调优仅作为后续方向提及，仓库没有相应的完整优化实现。

## 5. 性能对比

以下数据来自原笔记的性能总结，测试尺寸为 `M=N=K=4096`，以 cuBLAS 吞吐归一化：

| 实现 | cuBLAS 吞吐占比 |
| --- | ---: |
| cuBLAS | 100.0% |
| Naive | 10.6% |
| Shared Memory | 9.1% |
| 1D Tile | 15.8% |
| 2D Tile | 35.2% |
| `float4` | 85.7% |
| Double Buffering | 91.2% |
| Shared Memory Bank Conflict 优化 | **99.5%** |

原性能总结没有单列 Register Cache、1D Split Tile 和 2D Split Tile 变体的数据，因此这里不补造结果，也不把 99.5% 自动归给全部变体。

从这些结果可以看到：单独引入 Shared Memory 并没有获得加速；线程内复用和向量化带来主要提升，双缓冲与访问布局优化进一步缩小了与 cuBLAS 的差距。

**结果解读边界：** 原笔记的文字内容未完整列出 GPU 型号、CUDA/cuBLAS 版本、时钟条件和重复统计口径。这些比例只表示原测试条件下的结果，不能作为跨硬件保证，也不能据此推断 Tensor Core 路径的性能。当前代码使用 FP32 乘加，没有显式 Tensor Core 实现。

## 6. 正确性与适用范围

### 输入尺寸与尾部处理

| Kernel | 当前源码中的边界处理 |
| --- | --- |
| `0` | cuBLAS 支持通用合法尺寸 |
| `1–5` | 实现中包含边界判断；分块加载越界位置补零 |
| `6–10` | 未实现完整的边界判断与尾部补零 |

对于当前固定参数下的 `6–10`，请使用 **M、N 均为 128 的倍数，K 为 8 的倍数**的正整数尺寸。仅满足“元素总数是 4 的倍数”不足以保证安全。

程序接受矩形矩阵参数，不代表每个 Kernel 都对任意矩形尺寸做了正确处理。`1–5` 的边界支持是源码层面的判断，仍应通过 GPU 测试验证；本次 README 整理未执行 CUDA 回归测试。

### 当前校验方式

程序使用固定随机种子 `1101`，在 `[-5, 5]` 范围生成输入，先运行所选 Kernel 与 cuBLAS，再逐元素比较结果。当前阈值为**绝对误差 `0.1`**，不是相对误差。

这是基础校验，不是完整的数值测试集。它没有显式检查 NaN/Inf，也没有覆盖不同 alpha、beta 组合。若要进一步完善，应补充绝对/相对误差联合判定、非有限值检查、多个随机种子以及尺寸边界测试。

### 基准程序限制

- `repeat_times` 当前为 **1**；虽然计时前已执行过待测 Kernel，但单次 Event 计时仍容易受噪声影响。
- 计时仅围绕 Kernel 调用，不包含内存分配、主机到设备/设备到主机的数据传输和结果校验，不是端到端耗时。
- 参数使用 `atoi` 解析，没有完整的合法性检查。请自行确保尺寸是正整数且显存足够。
- 程序提示中的 `6400` 没有对应的硬性上限检查；它也是原扫描脚本的最大尺寸，不是所有设备的通用容量上限。
- Block/Tile 参数写死在各测试封装中，未实现自动调优。

这些实现适合学习和对照优化过程；作为生产级通用 SGEMM 使用前，还需要补齐边界、数值验证、参数检查和稳定基准。

## 7. 性能分析

建议使用 **Nsight Systems** 查看时间线、CPU/GPU 交互和整体执行过程，使用 **Nsight Compute** 分析单个 Kernel 的访存、指令与资源占用。

项目已经启用 `-lineinfo`，可在工具中将指标关联到源码。安装 Nsight Compute 且具有采样权限后，可运行：

```bash
ncu --set full --kernel-name regex:no_share_conflict_kernel -o profile_kernel8 ./sgemm 8 4096
```

`--kernel-name` 过滤项目 Kernel，避免把正确性校验中的 cuBLAS Kernel 混入主要分析对象。Profiling 会改变执行开销，**不要把采样时输出的 GFLOPS 当作正常运行的基准结果**。

重点关注：

| 指标/现象 | 分析方向 |
| --- | --- |
| Global Memory 吞吐与 Long Scoreboard | 全局访存依赖、数据复用、延迟隐藏 |
| MIO Throttle | 共享内存等 MIO 操作的发射压力，需结合源码定位 |
| Shared Memory Bank Conflict | 同一访问请求的 Bank 映射与访问布局 |
| 寄存器数、Local Memory 访问 | Tile 是否过大、是否发生 Spill |
| Occupancy、活跃 Warp 与 Stall | 寄存器/共享内存资源限制及可调度工作量 |
| Barrier Stall | 同步等待及不同线程的进度差异 |

具体指标名称与解释会随 GPU 架构和工具版本变化；不能只凭单个 Stall 指标就断定瓶颈。

复测性能时应记录 GPU、驱动、CUDA/cuBLAS 版本、编译架构、矩阵尺寸与精度配置，并增加预热、重复次数和多轮统计。与 cuBLAS 比较时，明确两边的 Math Mode 与精度约束，避免比较不同数值语义的计算路径。

## 8. 参考资料与许可证

本项目实现沿用 YangLinzhuo 的原始 CUDA SGEMM 项目，本 README 将原笔记改写为面向仓库使用的说明。保留来源，便于追溯实现与性能结果。

- [CUDA SGEMM 优化笔记 — 原始文章](http://linn-ylz.com/Computer-Science/CUDA/CUDA-SGEMM-optimization-notes/)
- [YangLinzhuo/cuda-sgemm-optimization — 原始代码仓库](https://github.com/YangLinzhuo/cuda-sgemm-optimization)
- [CUDA SGEMM 矩阵乘法优化笔记：从入门到 cuBLAS](https://zhuanlan.zhihu.com/p/518857175)
- [NVIDIA SGEMM Practice](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE)
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance — Simon Boehm](https://siboehm.com/articles/22/CUDA-MMM)
- [CUDA Bank Conflict 笔记](http://zh0ngtian.tech/posts/96744e8c.html)
- [深入浅出 GPU 优化系列：GEMM 优化（三）](https://zhuanlan.zhihu.com/p/481600052)
- [cuBLAS 行主序矩阵调用说明](https://www.cnblogs.com/cuancuancuanhao/p/7763256.html)

项目采用 [MIT License](LICENSE)，保留原始版权声明 `Copyright (c) 2023 yanglinzhuo`。
