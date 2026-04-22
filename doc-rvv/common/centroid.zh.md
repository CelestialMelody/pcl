# `centroid.hpp`：RVV 优化实现说明

本文记录 `common/include/pcl/common/impl/centroid.hpp` 在本仓库相对上游的 `__RVV10__` 扩展实现、分流逻辑、测试与性能数据；范围限于该头文件内质心、协方差与 `demeanPointCloud` 相关模板，不包含 `computeNDCentroid` 等未改动的条目。

本仓库文件：`[common/include/pcl/common/impl/centroid.hpp](../../common/include/pcl/common/impl/centroid.hpp)`

上游对照文件：[centroid.hpp](https://github.com/PointCloudLibrary/pcl/blob/master/common/include/pcl/common/impl/centroid.hpp)

## 1. 背景与需求

上游 `centroid.hpp` 中，质心与协方差相关计算以标量循环与 Eigen 矩阵为主；`demeanPointCloud` 对 `PointCloud` / `Eigen::Matrix` 等重载为逐点或逐列赋值与减法。上游该文件不涉及 x86 SSE/AVX intrinsic，亦无可移植向量扩展开关。

本仓库在 `__RVV10__` 下的约束如下：

- 对外 API 与模板签名保持不变：公开函数名与参数列表与上游一致。
- 语义对齐：稠密路径在拆分出的 `*Standard` 中与上游稠密逻辑对齐；`is_dense == false` 仍走标量 NaN 检查分支。返回值类型、空输入与边界行为保持上游约定。
- 数据布局：点云为 AoS，`pcl::rvv_load::strided_load3_f32m2` / `indexed_load3_f32m2` 与 `pcl::rvv_store::strided_store3_f32m2` 依赖 `PointT` 为 standard-layout 且 `x/y/z` 为 `float`（`kRVVXYZPointCompatible`）；字段紧密相邻时 load/store 封装内部可选用 `vlsseg3e32` / `vssseg3e32`，否则为按字段的 strided 访问（见 `common/include/pcl/common/impl/rvv_point_load.hpp`）。
- 不适合或未完成向量化的情形：`n < 16` 时各 `*RVV` 入口回退 `*Standard`；`!cloud.is_dense` 时不进入 RVV 分流；`ConstCloudIterator` 重载仍为标量；`computeNDCentroid` 等未加 RVV 分支。索引子集上的 gather 受随机访存带宽约束，条带内仍用 `indexed_load3`，与顺序 stride 相比收益依赖数据集与硬件（参见访存策略文档中的 indexed 微基准）。

## 2. 与上游实现的差异


| 条目                                                           | 上游实现要点                        | 本仓库在 **RVV10** 下的变化                                                                                         |
| ------------------------------------------------------------ | ----------------------------- | ----------------------------------------------------------------------------------------------------------- |
| 编译条件                                                         | 无向量扩展宏                        | `#if defined(__RVV10__)` 引入 `riscv_vector.h`，并实现 `*RVV` 内核                                                  |
| `compute3DCentroid`（稠密整云 / 稠密 indices）                       | 单路径标量累加                       | `if constexpr (kRVVXYZPointCompatible)` 时分流至 `compute3DCentroidRVV`，否则 `compute3DCentroidStandard`          |
| `computeCovarianceMatrix`（给定质心 / 关于原点）稠密分支                   | 标量或拆出的 Standard 辅助            | 稠密且满足 `kRVVXYZPointCompatible` 时走 `computeCovarianceMatrixCentroidRVV` / `computeCovarianceMatrixOriginRVV` |
| `computeMeanAndCovarianceMatrix` 稠密分支                        | 单路径                           | 同上，分流至 `computeMeanAndCovarianceMatrixRVV`（整云与 indices 各一）                                                  |
| `demeanPointCloud` → `PointCloud` / indices / `Eigen`        | 逐点或逐列标量                       | 在 `is_dense` 且 `kRVVXYZPointCompatible` 时分流至 `demeanPointCloudRVV`（含原地与 indices→输出）                         |
| `n` 规模阈值                                                     | 无                             | 各 `*RVV` 内 `n < 16` 回退对应 `*Standard`                                                                        |
| `demeanPointCloud(cloud_in, centroid, cloud_out)` 双缓冲        | `cloud_out = cloud_in` 后原地减质心 | 当前仍为全量赋值后再原地 demean（见 §4.5），未在仓库中实现「读入写出单趟」融合                                                               |
| `computeNDCentroid`、稀疏 NaN 分支、`ConstCloudIterator` 质心/demean | 标量                            | 当前未 RVV 化，保持原实现                                                                                             |


数值一致性说明：

- RVV 路径在条带内使用 `f32` 向量累加与 `__riscv_vfredosum_vs_`* 将条带累加器归约到标量；与标量双精度或顺序累加相比，舍入顺序与舍入次数不同，$\sum_i x_i$、$\sum_i x_i^2$ 及后续公式在极限情况下可能与纯标量 `double` 路径有微小差异。
- `computeMeanAndCovarianceMatrixRVV` 中 $K$ 的选取仍为先序第一个 `isFinite` 点，与上游一致；其后 FMA 与归约顺序与标量路径不同，协方差元素在浮点意义上为近似对齐。
- `demeanPointCloud` 的 RVV 路径对 `x/y/z` 做 `float` 级减法，与上游对 `centroid` 转为 `float` 再减的行为一致；未引入新的「最大值相等取端点」类逻辑。
- 若需与上游逐比特对比双精度结果，当前材料不足，未下结论；应以本仓库单测与业务容差为准。

## 3. 总体设计

### 3.1 分流条件

- 编译期：`#if defined(__RVV10__)` 编译进 RVV 实现；未定义时行为与上游标量路径一致（由 `*Standard` 或原分支承担）。
- 运行期：`cloud.is_dense == true` 且 `PointT` 满足 `kRVVXYZPointCompatible` 时，公开 API 进入对应 `*RVV`；否则 `*Standard` 或非稠密标量分支。`indices` 版本额外要求索引容器非空等上游已有前置条件。
- 规模：`n < 16`（或 indices 长度小于 16）时 `*RVV` 直接调用对应 `*Standard`，避免短向量下 `vsetvl` 与归约固定开销占优。

### 3.2 组织方式

- 公开函数体保持为薄分发层：`dense` + `kRVVXYZPointCompatible` → `*RVV`，否则标量辅助函数。
- `*Standard` 由上游稠密逻辑拆分命名，供回退与非 RVV 构建共用；`*RVV` 仅在有 `__RVV10_`_ 时可见。
- 访存与类型约束集中在 `pcl::rvv_load` / `pcl::rvv_store`（`rvv_point_load.hpp` / `rvv_point_store.hpp`），`centroid.hpp` 主要编排条带循环与 Eigen 系数写回。

## 4. 详细实现

### 4.1 入口与函数分发

公开 API 在稠密且 `PointT` 满足 `kRVVXYZPointCompatible` 时进入 RVV；否则走 `*Standard` 或非稠密分支。`*RVV` 内部在 `n < 16` 时一律回退 `*Standard`。

对应实现中，`compute3DCentroid` 稠密分流骨架如下：

```cpp
// common/include/pcl/common/impl/centroid.hpp: compute3DCentroid(...)
  if (cloud.is_dense)
  {
#if defined(__RVV10__)
    if constexpr (pcl::rvv_load::kRVVXYZPointCompatible<PointT>)
      return compute3DCentroidRVV (cloud, centroid);
#endif
    return compute3DCentroidStandard (cloud, centroid);
  }
```

### 4.2 `compute3DCentroidRVV`

语义与上游稠密质心一致：对有限点集求 $\frac{1}{N}\sum_i (x_i,y_i,z_i)$ 并写入齐次坐标第四维为 $1$。

回退：`n < 16` 时调用 `compute3DCentroidStandard`。条带推进使用 `__riscv_vsetvl_e32m2(n - i)`，与 `m2` LMUL 配套；累加器用 `vfadd` 的 `_tu` 形态在条带间保留未尾段 lane（尾段语义见横切文档）。

数据流：从 `cloud.data()` 起按 `sizeof(PointT)` 步进，`strided_load3_f32m2` 取 `x/y/z`；三轴条带累加后，用 `__riscv_vfredosum_vs_f32m2_f32m1` 将 `m2` 累加器归约到标量，再乘以 $1/N$。选用有序归约而非跨条带链式标量加，是为了在固定 `vlmax` 下用单条归约指令闭合每条累加器向量；与标量从左到右累加相比顺序不同。

对应实现中「条带累加 + `vfredosum`」片段如下：

```cpp
// common/include/pcl/common/impl/centroid.hpp: compute3DCentroidRVV(...)
    const std::size_t vl = __riscv_vsetvl_e32m2 (n - i);
    vfloat32m2_t vx, vy, vz;
    pcl::rvv_load::strided_load3_f32m2<
        sizeof (PointT), offsetof (PointT, x), offsetof (PointT, y), offsetof (PointT, z)> (
        base + i * sizeof (PointT), vl, vx, vy, vz);
    v_acc_x = __riscv_vfadd_vv_f32m2_tu (v_acc_x, v_acc_x, vx, vl);
    v_acc_y = __riscv_vfadd_vv_f32m2_tu (v_acc_y, v_acc_y, vy, vl);
    v_acc_z = __riscv_vfadd_vv_f32m2_tu (v_acc_z, v_acc_z, vz, vl);
  const vfloat32m1_t v_zero = __riscv_vfmv_s_f_f32m1 (0.0f, 1);
  const Scalar sx = static_cast<Scalar> (__riscv_vfmv_f_s_f32m1_f32 (
      __riscv_vfredosum_vs_f32m2_f32m1 (v_acc_x, v_zero, vlmax)));
```

### 4.3 `computeMeanAndCovarianceMatrixRVV`

语义与上游「平移后二阶矩 + 均值」公式一致：先标量扫描确定 $K$，再在条带内对 $(x-K_x,\ldots)$ 做二次型与一次项累加，最后除以 $N$ 并填充协方差与质心。

全云版本从连续 AoS 加载；`indices` 版本对索引向量做 `vle32` 得到 `v_idx`，经 `byte_offsets_u32m2` 与 `indexed_load3_f32m2` 从 `cloud` 基址 gather `x/y/z`，再进入同一套 FMA 累加。条带内用 `__riscv_vfmacc_*_f32m2_tu` 将积累加到九个累加器上，注释中指向尾段 undisturbed 策略。

对应实现中 indices 条带「索引化偏移 + indexed load + FMA」片段如下：

```cpp
// common/include/pcl/common/impl/centroid.hpp: computeMeanAndCovarianceMatrixRVV(cloud, indices, ...)
    const vint32m2_t v_idx_i32 = __riscv_vle32_v_i32m2 (idx_i32 + i, vl);
    const vuint32m2_t v_off = pcl::rvv_load::byte_offsets_u32m2<PointT> (v_idx, vl);
    pcl::rvv_load::indexed_load3_f32m2<
        PointT, offsetof (PointT, x), offsetof (PointT, y), offsetof (PointT, z)> (
        base, v_off, vl, vx, vy, vz);
    v_acc0 = __riscv_vfmacc_vv_f32m2_tu (v_acc0, vx, vx, vl);
```

归约阶段对九个累加器各做一次 `vfredosum`，再写入 `Eigen::Matrix` 并按上游公式填对称协方差项。浮点顺序与标量双重循环不同，可能带来末位差异；无最大值比较，无端点歧义问题。

### 4.4 `computeCovarianceMatrixCentroidRVV` 与 `computeCovarianceMatrixOriginRVV`

`computeCovarianceMatrixCentroidRVV` 对应「已知质心 $\mathbf{c}$」的去均值二阶矩：条带内先 `vfsub` 得到 $(x-c_x,\ldots)$，再用六个 `vfmacc` 累加 $\sum (x')^2$、$\sum x'y'$ 等，归约后直接写入对称的 $3\times 3$ 上三角等价元素（与上游给定质心分支同一公式）。

`computeCovarianceMatrixOriginRVV` 对应「关于原点」的 $\frac{1}{N}\sum_i \mathbf{p}_i \mathbf{p}_i^\top$ 上三角六项：条带内对原始 $x,y,z$ 做六次 FMA，不归一化到向量寄存器；`vfredosum` 后乘以 `inv_n = 1/N`，再按 `coeffRef` 布局写入与上游 `computeCovarianceMatrixOriginStandard` 相同的 six-pack 映射。

二者均为六个累加器（非均值协方差的九个），`n < 16` 回退各自 `*Standard`。

对应实现中，给定质心路径「减质心 + 六项 FMA」与原点路径「六项 FMA + 标量 `inv_n` + `coeffRef` 写回」可对照如下：

```cpp
// common/include/pcl/common/impl/centroid.hpp: computeCovarianceMatrixCentroidRVV(...)
    vx = __riscv_vfsub_vf_f32m2 (vx, cx, vl);
    vy = __riscv_vfsub_vf_f32m2 (vy, cy, vl);
    vz = __riscv_vfsub_vf_f32m2 (vz, cz, vl);
    v_acc0 = __riscv_vfmacc_vv_f32m2_tu (v_acc0, vx, vx, vl);
    v_acc1 = __riscv_vfmacc_vv_f32m2_tu (v_acc1, vx, vy, vl);
    v_acc2 = __riscv_vfmacc_vv_f32m2_tu (v_acc2, vx, vz, vl);
    v_acc3 = __riscv_vfmacc_vv_f32m2_tu (v_acc3, vy, vy, vl);
    v_acc4 = __riscv_vfmacc_vv_f32m2_tu (v_acc4, vy, vz, vl);
    v_acc5 = __riscv_vfmacc_vv_f32m2_tu (v_acc5, vz, vz, vl);
  covariance_matrix (0, 0) = static_cast<Scalar> (s0);
  covariance_matrix (1, 1) = static_cast<Scalar> (s3);
  covariance_matrix (2, 2) = static_cast<Scalar> (s5);
  covariance_matrix (1, 0) = covariance_matrix (0, 1);
```

```cpp
// common/include/pcl/common/impl/centroid.hpp: computeCovarianceMatrixOriginRVV(...)
    pcl::rvv_load::strided_load3_f32m2<
        sizeof (PointT), offsetof (PointT, x), offsetof (PointT, y), offsetof (PointT, z)> (
        base + i * sizeof (PointT), vl, vx, vy, vz);
    v_acc0 = __riscv_vfmacc_vv_f32m2_tu (v_acc0, vx, vx, vl);
    v_acc1 = __riscv_vfmacc_vv_f32m2_tu (v_acc1, vx, vy, vl);
    v_acc2 = __riscv_vfmacc_vv_f32m2_tu (v_acc2, vx, vz, vl);
    v_acc3 = __riscv_vfmacc_vv_f32m2_tu (v_acc3, vy, vy, vl);
    v_acc4 = __riscv_vfmacc_vv_f32m2_tu (v_acc4, vy, vz, vl);
    v_acc5 = __riscv_vfmacc_vv_f32m2_tu (v_acc5, vz, vz, vl);
  const Scalar inv_n = static_cast<Scalar> (1) / static_cast<Scalar> (n);
  accu[0] = static_cast<Scalar> (s0) * inv_n;
  covariance_matrix.coeffRef (0) = accu[0];
  covariance_matrix.coeffRef (1) = covariance_matrix.coeffRef (3) = accu[1];
```

公开 `computeCovarianceMatrix(cloud, centroid, …)` 与 `computeCovarianceMatrix(cloud, …)` 在稠密且 `kRVVXYZPointCompatible` 时分别调用上述两函数，否则进入对应 `*Standard`。

### 4.5 `demeanPointCloudRVV`

`demeanPointCloud(cloud_in, centroid, cloud_out)` 在稠密且类型兼容时先 `cloud_out = cloud_in` 再调用原地 `demeanPointCloudRVV(cloud_out, …)`，因此性能上仍是整云拷贝叠加一次向量减质心；与 §4.2–4.4 相比算术强度更低，条带内核主要体现为访存与 `vfsub`，板卡上加速比偏小（§5.3）。`indices` 重载则按索引从 `cloud_in` gather `x/y/z`，减质心后写入 `cloud_out` 的第 $i$ 个连续槽位，与标量循环「`out[i] = in[indices[i]]` 再去均值」的坐标语义一致。

原地条带内「载入—减质心—写回」对应实现如下：

```cpp
// common/include/pcl/common/impl/centroid.hpp: demeanPointCloudRVV(cloud_out, ...)
    pcl::rvv_load::strided_load3_f32m2<
        sizeof (PointT), offsetof (PointT, x), offsetof (PointT, y), offsetof (PointT, z)> (
        base + i * sizeof (PointT), vl, vx, vy, vz);
    vx = __riscv_vfsub_vf_f32m2 (vx, cx, vl);
    vy = __riscv_vfsub_vf_f32m2 (vy, cy, vl);
    vz = __riscv_vfsub_vf_f32m2 (vz, cz, vl);
    pcl::rvv_store::strided_store3_f32m2<
        sizeof (PointT), offsetof (PointT, x), offsetof (PointT, y), offsetof (PointT, z)> (
        base + i * sizeof (PointT), vl, vx, vy, vz);
```

`indices → PointCloud` 重载在条带内用 `vle32` 取索引、`indexed_load3_f32m2` 从 `cloud_in` 取坐标，写回时使用 `out_base + i * sizeof(PointT)` 的 strided store，体现「输入随机、输出顺序」的访存模式：

```cpp
// common/include/pcl/common/impl/centroid.hpp: demeanPointCloudRVV(cloud_in, indices, ..., cloud_out)
    const vint32m2_t v_idx_i32 = __riscv_vle32_v_i32m2 (idx_i32 + i, vl);
    const vuint32m2_t v_idx = __riscv_vreinterpret_v_i32m2_u32m2 (v_idx_i32);
    const vuint32m2_t v_off_in = pcl::rvv_load::byte_offsets_u32m2<PointT> (v_idx, vl);
    pcl::rvv_load::indexed_load3_f32m2<
        PointT, offsetof (PointT, x), offsetof (PointT, y), offsetof (PointT, z)> (
        in_base, v_off_in, vl, vx, vy, vz);
    vx = __riscv_vfsub_vf_f32m2 (vx, cx, vl);
    vy = __riscv_vfsub_vf_f32m2 (vy, cy, vl);
    vz = __riscv_vfsub_vf_f32m2 (vz, cz, vl);
    pcl::rvv_store::strided_store3_f32m2<
        sizeof (PointT), offsetof (PointT, x), offsetof (PointT, y), offsetof (PointT, z)> (
        out_base + i * sizeof (PointT), vl, vx, vy, vz);
```

`Eigen::Matrix` 重载在 `Scalar` 为 `float` 且点类型兼容时调用 `demeanPointCloudEigenRVV`，按列 stride 写 `4×N` 矩阵，条带内为 gather/load 与列向 store，同属访存受限路径。

## 5. 测试与验证

### 5.1 测试入口与运行方式

- 单测源：`test-rvv/common/centroid/test_centroid.cpp`
- 基准源：`test-rvv/common/centroid/bench_centroid.cpp`
- 构建与运行：`test-rvv/common/centroid/Makefile`；板端辅助：`test-rvv/common/centroid/board.mk`
- 典型目标：`run_test_std` / `run_test_rvv`、`run_bench_std` / `run_bench_rvv`、`run_bench_compare`
- 运行环境：板卡侧日志标明 `Milkv-Jupiter`；QEMU 示例：`qemu-riscv64 -cpu rv64,v=true,vlen=256,elen=64`（见 `test-rvv/common/centroid/output/qemu/run_bench_std.log` 构建行）。Std 与 RVV 为不同编译宏下的独立可执行文件（如 `bench_centroid_std` / `bench_centroid_rvv`）。

### 5.2 日志与数据来源

本文 §5.3 性能表数据取自板卡汇总日志 `[test-rvv/common/centroid/output/board/bench_compare.log](../../test-rvv/common/centroid/output/board/bench_compare.log)`（`Cloud size: 1000000`，`Iterations: 20`，`Dataset: points=1000000 seed=42`）。

同目录下 QEMU 原始输出见 `[test-rvv/common/centroid/output/qemu/](../../test-rvv/common/centroid/output/qemu/)`（`run_bench_std.log` / `run_bench_rvv.log`）；QEMU 与板卡数值不可混用为同一性能结论。

单测通过记录见 `[test-rvv/common/centroid/output/qemu/run_test.log](../../test-rvv/common/centroid/output/qemu/run_test.log)`（`[  PASSED  ] 23 tests`）。

### 5.3 测试结果与说明


| Benchmark 项                                       | Std Avg (ms) | RVV Avg (ms) | Speedup |
| ------------------------------------------------- | ------------ | ------------ | ------- |
| compute3DCentroid (cloud)                         | 5.0541       | 2.4246       | 2.08x   |
| compute3DCentroid (indices)                       | 10.5973      | 4.0369       | 2.63x   |
| computeMeanAndCovarianceMatrix (cloud)            | 17.9383      | 2.8763       | 6.24x   |
| computeMeanAndCovarianceMatrix (indices)          | 21.6620      | 4.4339       | 4.89x   |
| computeCovarianceMatrix (cloud, given centroid)   | 15.6358      | 2.7710       | 5.64x   |
| computeCovarianceMatrix (indices, given centroid) | 16.8729      | 4.2656       | 3.96x   |
| computeCovarianceMatrix (cloud, about origin)     | 10.4723      | 2.6889       | 3.89x   |
| computeCovarianceMatrix (indices, about origin)   | 17.1524      | 4.0728       | 4.21x   |
| demeanPointCloud (cloud -> PointCloud)            | 17.1303      | 11.2783      | 1.52x   |
| demeanPointCloud (indices -> PointCloud)          | 15.9871      | 12.4232      | 1.29x   |
| demeanPointCloud (cloud -> Eigen 4xN)             | 17.5953      | 15.1959      | 1.16x   |


Speedup 按同一日志中 Std Avg 与 RVV Avg 之比计算，与 `bench_compare.log` 中汇总行一致。`demeanPointCloud` 类条目加速比明显低于二阶矩类条目，与先全量拷贝再原地写回、算术强度低、受 DDR 带宽约束相符。

## 6. 总结

本仓库在保持 `centroid.hpp` 对外 API 与模板签名不变的前提下，通过 `__RVV10__` 与 `if constexpr (kRVVXYZPointCompatible)` 将稠密路径分流到 `*RVV` 实现；`n < 16` 与非稠密路径回退标量，语义与上游约定对齐，浮点末位可能因向量归约与 FMA 顺序与标量路径存在差异。

主要技术动作包括：AoS 上 `strided_load3` / `indexed_load3` 与 `strided_store3`，条带内 `vfadd`/`vfsub`/`vfmacc` 与 `vfredosum` 归约，`vsetvl_e32m2` 推进；`demeanPointCloud` 仍为拷贝后原地更新。尾段策略依赖 `_tu` 累加器形态，与横切文档一致。

已知限制：`ConstCloudIterator` 与 `computeNDCentroid` 未向量化；`demeanPointCloud(cloud_in, cloud_out)` 含全量赋值。

## 7. 其他相关文档

访存形态与微基准：[doc-rvv/rvv/RVV Load Store Strategy.zh.md](../rvv/RVV%20Load%20Store%20Strategy.zh.md)

尾段 `_tu` 语义：[doc-rvv/rvv/Tail-Agnostic-Tail-Undisturbed.zh.md](../rvv/Tail-Agnostic-Tail-Undisturbed.zh.md)

模块侧评估索引：[doc-rvv/common/module-evaluation.zh.md](module-evaluation.zh.md)`