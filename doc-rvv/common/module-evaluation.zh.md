# common/common 模块：RVV 优化目标评估

本文档记录对 `common/include/pcl/common`（含 `impl/` 与顶层 `.h`）的阅读结论：哪些路径适合后续 bench、向量化诊断与 RVV 实现。已完成的 `impl/common.hpp` 见 `[common.zh.md](./common.zh.md)`。

**评估标准**：

- 是否存在随点云规模 `N`、核长度或特征维度 `dim` 增长的循环，且以浮点乘加、规约为主。
- 数据访问是否相对规整；分支与 `gather` 成本是否可控。
- 热点是否落在 PCL 自写循环内，还是已委托给 Eigen / 标准库（后者优先改链接库或算法，而非在 PCL 内重复写向量核）。

---

## 1. 优先处理


| 目标             | 路径                           | 说明                                                                                                                                                                                                                 |
| -------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 点云变换           | `impl/transforms.hpp`        | `transformPointCloud`、`transformPointCloudWithNormals`、`PointXY` 仿射等：外层对 `cloud.size()` 线性遍历；`detail::Transformer` 在 x86 上已有 SSE2/AVX 特化，RISC-V 可补 RVV 版矩阵-向量与批量点变换。`is_dense == false` 时需按 NaN 跳过，实现比 dense 分支复杂。 |
| 质心 / 协方差 / 去均值 | `impl/centroid.hpp`          | `compute3DCentroid` 多重重载、`demeanPointCloud` 等：对点或 `indices` 的 `for`/`while`。与 `common.hpp` 已做条目同属点云规约，函数面更大（indices、非有限点分支）。                                                                                       |
| 可分离高斯卷积        | `impl/gaussian.hpp`          | `convolveRows` / `convolveCols`：行/列二重循环，内层为核权乘加。核宽较小时需注意条带启动与尾部标量收尾成本。                                                                                                                                             |
| 最远点对           | `distances.h`（无对应 `impl/`）   | `getMaxSegment`：对全云或 `indices` 的 O(n²) 双重循环，内层为平方距离与取最大。可向量化「固定 `i`、扫描 `j`」的内层，或先改算法再考虑 SIMD。同文件内 `sqrPointToLineDistance`、`squaredEuclideanDistance` 等为单次或小批量运算，优先级低于 `getMaxSegment`。                            |
| 向量范数           | `impl/norms.hpp`             | `L1_Norm`、`L2_Norm_SQR`、`Linf_Norm` 等在 `dim` 上循环。`dim` 较大时 RVV 更有意义；`dim` 常为 3～数十时需 bench 验证。`CS_Norm`、`Div_Norm`、`KL_Norm` 等分支多、含 `log` 与除法，可先只做 L1 / L2² / Linf。                                                 |
| 投影矩阵估计         | `impl/projection_matrix.hpp` | `estimateProjectionMatrix`：对点集累加对称块统计量，循环体以乘加为主；含 `isfinite` 与像素索引推导，测试需与标量路径对齐。                                                                                                                                   |


---

## 2. 暂不列入优化的文件

下列条目在静态阅读下不满足「宽循环 + PCL 自写热点」或收益预期偏低；若后续 profiling 推翻再移出本表。


| 文件                                             | 说明                                         |
| ---------------------------------------------- | ------------------------------------------ |
| `impl/accumulators.hpp`                        | 按字段累加，Eigen 与 `map` 等混合，非长向量规约循环。          |
| `impl/angles.hpp`                              | 单标量角度换算，无数组级循环。                            |
| `impl/bivariate_polynomial.hpp`                | 低次、短循环，控制流多。                               |
| `impl/copy_point.hpp`                          | `memcpy` 或编译期字段拷贝。                         |
| `impl/eigen.hpp`                               | 见第 5 节。                                    |
| `impl/file_io.hpp`                             | 目录与路径字符串。                                  |
| `impl/generate.hpp`                            | 随机填充，时间多在 RNG。                             |
| `impl/intensity.hpp`                           | 单点强度访问器；无批量 API 时文件级 RVV 意义有限。             |
| `impl/intersections.hpp`                       | 少量几何代数，无大规模数据并行。                           |
| `impl/io.hpp`                                  | 元数据、`copyPointCloud` 逐点逻辑。                 |
| `impl/pca.hpp`                                 | 大块在 Eigen 矩阵与特征解算。                         |
| `impl/piecewise_linear_function.hpp`           | 单次插值查询。                                    |
| `impl/polynomial_calculations.hpp`             | 见第 6 节。                                    |
| `impl/random.hpp`                              | RNG。                                       |
| `impl/spring.hpp`                              | 容器 `insert` / 扩容，内存与搬运为主。                  |
| `impl/transformation_from_correspondences.hpp` | 增量 3×3 与末尾 `JacobiSVD`。                    |
| `impl/vector_average.hpp`                      | 增量协方差 + `SelfAdjointEigenSolver`，时间多在特征解算。 |


顶层 `.h`（声明为主、实现少量或一次性）：


| 文件                                           | 说明                                                  |
| -------------------------------------------- | --------------------------------------------------- |
| `geometry.h`                                 | 两点距离、投影等，单次调用尺度。                                    |
| `concatenate.h`                              | `NdConcatenateFunctor` 按字段指针写，非规整 SIMD 循环。          |
| `point_tests.h`                              | `isFinite` 等标量谓词。                                   |
| `utils.h`                                    | `equal`、`ignore`。                                   |
| `colors.h`                                   | LUT 在静态初始化中填充 256 / 4000 项，运行期以查表为主；优化初始化对典型路径帮助有限。 |
| `synchronizer.h`                             | 时间同步与队列逻辑。                                          |
| `feature_histogram.h`、`poses_from_matches.h` | 接口在头文件；实现多在 `.cpp`，需跟源文件单独评估。                       |


---

## 3. 其他文件评估说明

### 3.1`common/include/pcl/common/fft`

- 代码位置：公开声明在 `fft/*.h`；蝶形运算在 `common/src/fft/kiss_fft.c`、`kiss_fftr.c`，经 `_kiss_fft_guts.h` 宏展开。
- 算法形态：递归 `kf_work`、radix 2/3/4/5 蝶形；访存为跨步与多指针，非「连续 `N` 个 float 一条 `vle`」模式。`USE_SIMD` 分支将 `kiss_fft_scalar` 设为 `__m128`，属于整库类型约定切换，无现成 RVV 等价开关。
- 结论：算术量大，但 RVV 适配成本高；是否改动应以目标 workload 上 `kiss_fft` / `kiss_fftr` 的采样为准。若占比低，优先处理第 1 节点云循环。

---

### 3.2 `impl/eigen.hpp`

- 内容类型：3×3（及 2×2）对称特征值闭式、`computeRoots` 中的 `sqrt` / `atan2` / 三角函数、固定长度 ≤3 的循环；`invert`*、`determinant` 为短公式链。
- 并行维度：单次调用内元素个数少，RVV 通道利用率低；瓶颈多为标量依赖链与超越函数，而非可条带化的长循环。
- `umeyama`：Eigen 3.3+ 委托 `Eigen::umeyama`，大矩阵运算在 Eigen 库内；PCL 头文件侧无独立长循环可供 RVV 替换。
- 结论：不作为 RVV 第一批改造对象。若大矩阵路径慢，先查 Eigen / BLAS 在目标架构上的向量化与链接方式。

---

### 3.3 `impl/polynomial_calculations.hpp`

- 求根系列（`solveLinear`～`solveQuartic`）：单次多项式、分支与 `pow` / `cbrt` / `acos` 等，无「一次处理 K 个独立实例」的数组接口，与 RVV 常见用法不匹配。
- `bivariatePolynomialApproximation`：对每个样本构造基向量 `C` 并累加 `A`、`b`；维数随阶数变化，上三角累加；末尾 `A.inverse() * b`。静态阅读下，维数升高时开销常落在稠密求逆，而非内层乘加循环 alone。
- 结论：求根路径不做 RVV 优先。拟合路径若需优化，应先评估用 Cholesky / `ldlt` 等替代显式 `inverse()`，再对热点循环做 bench；必要时再考虑 SIMD / RVV。

