# centroid.hpp：函数级梳理、筛选评估与 RVV 优先级

本文档是 `common/include/pcl/common/impl/centroid.hpp` 在 test-rvv/common/centroid 下的唯一规范说明：汇总函数（族）梳理、评估维度与函数级优先级总表，用于工作流程中的「函数级筛选与评估」及 baseline / bench / RVV 迭代。

---

## 1. 函数（族）梳理

下列均为 `namespace pcl` 内模板；按职责分组，组内再区分 cloud 全量 / `Indices` / `PointIndices` 转发。

### 1.1 `compute3DCentroid`

函数族说明：估计点云的 三维算术质心（各坐标分量的样本均值），写入 `Eigen::Matrix<Scalar, 4, 1>`，通常 `centroid[3]=1`（齐次）。模板参数为 `PointT`、`Scalar`；返回值为参与统计的点数（或 0 表示无效）。


| 重载                            | 作用                                              |
| ----------------------------- | ----------------------------------------------- |
| `ConstCloudIterator<PointT>`  | 按迭代器遍历，可配合 `isFinite` 跳过无效点。                    |
| `PointCloud` 全量               | `empty` 早退；`is_dense` 可走稠密快速路径；否则逐点 `isFinite`。 |
| `PointCloud` + `Indices`      | 对索引子集求质心；`is_dense` 与稀疏分支同上。                    |
| `PointCloud` + `PointIndices` | 转调 `indices.indices`。                           |

算法特征：主路径 O(n)，每点向质心向量贡献 x / y / z 累加。

### 1.2 `computeCovarianceMatrix`

函数族说明：与 3×3 对称协方差 / 二阶矩 相关的多套 API，模板参数为 `PointT`、`Scalar`；输出均为 `Eigen::Matrix<Scalar, 3, 3>`。语义上分为三类：给定质心（相对质心的二阶矩和，未除以 n）、给定质心且归一化、未给质心（关于坐标原点的二阶矩，各元素已除以 n）。下列仅列主干重载，`PointIndices` 均转发对应 `Indices` 版本。

**A. 给定质心（中心化二阶矩，和式，不除以 n）**


| 重载                                     | 作用                                                 |
| -------------------------------------- | -------------------------------------------------- |
| `(cloud, centroid, cov)`               | 全点相对 `centroid` 累加离差外积；`is_dense` 与 `isFinite` 两路。 |
| `(cloud, indices, centroid, cov)`      | 仅索引子集。                                             |
| `(cloud, PointIndices, centroid, cov)` | 转调 `indices.indices`。                              |


**B. `computeCovarianceMatrixNormalized`（在 A 的结果上除以点数）**


| 重载                         | 作用                                                                                                  |
| -------------------------- | --------------------------------------------------------------------------------------------------- |
| `(cloud, centroid, cov)` 等 | 若 `point_count != 0`，则 `covariance_matrix /= point_count`；重载形态与 A 一致（含 `Indices` / `PointIndices`）。 |


**C. 未给质心（关于原点的二阶矩，矩阵元素为样本二阶矩 / n）**


| 重载                           | 作用                                       |
| ---------------------------- | ---------------------------------------- |
| `(cloud, cov)`               | 6 元累加 xx, xy, xz, yy, yz, zz 后除以 n 填对称阵。 |
| `(cloud, indices, cov)`      | 索引子集。                                    |
| `(cloud, PointIndices, cov)` | 转调 `indices.indices`。                    |


算法特征：稠密主路径多为 O(n) 与 二阶项累加（或给定质心时等价规模）；A 与 C 的统计含义不同，测试与 bench 需区分。

### 1.3 `computeMeanAndCovarianceMatrix`

函数族说明：联合估计 3D 质心（`Eigen::Matrix<Scalar, 4, 1>`）与 3×3 协方差（与 §1.2 仅协方差类 API 不同）。模板参数 `PointT`、`Scalar`；实现上采用 锚点移位 与累加向量 `accu`（见源码），以减轻有限精度下的数值问题。


| 重载                                     | 作用                    |
| -------------------------------------- | --------------------- |
| `(cloud, cov, centroid)`               | 全云；`is_dense` 与稀疏分支。  |
| `(cloud, indices, cov, centroid)`      | 索引子集。                 |
| `(cloud, PointIndices, cov, centroid)` | 转调 `indices.indices`。 |


算法特征：主路径 O(n)，每点贡献二阶项与一阶项（用于恢复质心与协方差）；稠密路径可做 RVV 与 §1.1 / §1.2 类似的访存假设。

### 1.4 `computeCentroidAndOBB`

函数族说明：在 质心 + 协方差 基础上构造 有向包围盒（OBB）：先调用 `computeMeanAndCovarianceMatrix`，再对 3×3 协方差做特征分解（`SelfAdjointEigenSolver`），用主轴对齐点云后求各轴 min/max。

- 有 全 `PointCloud` 与 `Indices` 两套重载。

### 1.5 `demeanPointCloud`

函数族说明：对每个点做 去均值，即用 `点坐标 − centroid` 的 (x, y, z) 作为结果（`Eigen::Matrix<Scalar, 4, 1>` 的齐次第 4 维不参与减法）。模板参数 `PointT`、`Scalar`；输出可为 `PointCloud`（原地或按索引拷贝）或 `Eigen::Matrix`（典型 `4×n`，仅前三行写坐标）。源码按 输出容器 与 遍历方式 分下列重载：

**输出为 `pcl::PointCloud<PointT>`**


| 重载（要点）                                                 | 作用                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `(ConstCloudIterator<PointT>& iter, centroid, cloud_out, npts)` | 迭代器遍历；`npts==0` 时先扫一遍计数并重置迭代器；写 `cloud_out[i].{x,y,z}`。 |
| `(cloud_in, centroid, cloud_out)`                            | `cloud_out = cloud_in` 后对 `cloud_out` 逐点 原地 减质心（`point.{x,y,z} -= centroid`）。 |
| `(cloud_in, Indices, centroid, cloud_out)`                   | 按索引子集 拷贝 到 `cloud_out`（尺寸 `indices.size()`），每点 `cloud_in[indices[i]] - centroid`。若 `indices.size()==cloud_in.size()` 则保留 `width/height`，否则 `height=1`。 |
| `(cloud_in, PointIndices, centroid, cloud_out)`              | 转调 `indices.indices`。                                     |


输出为 `Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>`（典型为 `4×n`，仅前 3 行写 `x,y,z`）


| 重载（要点）                                                          | 作用                                                        |
| --------------------------------------------------------------- | --------------------------------------------------------- |
| `(ConstCloudIterator<PointT>& iter, centroid, cloud_out, npts)` | 与上类似先定 `npts`；列为一点，`cloud_out(0:2, i) = 点坐标 - centroid`。  |
| `(cloud_in, centroid, cloud_out)`                               | `npts = cloud_in.size()`，逐点填 `4×n` 矩阵前三行。                 |
| `(cloud_in, Indices, centroid, cloud_out)`                      | `npts = indices.size()`，第 `i` 列来自 `cloud_in[indices[i]]`。 |
| `(cloud_in, PointIndices, centroid, cloud_out)`                 | 转调 `indices.indices`。                                     |


算法特征：主路径为 O(n)，每点固定 3 次减法（及写回）；`Indices` 版对输入为 gather，对 `PointCloud` 输出为连续写；对 Eigen 输出为 按列 写，访存模式与 `PointCloud` 连续点序不同。RVV 优化需按重载分别考虑 in-place / gather / SoA 矩阵列。

### 1.6 `computeNDCentroid`

- 基于 `fieldList` 与 `for_each_type` 的 N 维通用质心，维度由点类型字段决定。

### 1.7 `CentroidPoint` / `computeCentroid(PointOutT)`

- `CentroidPoint::add` / `get`：融合累加与类型过滤，面向泛型输出点类型。
- `computeCentroid(cloud[, indices], PointOutT&)`：通过 `CentroidPoint` 聚合。

---

## 2. 函数级筛选：评估维度

对每个函数族建议从下列维度做 低 / 中 / 高 定性，再映射到 §3 优先级。


| 维度     | 含义                                  | 与向量化/RVV 的关系                 |
| ------ | ----------------------------------- | ---------------------------- |
| 循环规模   | 主循环是否 O(n)、`n` 是否常大                 | 规模大时易摊薄向量启动成本                |
| 算术密度   | 每点乘加、乘累加、归约次数                       | 算术越密，相对访存越值得做 SIMD/RVV       |
| 访存规整性  | 连续 AoS、是否需 gather/scatter           | 规整 strided 访问优先；索引 gather 次之 |
| 分支复杂度  | `is_dense` / `isFinite`、多分支 min-max | 分支多则实现与测试成本高                 |
| 数值语义风险 | 累加顺序、非结合律、双路径一致性                    | 需容差测试与明确验收标准                 |
| 可测试性   | 是否易用单测/bench 固定输入与规模                | 不可测则不宜列为高优先级大改               |


映射提示：循环规模与算术密度均高、访存规整、可测性好 → 倾向 高；强依赖 profile/vec 报告 → 中；非热点循环或 MPL/融合主导 → 低。

---

## 3. 函数级优先级与 RVV 候选表

下表为 可执行候选清单：优化方向与风险针对 RISC-V RVV（`__RVV10__`） 路径；状态反映当前仓库中该函数族是否已有合入的 RVV 优化（以本目录测试与 `centroid.hpp` 为准，需随迭代更新）。


| 优先级 | 状态   | 函数族                                                       | 优化方向                                                     | 主要风险             | 预期收益         | 回退条件                                |
| ------ | ------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------- | ---------------- | --------------------------------------- |
| 高     | 已完成 | `compute3DCentroid`（cloud / indices）                       | dense 路径 RVV：x/y/z 归约与索引路径                         | 点类型布局与精度     | 线性吞吐明显提升 | 不满足项目门控或 bench 无收益时保持标量 |
| 高     | 已完成 | `computeMeanAndCovarianceMatrix`（cloud / indices）          | dense：9 路累加（xx…zz 与 x/y/z 一次和）                     | 累加顺序导致浮点细差 | 中–高            | `n < 16` 或类型/布局不适用时标量        |
| 中     | 已完成 | `computeCovarianceMatrix`（给定 centroid，cloud / indices）  | dense：`kRVVXYZPointCompatible` 时 6 路 FMA 累加；`n<16` 标量 | 累加顺序浮点细差     | 中等             | `n < 16` 或非稠密/类型不适用时标量      |
| 中     | 已完成 | `computeCovarianceMatrix`（关于原点 / `cov` 无 centroid，cloud / indices） | dense：同上 6 路 FMA，再 `/ n` 写入 `coeffRef`               | 同上                 | 中等             | 同上                                    |
| 中     | 已完成 | `demeanPointCloud`（cloud / indices）                        | 批量减中心并写回                                             | 写回带宽             | 中等偏低         | 带宽受限则标量                          |
| 低     | 暂缓   | `computeCentroidAndOBB`                                      | 内部 min/max 等循环                                          | 特征分解占比较高     | 低–中            | profile 非热点则不投                    |
| 低     | 暂缓   | `computeNDCentroid` / `CentroidPoint`                        | MPL/融合路径                                                 | 可维护性差           | 低               | 不进行 RVV 化                           |


---

## 4. 与后续流程的衔接

1. Baseline：对「高 / 中」候选在标量构建下记录 bench（见 README 中 `run_bench_std`）。
2. 静态诊断：对目标翻译单元或剥离内核使用 `vec-missed` / `Rpass-missed`，记录未向量化的原因与循环位置。
3. RVV 实现：在功能等价前提下按项目统一策略（类型与布局门控、小规模标量回退等）添加或调整 RVV 路径。
4. 验证：QEMU 单测 → 板卡 bench → 相对 baseline 计算 speedup；未达标则分析实现或保留标量。
