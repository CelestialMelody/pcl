# `distances.h`：RVV 优化实现说明

本文记录 `common/include/pcl/common/distances.h` 在本仓库相对上游的 `__RVV10__` 扩展实现、分流逻辑、测试与性能数据。
本仓库文件：[`common/include/pcl/common/distances.h`](../../common/include/pcl/common/distances.h)
上游对照文件：[distances.h`](https://github.com/PointCloudLibrary/pcl/blob/master/common/include/pcl/common/distances.h)

## 1. 背景与需求

上游 `distances.h` 中，`getMaxSegment` 两个重载均为标量双层循环，复杂度 $O(N^2)$；`squaredEuclideanDistance`、`euclideanDistance`、`sqrPointToLineDistance` 维持标量/Eigen 路径。

本仓库的 `__RVV10__` 扩展约束如下：

- API 与模板签名保持不变：对外仍是 `getMaxSegment(...)` 两个重载。
- 语义对齐上游：仍计算最远点对长度，返回值与空输入行为保持一致（无有效点对时返回 `std::numeric_limits<double>::min()`）。
- 数据布局约束：点云是 AoS（`PointT`），向量访存通过 `strided`/`segmented` load 从结构体中提取 `x/y/z`。
- 不适合向量化场景：`n < 512` 直接回退标量；`indices` 版本先打包再算，避免在 $O(N^2)$ 循环内做随机 gather。
- 横切设计参考：`rvv` 访存策略由 `pcl/common/rvv_point_load.h` 与 `impl/rvv_point_load.hpp` 统一封装（编译期在 `vlsseg3e32` 与 `3x vlse32` 间选择）。

## 2. 与上游实现的差异

| 条目 | 上游实现要点 | 本仓库在 **RVV10** 下的变化 |
| --- | --- | --- |
| 头文件依赖 | 无 RVV 相关 include | `#if defined(__RVV10__)` 下引入 `riscv_vector.h`、`rvv_point_load.h` |
| `getMaxSegment(cloud, ...)` 入口 | 直接标量实现 | 入口改为 `RVV/Standard` 分发：`__RVV10__` 走 `getMaxSegmentRVV` |
| `getMaxSegmentRVV(cloud, ...)` | 上游无此函数 | 新增 RVV 内核：`vsetvl` 条带循环 + 向量减法/FMA + `vfredmax` 归约 + `vfirst` 定位 lane |
| RVV 回退条件 | 无 | `n < 512` 回退 `getMaxSegmentStandard` |
| `getMaxSegment(indices, ...)` 入口 | 直接标量实现 | 同样改为 `RVV/Standard` 分发 |
| `getMaxSegmentRVV(indices, ...)` | 上游无此函数 | `n==0` 早退；`n<512` 回退；其余先将 `indices` 对应点打包为连续 `packed_cloud`，再复用 cloud-RVV 内核 |
| 其他距离函数 | 标量/Eigen | 未 RVV 化，保持原实现 |

数值一致性说明：
当前实现保持同一数学目标，但 RVV 路径使用 `f32` 向量算子与向量归约，计算顺序与标量双循环不同；并且最值定位通过 `vmfeq + vfirst`。在存在近似相等候选值时，端点索引可能与标量路径不同，但单测未观察到行为错误。当前材料不足，未下更细粒度误差上界结论。

## 3. 总体设计

### 3.1 分流条件

- 编译期：`#if defined(__RVV10__)` 决定是否暴露 RVV 路径。
- 运行期：
  - `cloud` 版本：`n < 512` 回退 `Standard`。
  - `indices` 版本：`n == 0` 直接返回最小值；`n < 512` 回退；其余走打包 + RVV。
- 特殊路径原则：优先保证语义对齐与可维护性，避免在小规模输入上引入 RVV 启动成本。

### 3.2 组织方式

- 公共 API 保持不变，仅在内部替换为 `RVV/Standard` 分发壳。
- `RVV` 负责大规模数据条带计算；`Standard` 保留参考语义与回退保障。
- 访存细节不直接散落在 `distances.h`，而是调用 `pcl::rvv_load::strided_load3_f32m2` 统一处理 AoS 字段加载。

## 4. 详细实现

### 4.1 入口与函数分发

符号与回退条件如下：

| 符号 | 回退条件 |
| --- | --- |
| `getMaxSegmentRVV(cloud, ...)` | `cloud.size() < 512` |
| `getMaxSegmentRVV(cloud, indices, ...)` | `indices.size() < 512`（且 `n==0` 直接返回） |

```cpp
// common/include/pcl/common/distances.h（省略了无关代码）
template <typename PointT> double inline
getMaxSegment (const pcl::PointCloud<PointT> &cloud, PointT &pmin, PointT &pmax)
{
#if defined(__RVV10__)
  return getMaxSegmentRVV (cloud, pmin, pmax);
#else
  return getMaxSegmentStandard (cloud, pmin, pmax);
#endif
}
```

设计取舍：公开接口不改动，避免调用点变更；所有架构差异都留在头文件内部可内联路径中。

### 4.2 `getMaxSegmentRVV(const PointCloud<PointT>&, ...)`

该函数仍按上游语义求最远点对：遍历 $(i,j)$，最大化 $\|p_i-p_j\|_2^2$ 并返回 $\sqrt{\max}$；整体复杂度维持 $O(N^2)$。RVV 化只替换内层对 $(p_i - p_j)$ 的距离计算与“条带内最大值”求解方式。

实现分两层回退/分流：

- `n < 512`：直接回退 `getMaxSegmentStandard`，避免在小规模输入上被 `vsetvl`、向量归约与 lane 定位开销抵消。
- `n >= 512`：进入条带循环。

条带循环骨架是“外层标量 `i` + 内层 `j` 分块推进”：

- 内层 `j` 使用 `vl = __riscv_vsetvl_e32m2(n - j)` 自适应尾段，`j += vl` 推进，避免额外标量 tail-loop。
- `p_j` 的 `x/y/z` 通过 `pcl::rvv_load::strided_load3_f32m2<sizeof(PointT), offsetof(PointT,x), ...>` 从 AoS 结构体中做 stride/segment load。该封装在字段紧邻时优先走 `vlsseg3e32`，否则退化为 `3x vlse32`（见 `pcl/common/rvv_point_load.h`/`impl/rvv_point_load.hpp`）。

对应实现中“条带推进 + AoS 载入 + 归约 + lane 定位”的主干片段如下：

```cpp
// common/include/pcl/common/distances.h: getMaxSegmentRVV(cloud, ...)
const std::size_t vl = __riscv_vsetvl_e32m2 (n - j);

vfloat32m2_t vx, vy, vz;
pcl::rvv_load::strided_load3_f32m2<sizeof (PointT),
                                  offsetof (PointT, x),
                                  offsetof (PointT, y),
                                  offsetof (PointT, z)>(
    reinterpret_cast<const std::uint8_t*>(&cloud[j]), vl, vx, vy, vz);

vx = __riscv_vfsub_vf_f32m2 (vx, xi, vl);
vy = __riscv_vfsub_vf_f32m2 (vy, yi, vl);
vz = __riscv_vfsub_vf_f32m2 (vz, zi, vl);

vfloat32m2_t vdist2 = __riscv_vfmul_vv_f32m2 (vx, vx, vl);
vdist2 = __riscv_vfmacc_vv_f32m2 (vdist2, vy, vy, vl);
vdist2 = __riscv_vfmacc_vv_f32m2 (vdist2, vz, vz, vl);

const vfloat32m1_t vinit = __riscv_vfmv_s_f_f32m1 (max_dist, 1);
const vfloat32m1_t vmax1 = __riscv_vfredmax_vs_f32m2_f32m1 (vdist2, vinit, vl);
const float vmax = __riscv_vfmv_f_s_f32m1_f32 (vmax1);

if (vmax > max_dist) {
  const vbool16_t m = __riscv_vmfeq_vf_f32m2_b16 (vdist2, vmax, vl);
  const long lane = __riscv_vfirst_m_b16 (m, vl);
  if (lane >= 0) {
    max_dist = vmax;
    i_min = i;
    i_max = j + static_cast<std::size_t>(lane);
  }
}
```

条带内计算与归约/索引恢复策略如下：

- 计算：`vfsub` 做 `vx-=xi`、`vy-=yi`、`vz-=zi`，再用 `vfmul + vfmacc` 累加得到 `vdist2 = dx^2 + dy^2 + dz^2`。
- 归约：用 `vfredmax` 将 `vdist2` 归约为条带最大值 `vmax`，并与当前全局 `max_dist` 比较。
- 索引恢复：当 `vmax` 更新全局最大值时，使用 `vmfeq(vdist2, vmax)` 得到掩码，再用 `vfirst` 找到首次命中 lane，将 `i_max = j + lane`。该选择在“条带内有多个等值最大项”时偏向最先出现的 lane；与标量路径的端点选择可能不一致，但不改变返回长度语义。

写回阶段保持标量：只维护 `i_min/i_max`，结束后一次性写回 `pmin = cloud[i_min]`、`pmax = cloud[i_max]`。

### 4.3 `getMaxSegmentRVV(const PointCloud<PointT>&, const Indices&, ...)`

该函数语义与上游一致：仅在 `indices` 指定的点子集上求最远点对长度。

回退与边界处理：

- `n == 0`：直接返回 `std::numeric_limits<double>::min()`。
- `n < 512`：回退 `getMaxSegmentStandard(cloud, indices, ...)`。

当 `n >= 512` 时，本仓库不在 $O(N^2)$ 内核中做 indexed gather（例如 `vluxei32`/`vluxseg3ei32`）来按 `indices` 读取点坐标，而是先做一次 $O(N)$ 打包：

- 将 `cloud[indices[k]]` 顺序拷贝到 `packed_cloud[k]`（`pcl::PointCloud<PointT>`，保持与 PCL 内部 allocator 兼容）。
- 随后调用 `getMaxSegmentRVV(packed_cloud, ...)` 复用 cloud-RVV 内核。

对应实现中“打包子集 + 复用 cloud-RVV 内核”的片段如下：

```cpp
// common/include/pcl/common/distances.h: getMaxSegmentRVV(cloud, indices, ...)
pcl::PointCloud<PointT> packed_cloud;
packed_cloud.resize (n);
packed_cloud.width = static_cast<std::uint32_t>(n);
packed_cloud.height = 1;
packed_cloud.is_dense = true;
for (std::size_t k = 0; k < n; ++k)
  packed_cloud[k] = cloud[indices[k]];

return getMaxSegmentRVV (packed_cloud, pmin, pmax);
```

该取舍将“随机访存 + $O(N^2)$ 重复加载”的成本转化为“一次性线性拷贝 + 连续 AoS stride/segment load”。代价是额外内存与一次拷贝；收益依赖硬件缓存/内存带宽与向量访存实现，材料仅覆盖板卡与 QEMU 的当前测量结果。

## 5. 测试与验证

### 5.1 测试入口与运行方式

- 单测源：`test-rvv/common/distances/test_distances.cpp`
- 基准源：`test-rvv/common/distances/bench_distances.cpp`
- 构建/运行入口：`test-rvv/common/distances/Makefile`、板端 `test-rvv/common/distances/board.mk`
- 常用目标：
  - `make run_test_std` / `make run_test_rvv`
  - `make run_bench_std` / `make run_bench_rvv`
  - `make run_bench_compare`
- 运行环境：
  - QEMU：`qemu-riscv64 -cpu rv64,v=true,vlen=256,elen=64`
  - 板端：`Milkv-Jupiter`（见 `board.mk` 与板端日志）

### 5.2 测试结果与说明

以下表格使用板卡侧 `test-rvv/common/distances/output/board/bench_compare.log` 中同参数数据（`maxseg_points=2500, iterations=20`）：

| Benchmark 项 | Std Avg (ms) | RVV Avg (ms) | Speedup |
| --- | ---: | ---: | ---: |
| getMaxSegment (cloud, O(n^2)) | 62.7429 | 12.4788 | 5.03x |
| getMaxSegment (indices, O(n^2)) | 87.8233 | 12.4328 | 7.06x |

补充说明：

- 单测在 Std/RVV 路径均通过（5/5）。
- 板卡对比日志中 `Iterations` 字段解析为 “未解析到 Iter/Iterations”，但原始 bench 输出包含 `iterations: 20`，表中 `Avg (ms)` 直接取自同一份日志里的 bench 输出与对比汇总。
- 板端提供了独立微基准 `bench_getmaxsegment_load_compare`：`vlsseg3e32` 相对 `3x vlse32` 约 `1.139x~1.150x`；该结果用于访存策略比较，不等同于 `distances.h` 全路径端到端加速比。

## 6. 总结

本仓库在不改变 `distances.h` 对外接口与模板签名的前提下，引入了 `__RVV10__` 条件分流，并为 `getMaxSegment` 两个重载增加 RVV 实现与回退路径。

主要工作是 AoS 条带访存、向量化平方累加、`vfredmax` 归约和 `vmfeq+vfirst` 索引恢复。`indices` 路径通过先打包再计算，避免在 $O(N^2)$ 循环内做随机 gather。当前日志显示 QEMU 上端到端性能尚未优于标量路径；同时，归约顺序与浮点运算路径差异可能导致边界场景端点选择不同。
