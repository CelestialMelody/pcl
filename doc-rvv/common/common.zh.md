# common：RVV 优化实现说明

本文说明本仓库 [`common/include/pcl/common/impl/common.hpp`](../../common/include/pcl/common/impl/common.hpp) 相对上游 [PointCloudLibrary `common.hpp`](https://github.com/PointCloudLibrary/pcl/blob/master/common/include/pcl/common/impl/common.hpp) 的引入 RISC-V Vector (RVV 1.0) 扩展的底层适配细节。

---

## 1. 背景与需求

上游文件以标量循环与 Eigen 表达式为主：`getMeanStd` 对 `std::vector<float>` 逐元素累加到 `double`；`getPointsInBox`、`getMaxDistance`、`getMinMax3D` 在 dense 与非 dense 分支上分别处理；`calculatePolygonArea` 用顶点叉积累加再取范数。x86 上另有 `__SSE__` / `__AVX__` 的 `acos` 近似与 `getAcuteAngle3D` 批量版本，RISC-V 目标没有对应条目。

本仓库在 `__RVV10__` 下需要：模板签名与调用约定不变；行为与标量路径可对应；在点数与数据布局允许时，用步进 load、indexed load 与浮点归约缩短热点路径。点云多为 AoS，相邻点的 `x` 之间隔 `sizeof(PointT)`，向量侧以 `vlse32`、`vluxei32` 为主。

strip-mine 与 tail 语义见 [`doc-rvv/rvv/Tail-Agnostic-Tail-Undisturbed.zh.md`](../rvv/Tail-Agnostic-Tail-Undisturbed.zh.md)；选用 `vfloat32m2_t` 的说明见 [`doc-rvv/rvv/Why vfloat32m2_t.zh.md`](../rvv/Why%20vfloat32m2_t.zh.md)。

---

## 2. 与上游实现的差异

对照对象为上游 [common.hpp](https://github.com/PointCloudLibrary/pcl/blob/master/common/include/pcl/common/impl/common.hpp)（无 `__RVV10__`、无 RVV 内建）。


| 条目                                        | 上游实现要点                                                                            | 本仓库在 `__RVV10__` 下的变化                                                                                                                     |
| ----------------------------------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `getMeanStd`                              | 循环内 `sum += value`（`float`→`double`）、`sq_sum += value * value`，全程在标量 `double` 中累加 | 抽出 `getMeanStdKernelRVV`：`vfloat32m2_t` 上 `vfadd_tu` / `vfmacc_tu` 条带累加，末尾 `vfredosum` 再写入 `double`，舍入顺序与上游不完全相同（见下文注释）                   |
| `getPointsInBox`                          | dense：逐点六比较写 `indices`；非 dense：`isfinite` + 盒判断                                   | `getPointsInBoxRVV` 内联 dense 向量条带（`vlse` + 比较 + `vid` + `vcompress`）；`getPointsInBoxStandard` 内联 dense 标量循环与非 dense 分支；不再拆 `*Dense*` 辅助函数 |
| `getMaxDistance`                          | `.norm()` 即每点一次 `sqrt`                                                            | dense 且 `n ≥ 16`：`getMaxDistanceRVV` 比较 L2²（`vfredmax` + `vcompress` 取平局下标）；indices 版用 `vluxei32` gather；否则 `getMaxDistanceStandard`      |
| `getMinMax3D`                             | `Eigen::Vector4f` 上 `cwiseMin` / `cwiseMax`                                       | dense 且 `n ≥ 16`：`getMinMax3DRVV`，条带内 `vfmin`/`vfmax` 用 `_tu`，条带外 `vfredmin`/`vfredmax`；indices 版同结构                                      |
| `calculatePolygonArea`                    | `va.cross(vb)` 累加后 `res.norm()`                                                   | `n ≥ 16`：`calculatePolygonAreaRVV`（`vlse` + `vfmsac` 叉积分量 + `vfadd_tu` + `vfredusum`）；最后一条边标量叉积；否则 `calculatePolygonAreaStandard`         |
| `getAngle3D`（标量 API）                      | `normalized().dot` + `std::acos`                                                  | 公开 API 未改；本仓库另提供 `acos_RVV_f32m2`、`getAcuteAngle3DRVV_f32m2`，供 x86 的 SSE/AVX 风格批量角计算在 RVV 上的对应实现                                          |
| `getCircumcircleRadius`、`getMinMax`（直方图）等 | 标量公式或循环                                                                           | 当前与上游同构，未加 RVV 路径                                                                                                                         |


上游 `getMeanStd` 的循环体在常见实现中等价于「每个样本先提升到 `double` 再累加」；本仓库 RVV 内核在向量寄存器内用 `float` 做部分和，再以一次归约进入 `double`，与上游的逐次宽化累加在舍入上可能不同。

---

## 3. 总体设计

**分流条件** 多数向量路径要求 `cloud.is_dense` 且点数不低于阈值（通常为 16），否则调用 `*Standard` 或与上游同构的循环。非 dense 点云保持标量遍历与 `isfinite` 判断，不在向量循环里做不规则掩码。

**组织方式** 标量侧为 `*Standard`；RVV 侧为 `*RVV` ，由 `#if defined(__RVV10__)` 包裹的公开模板转发。`common.h` 在 `__RVV10__` 下包含 `<riscv_vector.h>`，数学辅助与热点集中在 `common.hpp`。

---

## 4. 详细实现

### 4.1 getMeanStdKernelRVV

`pcl::getMeanStd` 在样本数大于 1 时需要一次遍历得到 \(\sum_i x_i\) 与 \(\sum_i x_i^2\)，再在标量里算均值与无偏样本方差。标量内核把每个 `float` 宽化到 `double` 再累加；RVV 内核用连续内存上的条带代替 `for`。

#### 条带累加与 `_tu`

用 `vle32` 按条带加载；`v_acc_sum`、`v_acc_sq` 在多次 `vsetvl` 之间保留。条带末尾若 `vl` 小于 `VLMAX`，累加须用 `vfadd` / `vfmacc` 的 `_tu`（tail undisturbed）形态：默认 `_ta` 可能改写未参与本条累加的 lane，而随后的 `vfredosum_vs(..., max_vl)` 仍按 `max_vl`（即首次 `vsetvl` 得到的 VLMAX）把整寄存器宽都折进去，会把「垃圾 lane」加进和里。详细见文档 [Tail-Agnostic.md](doc-rvv/rvv/Tail-Agnostic-Tail-Undisturbed.zh.md) 。

#### 浮点累加顺序上的取舍

部分和在 `float` 向量里完成，最后一步 `vfredosum` 再落到标量并转 `double`，避免在环里做大量 `vfwcvt` / 双精度向量；代价是舍入顺序与「每样本先 `double` 再累加」不一致，与上游可能差在末几位。

```cpp
inline void
getMeanStdKernelRVV (const float* data, std::size_t n, double& sum, double& sq_sum)
{
  sum = 0;
  sq_sum = 0;
  std::size_t i = 0;
  const std::size_t max_vl = __riscv_vsetvl_e32m2 (n);
  vfloat32m2_t v_acc_sum = __riscv_vfmv_v_f_f32m2 (0.0f, max_vl);
  vfloat32m2_t v_acc_sq  = __riscv_vfmv_v_f_f32m2 (0.0f, max_vl);
  while (i < n)
  {
    const std::size_t vl = __riscv_vsetvl_e32m2 (n - i);
    const vfloat32m2_t v = __riscv_vle32_v_f32m2 (data + i, vl);
    v_acc_sum = __riscv_vfadd_vv_f32m2_tu (v_acc_sum, v_acc_sum, v, vl);
    v_acc_sq  = __riscv_vfmacc_vv_f32m2_tu (v_acc_sq, v, v, vl);
    i += vl;
  }
  vfloat32m1_t v_zero = __riscv_vfmv_s_f_f32m1 (0.0f, 1);
  vfloat32m1_t v_sum  = __riscv_vfredosum_vs_f32m2_f32m1 (v_acc_sum, v_zero, max_vl);
  vfloat32m1_t v_sq   = __riscv_vfredosum_vs_f32m2_f32m1 (v_acc_sq,  v_zero, max_vl);
  sum    = static_cast<double>(__riscv_vfmv_f_s_f32m1_f32 (v_sum));
  sq_sum = static_cast<double>(__riscv_vfmv_f_s_f32m1_f32 (v_sq));
}
```

`pcl::getMeanStd` 在 `values.size() > 1` 时调用该内核；单元素与空向量分支与上游一致。

### 4.2 入口与函数分发

`common.hpp` 里多条热点路径共用同一套组织方式：

1. `pcl::Foo` 只做 `#if defined(__RVV10__)` 二选一：编译期在 `FooRVV` 与 `FooStandard` 之间切换，公开模板内部不再嵌套 `#if`。
2. `FooRVV` 首段判断是否需要向量：点云类接口在 `!cloud.is_dense || n < 16` 时直接调用对应的 `FooStandard`，与 `getMaxDistanceRVV`、`getMinMax3DRVV` 等一致；`calculatePolygonAreaRVV` 无 `is_dense` 字段，退回条件为 `num_points < 16`。
3. `FooStandard` 保留与上游等价的标量语义（含 dense / 非 dense 分支）。`getPointsInBoxStandard` 与 `getPointsInBoxRVV` 在同一条骨架下把 dense 标量循环与 dense 向量条带分别折叠进两个符号；其它接口仍保留独立的 `*Standard` 函数名。

`getMeanStd` 不经过点云：在 `values.size() > 1` 时对 `getMeanStdKernelRVV` / `getMeanStdKernelStandard` 做上述第 1 点式的二选一；内核层没有 `is_dense`，也不在 `KernelRVV` 内再退回。

#### 符号与退回条件

| `pcl` 符号（或内核） | 标量 / 回退目标 | RVV 入口 | `*RVV` 内退回标量条件 |
| --- | --- | --- | --- |
| `getMeanStd`（`size()>1`） | `getMeanStdKernelStandard` | `getMeanStdKernelRVV` | 无（不调用 Standard） |
| `getPointsInBox` | `getPointsInBoxStandard` | `getPointsInBoxRVV` | 非 dense 或 `n < 16` |
| `getMaxDistance`（全云 / 带 `indices`） | `getMaxDistanceStandard` | `getMaxDistanceRVV` | 同上（`n` 为 `cloud.size()` 或 `indices.size()`） |
| `getMinMax3D`（`Eigen::Vector4f`，全云 / 带 `indices`） | `getMinMax3DStandard` | `getMinMax3DRVV` | 同上 |
| `getMinMax3D`（`PointT` 输出） | 经 `Eigen` 重载，实质同上 | 同上 | 同上 |
| `calculatePolygonArea` | `calculatePolygonAreaStandard` | `calculatePolygonAreaRVV` | `num_points < 16` |

#### 公开入口代码形态

下列片段形态一致；`getPointsInBox` 在调用前后多 `indices.resize`，因其按写入个数收缩缓冲区。

```cpp
template<typename PointT> inline void
pcl::getMaxDistance (const pcl::PointCloud<PointT> &cloud, const Eigen::Vector4f &pivot_pt, Eigen::Vector4f &max_pt)
{
#if defined(__RVV10__)
  getMaxDistanceRVV (cloud, pivot_pt, max_pt);
#else
  getMaxDistanceStandard (cloud, pivot_pt, max_pt);
#endif
}
```

```cpp
template <typename PointT> inline void
pcl::getMinMax3D (const pcl::PointCloud<PointT> &cloud, Eigen::Vector4f &min_pt, Eigen::Vector4f &max_pt)
{
#if defined(__RVV10__)
  getMinMax3DRVV (cloud, min_pt, max_pt);
#else
  getMinMax3DStandard (cloud, min_pt, max_pt);
#endif
}
```

带 `indices` 的 `getMinMax3D` 与 `getMaxDistance` 第二重载同一写法。多边形面积：

```cpp
template <typename PointT> inline float
pcl::calculatePolygonArea (const pcl::PointCloud<PointT> &polygon)
{
#if defined(__RVV10__)
  return calculatePolygonAreaRVV (polygon);
#else
  return calculatePolygonAreaStandard (polygon);
#endif
}
```

`calculatePolygonAreaRVV` 在顶点数不足 16 时直接 `return calculatePolygonAreaStandard (polygon)`。

`getPointsInBox` 的 `*RVV` 早退与公开入口示例：

```cpp
  const std::size_t n = cloud.size ();
  if (!cloud.is_dense || n < 16)
    return getPointsInBoxStandard (cloud, min_pt, max_pt, indices);
```

```cpp
pcl::getPointsInBox (const pcl::PointCloud<PointT> &cloud,
                     Eigen::Vector4f &min_pt, Eigen::Vector4f &max_pt,
                     Indices &indices)
{
  indices.resize (cloud.size ());
  int l;
#if defined(__RVV10__)
  l = getPointsInBoxRVV (cloud, min_pt, max_pt, indices);
#else
  l = getPointsInBoxStandard (cloud, min_pt, max_pt, indices);
#endif
  indices.resize (l);
}
```

### 4.3 getPointsInBoxRVV

在轴对齐盒（`min_pt` / `max_pt` 给出各轴 `[min,max]`）内筛选点，把命中点的全局下标顺序写入 `indices` 前缀，返回个数 `l`。标量路径 `getPointsInBoxStandard` 对 dense 逐点六比较；非 dense 另做 `isfinite` 检查。

#### 条带内：vlse32 与盒内掩码

`stride`、`base` 与盒边界标量 `min_x`…`max_z` 在进入 `while` 前取好；`ptr_x/y/z` 指向条带起点处各分量。`vlse32` 步长为 `sizeof(PointT)`，对应 AoS。各轴用 `vmfge`/`vmfle` 与 `min_*`、`max_*` 比较，`vmand` 合成 `mask`：三轴同时落在闭区间内为真。

```cpp
  const std::size_t stride = sizeof (PointT);
  const uint8_t* base = reinterpret_cast<const uint8_t*>(cloud.data ());
  const float min_x = min_pt[0], min_y = min_pt[1], min_z = min_pt[2];
  const float max_x = max_pt[0], max_y = max_pt[1], max_z = max_pt[2];
  int l = 0;
  std::size_t i = 0;
  while (i < n)
  {
    const std::size_t vl = __riscv_vsetvl_e32m2 (n - i);
    const float* ptr_x = reinterpret_cast<const float*>(base + i * stride + offsetof (PointT, x));
    const float* ptr_y = reinterpret_cast<const float*>(base + i * stride + offsetof (PointT, y));
    const float* ptr_z = reinterpret_cast<const float*>(base + i * stride + offsetof (PointT, z));
    const vfloat32m2_t vx = __riscv_vlse32_v_f32m2 (ptr_x, stride, vl);
    const vfloat32m2_t vy = __riscv_vlse32_v_f32m2 (ptr_y, stride, vl);
    const vfloat32m2_t vz = __riscv_vlse32_v_f32m2 (ptr_z, stride, vl);

    vbool16_t in_x = __riscv_vmfge_vf_f32m2_b16 (vx, min_x, vl);
    in_x = __riscv_vmand_mm_b16 (in_x, __riscv_vmfle_vf_f32m2_b16 (vx, max_x, vl), vl);
    vbool16_t in_y = __riscv_vmfge_vf_f32m2_b16 (vy, min_y, vl);
    in_y = __riscv_vmand_mm_b16 (in_y, __riscv_vmfle_vf_f32m2_b16 (vy, max_y, vl), vl);
    vbool16_t in_z = __riscv_vmfge_vf_f32m2_b16 (vz, min_z, vl);
    in_z = __riscv_vmand_mm_b16 (in_z, __riscv_vmfle_vf_f32m2_b16 (vz, max_z, vl), vl);
    vbool16_t mask = __riscv_vmand_mm_b16 (__riscv_vmand_mm_b16 (in_x, in_y, vl), in_z, vl);
```

#### vid、vcompress 与写出

`vid` 给出条带内 lane 序号，与 `i` 相加得到全局点下标；`vcompress` 按 `mask` 把命中下标压到向量低段（前 `cnt` 个有效）。`vcpop` 统计命中个数；`cnt > 0` 时用 `vse32` 写入 `indices.data() + l`，并 `l += cnt`。条带按 `i` 递增扫描，故输出下标整体随点序递增。

```cpp
    const vuint32m2_t vid = __riscv_vadd_vx_u32m2 (__riscv_vid_v_u32m2 (vl), static_cast<uint32_t> (i), vl);
    const vuint32m2_t compressed = __riscv_vcompress_vm_u32m2 (vid, mask, vl);
    const std::size_t cnt = __riscv_vcpop_m_b16 (mask, vl);
    if (cnt > 0)
    {
      // cnt <= vl <= VLMAX for this strip; vcompress packs the first cnt lanes.
      const std::size_t vl_store = __riscv_vsetvl_e32m2 (cnt);
      std::uint32_t* const out_u32 =
          reinterpret_cast<std::uint32_t*> (indices.data () + l);
      __riscv_vse32_v_u32m2 (out_u32, compressed, vl_store);
      l += static_cast<int>(cnt);
    }
    i += vl;
  }
  return l;
```

### 4.4 getMaxDistanceRVV

在给定枢轴点 `pivot` 下，找点云中（或 `indices` 子集上）欧氏距离最远的那个点，并把其坐标写入 `max_pt`。标量实现每点调用 `.norm()`，即对 \(\sqrt{dx^2+dy^2+dz^2}\) 求最大值。

#### 比较距离的平方

对非负量，\(\arg\max \sqrt{f} = \arg\max f\)（此处 \(f\) 为平方和）。向量路径只维护 `v_d2 = dx^2+dy^2+dz^2`，省略 `sqrt`；输出仍是最远点坐标，与「距离最大」一致。

#### 条带内最大值与平局下标

每一拍用 `vfredmax_vs` 得到当前条带内最大的 `v_d2`。若刷新全局最大，需要对应全局下标：将 `chunk_max` 广播回向量，用 `vmfeq` 标出 `v_d2` 等于该值的 lane；`vid` 与条带基址 `i` 相加得全局索引，再 `vcompress` 挤到低段，取首元素作为本条候选。平局时取较小全局下标，与从低索引扫描的标量习惯一致。

#### indices 处理

条带加载索引数组，用 `vluxei32` 以 `index * sizeof(PointT)` 为偏移从点云基址 gather `x/y/z`，避免先拷贝子集为连续缓冲。

```cpp
  while (i < n) {
    const std::size_t vl = __riscv_vsetvl_e32m2(n - i);
    const float* ptr_x =
        reinterpret_cast<const float*>(base + i * stride + offsetof(PointT, x));
    const float* ptr_y =
        reinterpret_cast<const float*>(base + i * stride + offsetof(PointT, y));
    const float* ptr_z =
        reinterpret_cast<const float*>(base + i * stride + offsetof(PointT, z));
    const vfloat32m2_t vx = __riscv_vlse32_v_f32m2(ptr_x, stride, vl);
    const vfloat32m2_t vy = __riscv_vlse32_v_f32m2(ptr_y, stride, vl);
    const vfloat32m2_t vz = __riscv_vlse32_v_f32m2(ptr_z, stride, vl);
    const vfloat32m2_t v_dx = __riscv_vfrsub_vf_f32m2(vx, px, vl);
    const vfloat32m2_t v_dy = __riscv_vfrsub_vf_f32m2(vy, py, vl);
    const vfloat32m2_t v_dz = __riscv_vfrsub_vf_f32m2(vz, pz, vl);
    const vfloat32m2_t v_d2 = __riscv_vfmacc_vv_f32m2(
        __riscv_vfmacc_vv_f32m2(__riscv_vfmul_vv_f32m2(v_dx, v_dx, vl), v_dy, v_dy, vl),
        v_dz,
        v_dz,
        vl);
    const vfloat32m1_t v_max =
        __riscv_vfredmax_vs_f32m2_f32m1(v_d2, __riscv_vfmv_s_f_f32m1(-1.0f, 1), vl);
    const float chunk_max = __riscv_vfmv_f_s_f32m1_f32(v_max);
    if (chunk_max > max_chunk) {
      const vfloat32m2_t v_broadcast = __riscv_vfmv_v_f_f32m2(chunk_max, vl);
      const vbool16_t mask = __riscv_vmfeq_vv_f32m2_b16(v_d2, v_broadcast, vl);
      const vuint32m2_t vid =
          __riscv_vadd_vx_u32m2(__riscv_vid_v_u32m2(vl), static_cast<uint32_t>(i), vl);
      const vuint32m2_t comp = __riscv_vcompress_vm_u32m2(vid, mask, vl);
      idx_chunk = static_cast<int>(__riscv_vmv_x_s_u32m2_u32(comp));
      max_chunk = chunk_max;
    }
    i += vl;
  }
```

### 4.5 getMinMax3DRVV

求点云（或 `indices` 子集）在 `x,y,z` 上的轴对齐包围盒，写入 `min_pt` / `max_pt` 的前三分量。标量路径用 Eigen 的 `cwiseMin` / `cwiseMax` 逐点更新。

#### 条带内归并与 `_tu`

每个 `vl` 用 `vlse32`（或 indices 版的 `vluxei32`）取一批点的 `x,y,z`。六路寄存器分别维护当前已见的最小/最大 `x,y,z`；对每条新向量做 `vfmin` / `vfmax` 的 `_tu`。原因与 4.1 类似：末条 `vl` 变短时，`_ta` 可能弄脏高 lane，而后面 `vfredmin` / `vfredmax` 仍按 `vlmax` 扫整寄存器，会读到无效极值。

#### 条带外归约

全部点扫完后，对六个 `v_acc_*` 各做一次 `vfredmin` / `vfredmax`（`vl` 用 `vlmax`），得到全局 min/max；`Eigen::Vector4f` 的第四分量置 0。

```cpp
  while (i < n)
  {
    const std::size_t vl = __riscv_vsetvl_e32m2 (n - i);
    const float* ptr_x =
        reinterpret_cast<const float*>(base + i * stride + offsetof (PointT, x));
    // ... ptr_y, ptr_z, vlse32 ...
    const vfloat32m2_t vx = __riscv_vlse32_v_f32m2 (ptr_x, stride, vl);
    const vfloat32m2_t vy = __riscv_vlse32_v_f32m2 (ptr_y, stride, vl);
    const vfloat32m2_t vz = __riscv_vlse32_v_f32m2 (ptr_z, stride, vl);
    v_acc_min_x = __riscv_vfmin_vv_f32m2_tu (v_acc_min_x, v_acc_min_x, vx, vl);
    v_acc_min_y = __riscv_vfmin_vv_f32m2_tu (v_acc_min_y, v_acc_min_y, vy, vl);
    v_acc_min_z = __riscv_vfmin_vv_f32m2_tu (v_acc_min_z, v_acc_min_z, vz, vl);
    v_acc_max_x = __riscv_vfmax_vv_f32m2_tu (v_acc_max_x, v_acc_max_x, vx, vl);
    v_acc_max_y = __riscv_vfmax_vv_f32m2_tu (v_acc_max_y, v_acc_max_y, vy, vl);
    v_acc_max_z = __riscv_vfmax_vv_f32m2_tu (v_acc_max_z, v_acc_max_z, vz, vl);
    i += vl;
  }
```

### 4.6 calculatePolygonAreaRVV

对三维多边形顶点（通常共面），面积公式为 \(\tfrac{1}{2}\|\sum_i \mathbf{v}_i \times \mathbf{v}_{i+1}\|\)，下标模 \(n\) 闭合。标量实现用 `Eigen::Vector3f::cross` 累加 `res` 再 `res.norm()`。

#### 向量边 \((i,i+1)\)

点云为 AoS，对条带用 `vlse32` 同时加载一批边的两端点坐标（步长 `sizeof(PointT)`）。叉积分量与标量 \(\mathbf{a}\times\mathbf{b}\) 同分量式。`vfmsac.vv` 的规范为 \(vd \leftarrow vs1 \times vs2 - vd_{\text{old}}\)（与 `vfnmsac` 不同），代码用 `vfmul` 与 `vfmsac` 的操作数顺序与 Eigen 对齐。

#### 累加、归约与闭合边

各分量在条带间用 `vfadd_tu` 累到 `v_acc_*`；最后用 `vfredusum` 折到标量。边 \((0,1)\ldots(n-2,n-1)\) 在向量化循环里处理；闭合边 \((n-1,0)\) 只出现一次且跨过头尾，用标量叉积加进 `rx,ry,rz`，避免为单条边再写一套掩码循环。

```cpp
    const vfloat32m2_t cx = __riscv_vfmsac_vv_f32m2 (__riscv_vfmul_vv_f32m2 (az, by, vl), ay, bz, vl);
    const vfloat32m2_t cy = __riscv_vfmsac_vv_f32m2 (__riscv_vfmul_vv_f32m2 (ax, bz, vl), az, bx, vl);
    const vfloat32m2_t cz = __riscv_vfmsac_vv_f32m2 (__riscv_vfmul_vv_f32m2 (ay, bx, vl), ax, by, vl);
    v_acc_x = __riscv_vfadd_vv_f32m2_tu (v_acc_x, v_acc_x, cx, vl);
    v_acc_y = __riscv_vfadd_vv_f32m2_tu (v_acc_y, v_acc_y, cy, vl);
    v_acc_z = __riscv_vfadd_vv_f32m2_tu (v_acc_z, v_acc_z, cz, vl);
    i += vl;
  }

  vfloat32m1_t v_zero = __riscv_vfmv_s_f_f32m1 (0.0f, 1);
  float rx = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredusum_vs_f32m2_f32m1 (v_acc_x, v_zero, vlmax));
  float ry = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredusum_vs_f32m2_f32m1 (v_acc_y, v_zero, vlmax));
  float rz = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredusum_vs_f32m2_f32m1 (v_acc_z, v_zero, vlmax));

  // Last pair (n-1, 0)
  const float ax = polygon[n - 1].x, ay = polygon[n - 1].y, az = polygon[n - 1].z;
  const float bx = polygon[0].x, by = polygon[0].y, bz = polygon[0].z;
  rx += ay * bz - az * by;
  // ...
```

### 4.7 数学辅助：acos_RVV、getAcuteAngle3DRVV、atan2、expf

#### getAcuteAngle3DRVV

`getAngle3D` 公开 API 仍是双向量夹角（`std::acos`）；批量场景在 x86 上用 SSE/AVX 的 `acos` 近似与 `getAcuteAngle3D*`。RVV 侧提供同系数结构的 `acos_RVV_f32m2`，以及 `getAcuteAngle3DRVV_f32m2`：先算点积，`vfsgnjx` 得到 \(|\mathbf{u}\cdot \mathbf{v}|\)，`vfmin` 夹到 \([0,1]\) 后调用 `acos_RVV`，与锐角定义及 SSE 路径对齐。

#### atan2_RVV_f32m2

对 `|x|,|y|` 比较后可能交换分子分母，把比值限制在多项式有效区间，再用 `vmerge` 补象限，避免逐元素 `std::atan2`。实现文档请查阅同目录下的 [`atan2-RVV.zh.md`](atan2-RVV.zh.md)。

#### expf_RVV_f32m2

通过区间约化、多项式逼近、`2^k` 重组等方式实现。实现文档请查阅同目录下的  [`expf-RVV.zh.md`](expf-RVV.zh.md)。

---

## 5. 测试与验证

可执行文件与 Makefile 位于 `test-rvv/common/common/`。板卡侧一次完整对比的原始分项输出在：

- `test-rvv/common/common/output/board/bench_std.log`（标量或 `__RVV10__` 关闭时的基线）
- `test-rvv/common/common/output/board/bench_rvv.log`（RVV 构建）
- `test-rvv/common/common/output/board/bench_compare.log`（由 `analyze_bench_compare` 汇总；表头中的绝对路径为生成环境路径，与仓库内文件名可能不一致）

下列数据摘自 `bench_compare.log`：设备 Milkv-Jupiter，数据集规模 点云 200000 点、向量 500000 元素，每行迭代 20 次（Total = Avg × 20）。Speedup 列在表中为 RVV 相对 Std 的加速比。


| Benchmark 项                            | Std Avg (ms) | RVV Avg (ms) | Speedup |
| -------------------------------------- | ------------ | ------------ | ------- |
| getMeanStd (vector)                    | 2.5670       | 0.3241       | 7.92×   |
| getPointsInBox (dense cloud)           | 4.7103       | 1.2424       | 3.79×   |
| getMaxDistance (cloud, pivot)          | 5.5943       | 1.0643       | 5.26×   |
| getMaxDistance (cloud, indices, pivot) | 6.2503       | 1.2937       | 4.83×   |
| getMinMax3D (cloud, Eigen)             | 8.5031       | 0.7605       | 11.18×  |
| getMinMax3D (cloud, indices, Eigen)    | 9.0152       | 1.0292       | 8.76×   |
| getMinMax3D (cloud, PointT)            | 8.5023       | 0.7656       | 11.11×  |
| getAngle3D (x1000)                     | 0.1791       | 0.0073       | 24.53×  |
| calculatePolygonArea (256 pts) ×500    | 1.8608       | 0.6479       | 2.87×   |


同一目录下另有 `expf_test.log`、`atan2_test.log`、`expf_remez_vs_taylor.log` 等数学辅助单测输出，可与 `doc-rvv/common` 下文档交叉查阅。

---

## 6. 总结

本文件中的 RVV 扩展在接口上保持上游形态；在实现上将热点拆成步进 load / gather、向量算术与浮点归约。跨条带保留状态的累加或极值向量使用 `_tu` 内建，避免 tail 与 `vfred` 长度组合产生静默错误。`getMeanStd` 的 float 条带累加与上游 double 逐元累加存在已知数值差异，若需要与标量逐位一致，应走标量路径或收紧测试阈值。
