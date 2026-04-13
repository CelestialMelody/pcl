# RVV AOS 数据加载与写回策略

本报告汇总了 PCL 在 RISC-V 矢量扩展（RVV）优化过程中，针对点云 / 结构体数组（AoS）布局下的数据加载与写回策略研究。

---

## 1. 隔离微基准：`test-rvv/rvv/load_store`

### 1.1 测试程序说明

- `bench_rvv_load_compare`：按 Strided AoS → Contiguous（紧密交错 xyz）→ Indexed AoS gather 三组对比 load 形态。
- `bench_rvv_store_compare`：按 Strided AoS → Contiguous → Indexed scatter 对比 store / scatter 形态（含 4 字段与 3 字段 xyz 子场景）。

### 1.2 板卡实测结果：`output/board/load_store.log`

测试环境基于 Milk-V Jupiter 开发板，参数配置为 `n_points=262144`，迭代 `50` 次。

#### 1.2.1 Load（`bench_rvv_load_compare`）

| 场景 | Mode A（拆分指令） | Mode B（segment / 合并形态） | speedup (A/B) |
| --- | --- | --- | --- |
| Strided AoS（顺序遍历结构体，`x/y/z` 固定 stride） | `3× vlse32` → 2.108906 ms/iter | `vlsseg3e32` → 0.742259 ms/iter | 2.841× |
| Contiguous（`xyzxyz...` 紧密交错，stride=12） | `3× vlse32` → 1.054804 ms/iter | `vlseg3e32` → 0.661796 ms/iter | 1.594× |
| Indexed AoS gather（间接索引） | `3× vluxei32` → 53.119016 ms/iter | `vluxseg3ei32` → 52.253730 ms/iter | 1.017× |

分析：

- 顺序 / 等步长（strided 或 contiguous）：在顺序或等步长场景下，`vlsseg` / `vlseg` 类指令通过减少前端指令译码与发射压力，结合访存合并（Memory Coalescing），能够获得显著增益。
- 索引 gather：在 `Indexed Gather` 场景中，加速比缩减至 2% 以内。此时瓶颈已由指令条数转移至硬件的随机访存延迟，分段加载的语义优势被离散地址引发的 Cache Miss 掩盖。

#### 1.2.2 Store（`bench_rvv_store_compare`）

| 场景 | Mode A | Mode B | speedup (A/B) |
| --- | --- | --- | --- |
| Strided AoS — 4 字段 | `4× vsse32`（strided_store4_fields）4.360192 ms/iter | `vssseg4`（strided_store4_seg）3.990060 ms/iter | 1.093× |
| Contiguous — 4 字段（SoA 四连续数组） | `4× vse32` 2.691890 ms/iter | `vsseg4e32` 1.458553 ms/iter | 1.846× |
| Indexed scatter — 4 字段 | `4× vsuxei32` 20.783056 ms/iter | `vsuxseg4ei32` 16.721060 ms/iter | 1.243× |
| Strided AoS — 3 字段（xyz） | strided_store3_fields 4.322889 ms/iter | strided_store3_seg（`vssseg3`）3.982744 ms/iter | 1.085× |
| Contiguous — 3 字段（packed xyz） | `3× vse32` 1.899198 ms/iter | `vsseg3e32` 0.452573 ms/iter | 4.196× |
| Indexed scatter — 3 字段 | scatter_store3_fields 19.718161 ms/iter | scatter_store3_seg 16.442615 ms/iter | 1.199× |

分析：

- 连续写回（contiguous）：`vsseg3` / `vsseg4` 相对多次 `vse` 往往优势最大，与写合并、指令条数减少都相符。
- Strided AoS：段存储相对多次 `vsse` 约有 8～10% 量级提升，需结合字段布局与是否真满足 segment 对齐前提再改生产代码。
- Indexed scatter：`vsuxseg` 相对多次 `vsuxei` 约有 20～25% 量级提升，仍弱于 contiguous 下的倍数，主因仍是离散写地址。

---

## 2. 数据加载（`vluxei32` vs `vluxseg3ei32`）实测

### 2.1 背景说明

在 `sample_consensus/include/pcl/sample_consensus/impl/sac_model_normal_plane.hpp` 的 `countWithinDistanceRVV()` 中，曾经存在两种加载策略用于对比：

```cpp
#ifdef PCL_RVV_BENCHMARK_USE_VLUXSEG
    // --- Mode A: Indexed Segment Load (vluxseg3ei32) ---

    // 1) Load PointT (x, y, z) in a single instruction.
    const vfloat32m2x3_t v_xyz = __riscv_vluxseg3ei32_v_f32m2x3(
        reinterpret_cast<const float*>(points_base + offsetof(PointT, x)), v_off_pt, vl);

    const vfloat32m2_t v_px = __riscv_vget_v_f32m2x3_f32m2(v_xyz, 0);
    const vfloat32m2_t v_py = __riscv_vget_v_f32m2x3_f32m2(v_xyz, 1);
    const vfloat32m2_t v_pz = __riscv_vget_v_f32m2x3_f32m2(v_xyz, 2);

    // Byte offsets for PointNT
    const vuint32m2_t v_off_norm = __riscv_vmul_vx_u32m2(v_idx, sizeof(PointNT), vl);

    // 2) Load PointNT (nx, ny, nz) in a single transaction.
    const vfloat32m2x3_t v_nxyz = __riscv_vluxseg3ei32_v_f32m2x3(
        reinterpret_cast<const float*>(normals_base + offsetof(PointNT, normal_x)), v_off_norm, vl);

    const vfloat32m2_t v_nx = __riscv_vget_v_f32m2x3_f32m2(v_nxyz, 0);
    const vfloat32m2_t v_ny = __riscv_vget_v_f32m2x3_f32m2(v_nxyz, 1);
    const vfloat32m2_t v_nz = __riscv_vget_v_f32m2x3_f32m2(v_nxyz, 2);
#else
    // --- Mode B: Standard Gather (vluxei32) ---

    const vfloat32m2_t v_px = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(points_base + offsetof(PointT, x)), v_off_pt, vl);
    const vfloat32m2_t v_py = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(points_base + offsetof(PointT, y)), v_off_pt, vl);
    const vfloat32m2_t v_pz = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(points_base + offsetof(PointT, z)), v_off_pt, vl);

    // Byte offsets for PointNT
    const vuint32m2_t v_off_norm = __riscv_vmul_vx_u32m2(v_idx, sizeof(PointNT), vl);

    const vfloat32m2_t v_nx = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(normals_base + offsetof(PointNT, normal_x)), v_off_norm, vl);
    const vfloat32m2_t v_ny = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(normals_base + offsetof(PointNT, normal_y)), v_off_norm, vl);
    const vfloat32m2_t v_nz = __riscv_vluxei32_v_f32m2(
        reinterpret_cast<const float*>(normals_base + offsetof(PointNT, normal_z)), v_off_norm, vl);
#endif
```

- Mode A：`vluxseg3ei32`（indexed segment load）
  - 通过 segment load 一次性加载三元组（`x,y,z` 或 `nx,ny,nz`）
  - 使用元组寄存器类型（例如 `vfloat32m2x3_t`），再 `vget` 提取分量
- Mode B：`vluxei32`（gather）
  - 分别 gather `x/y/z` 与 `nx/ny/nz`（多条指令）
  - 通过 `base + offsetof(field)` + `byte_offsets` 访问结构体成员

二者在算法层面等价，主要差异来自：指令数量、寄存器压力、以及具体微架构对指令的实现质量。

### 2.2 板卡实测与分析

- 测试位置：`test-rvv/sample_consensus/plane_models/bench_sac_normal_plane_load_compare`
- 参数：`iters=200, warmup=5, threshold=0.05, normal_weight=0.1`
- 日志：`test-rvv/sample_consensus/plane_models/output/board/bench_load_compare.log`

测试在同一 binary 内对比两种路径，并检查两者 `check_count` 一致。

结果（多次运行）：

- gather（`vluxei32`）：约 `0.0759 ~ 0.0784 ms/iter`
- segment gather（`vluxseg3ei32`）：约 `0.0626 ~ 0.0692 ms/iter`
- speedup（gather/seg）：约 `1.10x ~ 1.24x`（多数 run 对 `vluxseg3ei32` 更有利，但仍存在波动）

分析：

- `vluxseg3ei32` 多数情况下更快，但波动仍存在。
- 与隔离微基准中 Indexed AoS 仅 ~1.017× 的结论并不矛盾：真实内核除 indexed load 外还有距离、角度、`curvature` 等额外 load 与计算，整体比例被重塑；且 `n`、热循环结构、编译器调度均不同。

---

## 3. 数据写回（`vsse32` vs `vssseg4e32`）实测

### 3.1 背景说明

测试 `pcl::2d::impl::edge.hpp`（Sobel/Prewitt/Canny 等）中 `PointXYZIEdge` 相关 RVV 写回策略对比：
`magnitude / direction / magnitude_x / magnitude_y` 这 4 个相邻 `float` 的写回选择：4 次 `vsse32` 还是 1 次段存储 `vssseg4e32`。

### 3.2 板卡实测与分析

- 测试位置：`test-rvv/2d/test_edge_store_bench.cpp`
- 参数：数组规模 `n = 640 * 480 = 307200`，`iterations = 20`，`warmup = 5`
- 对比模式：
  - Mode A：`4× __riscv_vsse32_v_f32m2` 分别写 4 个字段
  - Mode B：`1× __riscv_vssseg4e32_v_f32m2x4` 段存储写 4 个字段

Mode A 与 Mode B 的核心差异如下：

```cpp
vfloat32m2x4_t vt = __riscv_vlsseg4e32_v_f32m2x4(seg_base_in, stride_bytes, vl);

// Mode A: 4x vsse32 (one per field)
vfloat32m2_t v_mag = __riscv_vget_v_f32m2x4_f32m2(vt, 0);
vfloat32m2_t v_dir = __riscv_vget_v_f32m2x4_f32m2(vt, 1);
vfloat32m2_t v_mx  = __riscv_vget_v_f32m2x4_f32m2(vt, 2);
vfloat32m2_t v_my  = __riscv_vget_v_f32m2x4_f32m2(vt, 3);
__riscv_vsse32_v_f32m2(out_mag, stride_bytes, v_mag, vl);
__riscv_vsse32_v_f32m2(out_dir, stride_bytes, v_dir, vl);
__riscv_vsse32_v_f32m2(out_mx,  stride_bytes, v_mx,  vl);
__riscv_vsse32_v_f32m2(out_my,  stride_bytes, v_my,  vl);

// Mode B: 1x vssseg4e32 (segment store of 4 fields)
__riscv_vssseg4e32_v_f32m2x4(seg_base_out, stride_bytes, vt, vl);
```

分析与结论：

结果（多次运行，`test-rvv/2d/output/board/bench_store_compare.log` 摘录）：

- Mode A：约 `6.59 ~ 7.04 ms/iter`
- Mode B：约 `6.71 ~ 7.15 ms/iter`
- speedup（A/B）：约 `0.966x ~ 1.020x`

分析：

- 两种写回方式差异接近测量噪声范围，未见稳定收益。

---

## 4. 数据加载（`vlse32` vs `vlsseg3e32`）补充：`getMaxSegmentRVV`

### 4.1 背景说明

针对 `common/include/pcl/common/distances.h` 的 `getMaxSegmentRVV()` 内层加载（`PointXYZ` 的 `x/y/z`），对比：

- Mode A：`3x vlse32`（分别加载 `x/y/z`）
- Mode B：`vlsseg3e32`（一次段加载 `x,y,z`）

### 4.2 板卡实测与分析

- 参数：`n=2500, iters=20, warmup=3`
- 位置：`test-rvv/common/distances` 相关 bench

Mode A 与 Mode B 的核心差异如下（内层 `j` 循环对 `PointT` 的 `x/y/z` 加载）：

```cpp
const ptrdiff_t stride = static_cast<ptrdiff_t>(sizeof(PointT));

// Mode A: 3x vlse32 (strided loads for x/y/z)
const vfloat32m2_t vx = __riscv_vlse32_v_f32m2(&cloud[j].x, stride, vl);
const vfloat32m2_t vy = __riscv_vlse32_v_f32m2(&cloud[j].y, stride, vl);
const vfloat32m2_t vz = __riscv_vlse32_v_f32m2(&cloud[j].z, stride, vl);

// Mode B: 1x vlsseg3e32 (segment strided load of x,y,z)
const vfloat32m2x3_t v_xyz = __riscv_vlsseg3e32_v_f32m2x3(&cloud[j].x, stride, vl);
const vfloat32m2_t vx = __riscv_vget_v_f32m2x3_f32m2(v_xyz, 0);
const vfloat32m2_t vy = __riscv_vget_v_f32m2x3_f32m2(v_xyz, 1);
const vfloat32m2_t vz = __riscv_vget_v_f32m2x3_f32m2(v_xyz, 2);
```

结果（多次运行）：

- Mode A：约 `13.92 ~ 14.45 ms/iter`
- Mode B：约 `12.16 ~ 12.67 ms/iter`
- speedup（A/B）：约 1.14×（较稳定）

分析：

- “连续字段 + 大量重复访问”的场景，`vlsseg3e32` 更容易体现“少发指令/少寻址”的收益。
- 与隔离微基准中 Strided AoS 下 ~2.84×（`load_store.log`，`n=262144`）相比，倍数更小是正常的：`getMaxSegmentRVV` 含完整距离与 argmax 逻辑，load 仅占一部分；且 `n`、编译内联、LMUL 等均不同。
