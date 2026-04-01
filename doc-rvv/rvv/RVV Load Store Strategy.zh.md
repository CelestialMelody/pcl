# RVV AOS 数据加载与写回策略

## 数据加载方式（vluxei32 vs vluxseg3ei32）

本文用于记录 `pcl::SampleConsensusModelNormalPlane` 的 RVV 实现中，**点/法线数据加载方式**的选择依据与实测结论。

### 背景与对比点

在 `sample_consensus/include/pcl/sample_consensus/impl/sac_model_normal_plane.hpp` 的 `countWithinDistanceRVV()` 中，曾经存在两种加载策略用于对比：

```cpp
#ifdef PCL_RVV_BENCHMARK_USE_VLUXSEG
    // --- Approach B: Indexed Segment Load (vluxseg3ei32) ---

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
    // --- Approach A: Standard Gather (vluxei32) ---

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

- **Approach A：`vluxei32`（gather）**
  - 分别 gather `x/y/z` 与 `nx/ny/nz`（多条指令）
  - 通过 `base + offsetof(field)` + `byte_offsets` 访问结构体成员
- **Approach B：`vluxseg3ei32`（indexed segment load）**
  - 通过 segment load 一次性加载三元组（`x,y,z` 或 `nx,ny,nz`）
  - 使用元组寄存器类型（例如 `vfloat32m2x3_t`），再 `vget` 提取分量

二者在算法层面等价，主要差异来自：**指令数量、寄存器压力、以及具体微架构对指令的实现质量**。

---

### `vluxei32`（gather）

- **寄存器压力可能更小**
  - `vluxseg3` 会产生元组寄存器（例如 `m2x3`），等价于一次性占用连续的 3 组寄存器。
  - 当后续计算链很长、临时变量很多时，可能更容易触发 spilling（寄存器溢出到栈），导致性能反而下降。
- **字段加载更灵活**
  - 当只需要结构体里的部分字段，gather 可以更“按需取数”；segment 则倾向于成组加载。

### `vluxseg3ei32`（segment）

- **语义更贴近“加载一个点/一个法线三元组”**
  - 对 `PointXYZ` / `Normal` 这种典型 SoA-like 访问（按索引抓取结构体内的连续字段）更直观，代码可读性更好。
- **通常指令条数更少**
  - 理论上 segment load 可以减少多次 gather 的指令发射/译码开销。

### 板卡实测结论（Milk-V Jupiter，真实硬件）

在板卡上使用同一份输入 `sac_plane_test.pcd`（Points=3283），迭代 200 次，分别运行（多次重复）：

- `bench_sac_normal_plane_load_vluxseg`
- `bench_sac_normal_plane_load_gather`

测试显示：

- **gather（vluxei32）**：约 `0.0603 ~ 0.0697 ms/iter`（分布更集中）
- **vluxseg（vluxseg3ei32）**：约 `0.0594 ~ 0.0770 ms/iter`（波动更明显）

结论：

- **平均耗时差距极小**，但 **gather 在该板卡上更稳定**（方差更小、极端慢的尾部更少）。
- 这类“稳定性”在工程上同样重要：更稳定意味着更可预期的延迟，也更利于后续做性能回归分析。

### 最终策略

**优先采用** `vluxei32`**（gather）作为加载实现**（在 Milk-V Jupiter 上更稳定，且平均性能接近）。

---

## 数据写回方式 （`vsse32` vs `vssseg4e32`）

本文用于记录 `pcl::2d::impl::edge.hpp`（Sobel/Prewitt/Canny 等）中 `PointXYZIEdge` 相关 RVV 写回策略的实测结论：是否能把
`magnitude / direction / magnitude_x / magnitude_y` 这 4 个相邻 `float` 的写回从 4 次 `vsse32` 合并为 1 次段存储 `vssseg4e32`。

### 实验设置

使用独立微基准：`test-rvv/2d/test_edge_store_bench.cpp`。

- 背景：`edge.hpp` 中的写回点

```cpp
// pcl/2d/include/pcl/2d/impl/edge.hpp (computeMagnitudeDirectionRVV)
float* out_mx  = reinterpret_cast<float*>(base_out + j0 * stride_out + off_out_magnitude_x);
float* out_my  = reinterpret_cast<float*>(base_out + j0 * stride_out + off_out_magnitude_y);
float* out_mag = reinterpret_cast<float*>(base_out + j0 * stride_out + off_out_magnitude);
float* out_dir = reinterpret_cast<float*>(base_out + j0 * stride_out + off_out_direction);
__riscv_vsse32_v_f32m2(out_mx,  stride_out, v_mx,  vl);
__riscv_vsse32_v_f32m2(out_my,  stride_out, v_my,  vl);
__riscv_vsse32_v_f32m2(out_mag, stride_out, v_mag, vl);
__riscv_vsse32_v_f32m2(out_dir, stride_out, v_dir, vl);
```

- 目标内存布局：`pcl::PointXYZIEdge` 的 4 个 float 字段
  - `magnitude`（segment 0）
  - `direction`（segment 1）
  - `magnitude_x`（segment 2）
  - `magnitude_y`（segment 3）
- 数组规模：`n = 640 * 480 = 307200`
- `iterations = 20`，`warmup = 5`
- 固定只对比 store 指令形态：
  - Mode A：4 次 `__riscv_vsse32_v_f32m2` 分别写 4 个字段
  - Mode B：1 次 `__riscv_vssseg4e32_v_f32m2x4` 段存储写 4 个字段

微基准中，Mode A 与 Mode B 的核心差异如下（两者都用同样的 `vlsseg4e32` 生成 `vfloat32m2x4_t`，只比较 store）：

```cpp
// test-rvv/2d/test_edge_store_bench.cpp (simplified)
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

### 板卡实测结果

Mode A（4x `vsse32`）/ Mode B（1x `vssseg4e32`）耗时（ms/iter）：

- Run 1: 6.2849 / 6.8267
- Run 2: 6.70214 / 6.81787
- Run 3: 6.91148 / 6.88651
- Run 4: 6.86585 / 6.80519
- Run 5: 6.50117 / 6.70903

汇总（5 次平均）：

- Mode A 平均：约 `6.6531 ms/iter`
- Mode B 平均：约 `6.8091 ms/iter`

结论：在当前板卡上，`vssseg4e32` **没有带来稳定的收益**（speedup 在约 `0.92x ~ 1.01x` 波动）；因此不建议把 `edge.hpp` 里的 store 主实现从 `vsse32` 直接替换为 `vssseg4e32`。

补充说明：

- **字段/segment 对应关系必须严格一致**：`PointXYZIEdge` 的字段顺序为 `magnitude -> direction -> magnitude_x -> magnitude_y`，段存储时 segment 0..3 也应按该顺序写回；顺序错位会导致字段对调。
- **“指令条数更少”不等价于“写入更快/更稳”**：段存储是否能减少写停顿（Write Stall）高度依赖具体内存系统的写合并/写缓冲实现；从当前板卡数据看，`vssseg4e32` 的收益不稳定且平均不占优。

### 最终策略

写回仍优先采用：`__riscv_vsse32_v_f32m2`

