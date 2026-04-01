# Edge：RVV 优化实现说明

本文记录 PCL 2D 模块 `pcl::Edge<PointInT, PointOutT>` 在 RVV 1.0（`__RVV10__`）下的实现拆分、与上游的差异点，以及板卡侧测试与基准数据。对照上游文件：[`2d/include/pcl/2d/impl/edge.hpp`](https://github.com/PointCloudLibrary/pcl/blob/master/2d/include/pcl/2d/impl/edge.hpp)。

---

## 1. 背景

Edge 模块的调用路径由两部分组成：

- **梯度场生成**：以 Sobel/Prewitt/Roberts 等核对输入强度图做两次卷积，得到 `magnitude_x` 与 `magnitude_y`（在本实现里对应 `PointXYZI::intensity`）。
- **派生量计算**：对每个像素计算幅值 `magnitude = sqrt(mx^2 + my^2)` 与方向 `direction = atan2(my, mx)`，并写回 `PointOutT`（常见为 `PointXYZIEdge`）。Canny 流程额外需要把方向量化为 `0/45/90/135`（度）供 NMS（非极大值抑制）分支判断使用。

对向量化而言，约束来自数据布局与函数形态：

- 输出点云为 AoS，字段 `magnitude_x/magnitude_y/magnitude/direction` 不是连续数组；按字段写回需要 `offsetof` + 步长存储（`vsse32`）。
- 幅值/方向计算包含 `sqrt` 与 `atan2`；若仍按标量逐点计算，卷积之后的派生量阶段会成为可见的时间占比。
- Canny 的 NMS/追踪/阈值循环是典型控制流密集代码，向量化收益不稳定。本仓库只对“幅值/方向计算”和“方向量化”做 RVV 化，保持其余阶段与上游一致。

---

## 2. 与上游实现的差异

本仓库把上游内联在成员函数中的逐点计算拆成独立静态辅助函数，并增加 `__RVV10__` 下的实现分支：

- `computeMagnitudeDirectionRVV` / `computeMagnitudeDirectionStd`：对 `mx/my` 计算 `magnitude` 与 `direction` 并写入 `PointOutT`；
- `discretizeAnglesRVV` / `discretizeAnglesStd`：把 `direction` 从弧度转换到度并分箱为 `0/45/90/135`。

入口函数在 `#if defined(__RVV10__)` 下选择 RVV 版本，否则回退到 Std 版本。以 `detectEdgeSobel()` 为例，分流点如下：

```cpp
  const int height = input_->height;
  const int width = input_->width;
  const std::size_t n = static_cast<std::size_t>(height) * static_cast<std::size_t>(width);

  output.resize(n);
  output.height = height;
  output.width = width;

#if defined(__RVV10__)
  computeMagnitudeDirectionRVV(*magnitude_x, *magnitude_y, output, n);
#else
  computeMagnitudeDirectionStd(*magnitude_x, *magnitude_y, output, n);
#endif
```

`discretizeAngles()` 的分流点如下：

```cpp
Edge<PointInT, PointOutT>::discretizeAngles(pcl::PointCloud<PointOutT>& thet)
{
  const int height = thet.height;
  const int width = thet.width;

#if defined(__RVV10__)
  discretizeAnglesRVV(thet, height, width);
#else
  discretizeAnglesStd(thet, height, width);
#endif
}
```

---

## 3. 详细实现

### 3.1 幅值与方向：`computeMagnitudeDirectionRVV`

RVV 实现以 `PointXYZI` 输入的 `intensity` 作为 `mx/my`，并对输出 `PointOutT` 的四个字段分别做步长写回：

- `vlse32` 以 `sizeof(PointXYZI)` 为步长加载 `mx/my`；
- `magnitude` 使用 `vfsqrt` 计算 $\sqrt{mx^2 + my^2}$；
- `direction` 使用 `pcl::atan2_RVV_f32m2`；
- 四个字段分别 `vsse32` 写回（步长 `sizeof(PointOutT)`）。

```cpp
computeMagnitudeDirectionRVV(const pcl::PointCloud<pcl::PointXYZI>& magnitude_x,
                             const pcl::PointCloud<pcl::PointXYZI>& magnitude_y,
                             pcl::PointCloud<PointOutT>& output,
                             std::size_t n)
{
  const std::size_t stride_in = sizeof(pcl::PointXYZI);
  const std::size_t stride_out = sizeof(PointOutT);
  const std::size_t off_mx = offsetof(pcl::PointXYZI, intensity);
  const std::size_t off_out_magnitude_x = offsetof(PointOutT, magnitude_x);
  const std::size_t off_out_magnitude_y = offsetof(PointOutT, magnitude_y);
  const std::size_t off_out_magnitude = offsetof(PointOutT, magnitude);
  const std::size_t off_out_direction = offsetof(PointOutT, direction);

  const std::uint8_t* base_mx =
      reinterpret_cast<const std::uint8_t*>(magnitude_x.points.data()) + off_mx;
  const std::uint8_t* base_my =
      reinterpret_cast<const std::uint8_t*>(magnitude_y.points.data()) + off_mx;
  std::uint8_t* base_out = reinterpret_cast<std::uint8_t*>(output.points.data());

  std::size_t j0 = 0;
  while (j0 < n) {
    std::size_t vl = __riscv_vsetvl_e32m2(n - j0);

    const float* ptr_mx = reinterpret_cast<const float*>(base_mx + j0 * stride_in);
    const float* ptr_my = reinterpret_cast<const float*>(base_my + j0 * stride_in);
    vfloat32m2_t v_mx = __riscv_vlse32_v_f32m2(ptr_mx, stride_in, vl);
    vfloat32m2_t v_my = __riscv_vlse32_v_f32m2(ptr_my, stride_in, vl);

    vfloat32m2_t v_mag =
        __riscv_vfsqrt_v_f32m2(__riscv_vfadd_vv_f32m2(
            __riscv_vfmul_vv_f32m2(v_mx, v_mx, vl),
            __riscv_vfmul_vv_f32m2(v_my, v_my, vl), vl), vl);
    vfloat32m2_t v_dir = pcl::atan2_RVV_f32m2(v_my, v_mx, vl);

    float* out_mx = reinterpret_cast<float*>(base_out + j0 * stride_out + off_out_magnitude_x);
    float* out_my = reinterpret_cast<float*>(base_out + j0 * stride_out + off_out_magnitude_y);
    float* out_mag = reinterpret_cast<float*>(base_out + j0 * stride_out + off_out_magnitude);
    float* out_dir = reinterpret_cast<float*>(base_out + j0 * stride_out + off_out_direction);
    __riscv_vsse32_v_f32m2(out_mx, stride_out, v_mx, vl);
    __riscv_vsse32_v_f32m2(out_my, stride_out, v_my, vl);
    __riscv_vsse32_v_f32m2(out_mag, stride_out, v_mag, vl);
    __riscv_vsse32_v_f32m2(out_dir, stride_out, v_dir, vl);

    j0 += vl;
  }
}
```

`computeMagnitudeDirectionStd` 保留逐点 `std::sqrt`、`std::atan2`，用于不启用 RVV 时的语义对齐与回归对比（同文件 105 行附近）。

### 3.2 方向量化：`discretizeAnglesRVV`

上游的分箱条件以 `rad2deg` 后的度数区间判断为准；本仓库的 RVV 版本做了一个额外的折叠：把负角度加 180 折叠到 `[0, 180)`，使每个方向分箱对应单个区间，从而把“正负两套区间”的逻辑变成三个互斥掩码（45/90/135）与默认 0° 的 `vmerge` 链。

```cpp
discretizeAnglesRVV(pcl::PointCloud<PointOutT>& thet, int height, int width)
{
  const int n = height * width;
  const std::size_t stride = sizeof(PointOutT);
  const std::size_t off_dir = offsetof(PointOutT, direction);
  const float rad2deg = 180.0f / 3.14159265358979323846f;
  std::uint8_t* base =
      reinterpret_cast<std::uint8_t*>(thet.points.data()) + off_dir;

  std::size_t j0 = 0;
  while (j0 < static_cast<std::size_t>(n)) {
    std::size_t vl = __riscv_vsetvl_e32m2(static_cast<std::size_t>(n) - j0);
    float* ptr = reinterpret_cast<float*>(base + j0 * stride);
    vfloat32m2_t v_rad = __riscv_vlse32_v_f32m2(ptr, stride, vl);
    vfloat32m2_t v_deg =
        __riscv_vfmul_vf_f32m2(v_rad, rad2deg, vl);

    // Fold negative degrees into [0, 180]; each bin is one interval (see discretizeAnglesStd).
    vbool16_t m_neg = __riscv_vmflt_vf_f32m2_b16(v_deg, 0.0f, vl);
    vfloat32m2_t v_deg_fold =
        __riscv_vfadd_vf_f32m2(v_deg, 180.0f, vl);
    v_deg = __riscv_vmerge_vvm_f32m2(v_deg, v_deg_fold, m_neg, vl);

    const vfloat32m2_t v_0 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    const vfloat32m2_t v_45 = __riscv_vfmv_v_f_f32m2(45.0f, vl);
    const vfloat32m2_t v_90 = __riscv_vfmv_v_f_f32m2(90.0f, vl);
    const vfloat32m2_t v_135 = __riscv_vfmv_v_f_f32m2(135.0f, vl);

    vbool16_t m45 = __riscv_vmfgt_vf_f32m2_b16(v_deg, 22.5f, vl);
    m45 = __riscv_vmand_mm_b16(m45, __riscv_vmflt_vf_f32m2_b16(v_deg, 67.5f, vl), vl);

    vbool16_t m90 = __riscv_vmfge_vf_f32m2_b16(v_deg, 67.5f, vl);
    m90 = __riscv_vmand_mm_b16(m90, __riscv_vmfle_vf_f32m2_b16(v_deg, 112.5f, vl), vl);

    vbool16_t m135 = __riscv_vmfgt_vf_f32m2_b16(v_deg, 112.5f, vl);
    m135 = __riscv_vmand_mm_b16(m135, __riscv_vmflt_vf_f32m2_b16(v_deg, 157.5f, vl), vl);

    // vmerge(op1, op2, mask) => mask ? op2 : op1; remainder is 0°
    vfloat32m2_t result = __riscv_vmerge_vvm_f32m2(v_0, v_45, m45, vl);
    result = __riscv_vmerge_vvm_f32m2(result, v_90, m90, vl);
    result = __riscv_vmerge_vvm_f32m2(result, v_135, m135, vl);
    __riscv_vsse32_v_f32m2(ptr, stride, result, vl);
    j0 += vl;
  }
}
```

对照的标量分箱逻辑在 `discretizeAnglesStd` 中完整保留（`rad2deg` + 四段 `if/else if`），用于在条件边界上做行为对齐与回归对比。

---

## 4. 测试与基准

### 4.1 正确性测试

2D 模块的回归测试入口为 `test-rvv/2d/test_2d.cpp`。板卡侧的整套输出样例见 `test-rvv/2d/output/run_test.log`，Edge 相关用例包括 `Edge.sobel`、`Edge.prewitt`、`Edge.canny`、`Edge.DiscretizeAngles`。

### 4.2 基准测试与结果

基准程序为 `test-rvv/2d/bench_2d.cpp`，默认输入为合成强度图 `640 x 480`（307200 像素），iterations=20。板卡（Milk-V Jupyter）上的日志放在 `test-rvv/2d/output/board/`，其中 `compare.log` 为 `analyze_bench_compare.py` 对两份原始日志汇总后的表格输出。

以 `test-rvv/2d/output/board/compare.log` 的一组数据为例（设备 `Milkv-Jupiter`），Edge 三项对比如下：

| Item | Std Avg (ms/iter) | RVV Avg (ms/iter) | Speedup |
|---|---:|---:|---:|
| Edge Sobel (Magnitude) | 149.965 | 77.471 | 1.94x |
| Edge Canny (Full Pipeline) | 234.432 | 125.042 | 1.87x |
| Edge discretizeAngles | 12.395 | 7.320 | 1.69x |

结果差异与实现拆分一致：Sobel/Prewitt 在卷积之后要做逐点 `sqrt/atan2` 与字段写回，这一段在 RVV 下收益较稳定；Canny 的流水线仍包含 NMS、追踪等控制流密集步骤，整体 speedup 受这部分比例影响更明显。

---

## 5. 总结

Edge 模块在 RVV 下的改动集中在两处：对 `mx/my -> magnitude/direction` 的逐点派生量计算向量化，以及对 `direction` 的分箱量化向量化。Canny 的其余阶段保持与上游一致的标量控制流。结果上，Sobel 与方向量化的 speedup 可以直接在基准表中观察；Canny 的 speedup 则更多反映“向量化阶段在全流程中所占比例”。

