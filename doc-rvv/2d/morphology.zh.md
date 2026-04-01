# Morphology：RVV 优化实现说明

本文记录 `pcl::Morphology<PointT>` 在 RVV 1.0（`__RVV10__`）下的实现拆分、与上游的差异点，以及板卡侧基准数据。实现对应文件为 `2d/include/pcl/2d/impl/morphology.hpp`。

---

## 1. 背景

Morphology 模块提供二值/灰度形态学算子（腐蚀、膨胀、开运算、闭运算），并提供二值集合运算（并/交/差）。在典型用法中，输入与输出都是 `pcl::PointCloud<PointT>`，只读写 `PointT::intensity` 字段；数据在内存中为 AoS，因此 RVV 路径采用 `vlse32/vsse32` 做步长访存。

形态学的像素级计算有两个特征：

- **访问模板固定但需要跳过 0 tap**：结构元素是一个小核（例如 3×3），核中为 0 的位置不参与 min/max 或 0/1 判定。
- **边界需要与标量语义对齐**：边缘像素会出现越界 tap；与卷积/边缘检测类似，若把坐标裁剪写入向量主循环，会引入额外控制流与整数算术。

本仓库采用中心区向量化、边缘回退 helper 的组织方式，以便把边界复杂度限制在小范围内，且保持与标量 helper 的语义一致。

---

## 2. 与上游实现的差异

相对上游的逐像素标量循环，本仓库增加了 `__RVV10__` 下的分流与 RVV 内核：

- 二值：`erosionBinaryRVV` / `dilationBinaryRVV`，边缘回退 `computePixelErosionBinary` / `computePixelDilationBinary`
- 灰度：`erosionGrayRVV` / `dilationGrayRVV`，边缘回退 `computePixelMin` / `computePixelMax`

入口分流点以 `erosionBinary()` 为例：

```cpp
Morphology<PointT>::erosionBinary(pcl::PointCloud<PointT>& output)
{
  output.width = input_->width;
  output.height = input_->height;
  output.resize(input_->width * input_->height);
#if defined(__RVV10__)
  erosionBinaryRVV(output);
#else
  erosionBinaryStandard(output);
#endif
}
```

集合运算（并/交/差）保留标量实现，对应函数体里也明确写明“保留标量 set operation”。

---

## 3. 详细实现

### 3.1 Center/Edge 分区

RVV 内核在进入主循环前计算中心区边界（`row_lo/row_hi`、`col_lo/col_hi`）。中心区保证结构元素完全落在图像内，内层 tap 访问不需要做越界判断；边缘区域用四段显式循环回退到单像素 helper。

以二值腐蚀为例，边缘回退段如下：

```cpp
  // --- Edge region: single-pixel helper (same semantics as standard) ---
  for (int i = 0; i < row_lo; ++i)
    for (int j = 0; j < iw; ++j)
      computePixelErosionBinary(i, j, output);
  for (int i = row_hi; i < ih; ++i)
    for (int j = 0; j < iw; ++j)
      computePixelErosionBinary(i, j, output);
  for (int i = row_lo; i < row_hi; ++i) {
    for (int j = 0; j < col_lo; ++j)
      computePixelErosionBinary(i, j, output);
    for (int j = col_hi; j < iw; ++j)
      computePixelErosionBinary(i, j, output);
  }
```

灰度腐蚀/膨胀的边缘处理与此同形，只是 helper 分别为 `computePixelMin` / `computePixelMax`（见同文件 470 行附近与 594 行附近）。

### 3.2 二值：腐蚀/膨胀的中心区 RVV

二值腐蚀与膨胀的中心区都以 `vlse32` 加载输入强度，结合核 tap 的 `0/1` 开关做 min/max 归约，最后用比较掩码生成 `0/1` 输出并 `vsse32` 写回。二值腐蚀需要额外检查中心像素为 1（helper 的早退条件），实现里用 `vmfeq` 组合两个掩码完成。

```cpp
  // --- Center (safe) region: min over kernel-1 neighbors, then output = (center==1 &&
  // min==1) ---
  if (row_hi > row_lo && col_hi > col_lo) {
    const float v_one = 1.0f;
    const float v_zero = 0.0f;
    for (int i = row_lo; i < row_hi; ++i) {
      int j0 = col_lo;
      while (j0 < col_hi) {
        std::size_t vl = __riscv_vsetvl_e32m2(static_cast<std::size_t>(col_hi - j0));
        vfloat32m2_t v_min =
            __riscv_vfmv_v_f_f32m2(std::numeric_limits<float>::max(), vl);
        // ... tap loops + vfmin ...
        vfloat32m2_t v_center = __riscv_vlse32_v_f32m2(center_ptr, point_stride, vl);
        vbool16_t m_center_one = __riscv_vmfeq_vf_f32m2_b16(v_center, v_one, vl);
        vbool16_t m_min_one = __riscv_vmfeq_vf_f32m2_b16(v_min, v_one, vl);
        vbool16_t mask = __riscv_vmand_mm_b16(m_center_one, m_min_one, vl);
        vfloat32m2_t v_out = __riscv_vfmerge_vfm_f32m2(v_zero_vec, v_one, mask, vl);
        __riscv_vsse32_v_f32m2(out_ptr, point_stride, v_out, vl);
        j0 += static_cast<int>(vl);
      }
    }
  }
```

对应的 helper 语义在 `computePixelErosionBinary()` 中明确：如果中心像素为 0 直接写 0；否则只要任一有效 tap 不为 1 就写 0，全部满足才写 1。

```cpp
  if ((*input_)(j, i).intensity == 0.0f) {
    output(j, i).intensity = 0.0f;
    return;
  }
  // ... loop taps, early return on != 1 ...
  output(j, i).intensity = 1.0f;
```

### 3.3 灰度：腐蚀/膨胀的中心区 RVV 与 helper 语义

灰度腐蚀（min）与膨胀（max）在中心区分别用 `vfmin` / `vfmax` 归约。helper 中显式维护 `found` 标志：当所有 tap 都越界或 tap 在结构元素中为 0 时，输出为 `-1.0f`。这一点会影响边缘像素与小结构元素时的行为，因此 RVV 路径保留边缘回退到 helper，避免向量路径引入与 `found` 相关的额外控制流。

```cpp
  float min_val = (std::numeric_limits<float>::max)();
  bool found = false;
  // ... found = true when a valid tap exists ...
  output(j, i).intensity = found ? min_val : -1.0f;
```

---

## 4. 测试与基准

### 4.1 基准数据与输出位置

基准程序为 `test-rvv/2d/bench_2d.cpp`，默认输入为合成强度图 `640 x 480`（307200 像素），iterations=20。板卡日志位于 `test-rvv/2d/output/board/`，其中 `compare.log` 为 `analyze_bench_compare.py` 汇总输出。

### 4.2 结果

以 `test-rvv/2d/output/board/compare.log` 的一组数据为例，Morphology 相关条目如下（单位 ms/iter）：

| Item | Std Avg (ms/iter) | RVV Avg (ms/iter) | Speedup |
|---|---:|---:|---:|
| Morphology Erosion (Gray 3x3) | 61.884 | 13.996 | 4.42x |
| Morphology Dilation (Gray 3x3) | 62.017 | 14.144 | 4.38x |
| Morphology Opening (Gray 3x3) | 128.429 | 34.173 | 3.76x |
| Morphology Closing (Gray 3x3) | 127.806 | 34.833 | 3.67x |
| Morphology Erosion (Binary 3x3) | 20.089 | 15.349 | 1.31x |
| Morphology Dilation (Binary 3x3) | 28.568 | 15.532 | 1.84x |
| Morphology Opening (Binary 3x3) | 74.823 | 31.849 | 2.35x |
| Morphology Closing (Binary 3x3) | 77.837 | 32.460 | 2.40x |

灰度 3×3 的 speedup 更高，主要对应中心区 `vfmin/vfmax` 归约的吞吐；二值 3×3 则包含更多“0/1 判定 + 掩码合并 + 写回”，且在 AoS 步长访存下更容易受固定开销影响。

### 4.3 集合运算：保留标量的原因与数据

集合运算（并/交/差）当前保留标量实现；在 `compare.log` 中对应条目接近持平：

| Item | Std Avg (ms/iter) | RVV Avg (ms/iter) | Speedup |
|---|---:|---:|---:|
| Set Operation: Union (A | B) | 9.466 | 9.797 | 0.97x |
| Set Operation: Intersection (A & B) | 9.440 | 9.570 | 0.99x |
| Set Operation: Subtraction (A - B) | 9.495 | 9.641 | 0.98x |

这种行为与实现注释一致：在该场景下，运算是低计算密度的逐元素逻辑，且访问仍是 AoS 的步长模式，手写 RVV 不一定能带来稳定收益。

---

## 5. 总结

Morphology 的 RVV 化沿用 2D 模块的通用拆分：中心区向量化、边缘回退 helper，以保持边界与“无有效 tap（灰度输出 -1.0f）”等语义一致。板卡基准中，灰度 3×3 的腐蚀/膨胀与开闭运算获得 3.7x～4.4x 的加速；二值 3×3 的加速在 1.3x～2.4x。集合运算保留标量，避免在低计算密度与步长访存组合下出现不稳定回退。