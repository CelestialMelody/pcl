# Kernel：RVV 优化实现说明

本文记录 `pcl::kernel<PointT>` 在 RVV 1.0（`__RVV10__`）下生成 Gaussian/LoG 卷积核的实现拆分、与上游的差异点，以及板卡侧基准数据。对照上游文件：[`2d/include/pcl/2d/impl/kernel.hpp`](https://github.com/PointCloudLibrary/pcl/blob/master/2d/include/pcl/2d/impl/kernel.hpp)。

---

## 1. 背景

2D 模块中 `kernel<PointT>` 负责按 `kernel_type_` 生成不同卷积核。Gaussian 与 LoG 的核值由指数函数与简单多项式组合构成；核大小通常较小（3×3、5×5、11×11），单次生成处于微秒量级。

在这个量级上，影响吞吐的往往不是“浮点乘加是否向量化”，而是固定开销能否被压到足够低：例如 `idx / kernel_size`、`idx % kernel_size` 的逐元素除法/取模、临时分配与非连续写回。RVV 路径的目标是把“指数与归一化”集中到连续 `float` 缓冲区里完成，并把 AoS 写回延后到末端一次完成。

---

## 2. 与上游实现的差异

上游标量实现倾向于直接写 `PointCloud`，并在后续循环里完成归一化。本仓库对 Gaussian/LoG 的 RVV 化把计算拆为三段：

- **计算阶段**：填充连续 `buf`（`std::vector<float>`），Gaussian 计算 `exp(arg)`，LoG 计算 `(1-temp)*exp(-temp)`；
- **归约阶段**：在连续 `buf` 上做求和归约（`sumReduceRVV` 使用 `vfredosum`）；
- **归一化与写回**：在连续 `buf` 上做 `vfdiv_vf`，随后以 `&kernel[0].intensity` 为基址，用 `vsse32` 按 `sizeof(PointT)` 步长写回 `PointCloud`。

对应实现入口可在 `kernel<PointT>::gaussianKernel` / `kernel<PointT>::loGKernel` 中看到：RVV 分支调用 `compute*ValuesRVV`，再调用 `sumReduceRVV` 与 `normalizeAndWriteBackRVV`。

```cpp
kernel<PointT>::gaussianKernel(pcl::PointCloud<PointT>& kernel)
{
  const int n = kernel_size_ * kernel_size_;
  kernel.resize(static_cast<std::size_t>(n));
  kernel.height = static_cast<std::uint32_t>(kernel_size_);
  kernel.width = static_cast<std::uint32_t>(kernel_size_);

  const float sigma_sqr = 2.f * sigma_ * sigma_;
  std::vector<float> buf(static_cast<std::size_t>(n));
  const std::size_t un = static_cast<std::size_t>(n);

#if defined(__RVV10__)
  computeGaussianKernelValuesRVV(buf, kernel_size_, sigma_sqr, n);
#else
  computeGaussianKernelValues(buf, kernel_size_, sigma_sqr, n);
#endif

#if defined(__RVV10__)
  float sum = sumReduceRVV(buf, un);
  normalizeAndWriteBackRVV(buf, sum, kernel, un);
#else
  // ... scalar sum, normalize, write-back ...
#endif
}
```

这种结构的直接目的，是让 `pcl::expf_RVV_f32m2`、`vfredosum`、`vfdiv` 都在连续内存上运行；AoS 的步长写回只出现一次，避免在最内层循环里反复触发 stride store。

---

## 3. 详细实现

### 3.1 避免块内逐元素除法/取模

`computeGaussianKernelValuesRVV` / `computeLoGKernelValuesRVV` 都采用“块起点一次性计算 `(ii, jj)`，块内用自增推进”的方式生成二维坐标，避免对每个 lane 执行除法与取模。对应实现如下（LoG 与 Gaussian 形态一致）。

```cpp
computeGaussianKernelValuesRVV(std::vector<float>& buf, int kernel_size, float sigma_sqr, int n)
{
  // ...
  while (j0 < un) {
    std::size_t vl = __riscv_vsetvl_e32m2(un - j0);
    // ...
    int ii = static_cast<int>(j0) / kernel_size;
    int jj = static_cast<int>(j0) - ii * kernel_size;

    for (std::size_t i = 0; i < vl; ++i) {
      const int iks = ii - half_size;
      const int jks = jj - half_size;
      tmp[i] = static_cast<float>(iks * iks + jks * jks) * inv_sigma_sqr;

      ++jj;
      if (jj == kernel_size) {
        jj = 0;
        ++ii;
      }
    }

    vfloat32m2_t v_arg = __riscv_vle32_v_f32m2(tmp, vl);
    vfloat32m2_t v_exp = pcl::expf_RVV_f32m2(v_arg, vl);
    __riscv_vse32_v_f32m2(buf.data() + j0, v_exp, vl);
    j0 += vl;
  }
}
```

### 3.2 栈缓冲与 `MAX_SAFE_VL`

RVV 路径需要先构造每个 lane 的参数（Gaussian 的 `arg`，LoG 的 `-temp` 与 `1-temp`），再用 `vle32` 装入向量寄存器。这里选择栈上定长数组（`MAX_SAFE_VL=128`），并在 `vl > MAX_SAFE_VL` 时限制 `vl`，避免引入堆分配或 TLS 状态。

这一选择对应的工程约束是：核生成本身是短函数，固定开销更敏感；同时它位于 2D 算子链路前端，避免引入隐式全局状态有助于后续排障与复现。

### 3.3 求和归约、归一化与 AoS 写回

求和与归一化都在连续 `buf` 上进行。`sumReduceRVV` 使用 `vfredosum`，`normalizeAndWriteBackRVV` 先在 `buf` 上做 `vfdiv_vf`，再用 `vsse32` 将结果写到 `kernel[].intensity`。

```cpp
sumReduceRVV(const std::vector<float>& buf, std::size_t n)
{
  // ... vfadd accumulation ...
  vfloat32m1_t v_sum =
      __riscv_vfredosum_vs_f32m2_f32m1(v_acc, v_zero, max_vl);
  return __riscv_vfmv_f_s_f32m1_f32(v_sum);
}

normalizeAndWriteBackRVV(std::vector<float>& buf, float sum, pcl::PointCloud<PointT>& kernel, std::size_t n)
{
  // normalize buf (contiguous)
  // ... vfdiv_vf + vse32 ...
  // strided store to kernel[].intensity
  const std::size_t point_stride = sizeof(PointT);
  float* intensity_base = &kernel[0].intensity;
  // ... vsse32 ...
}
```

---

## 4. 测试与基准

### 4.1 基准测试

基准程序为 `test-rvv/2d/bench_2d.cpp`，默认图像尺寸 `640 x 480`，iterations=20。板卡日志位于 `test-rvv/2d/output/board/`，其中 `compare.log` 为 `analyze_bench_compare.py` 汇总输出，包含每个条目的 Avg/Total/Speedup。

### 4.2 结果

以 `test-rvv/2d/output/board/compare.log` 的一组数据为例，Kernel 相关条目如下（单位为 ms/iter，speedup 取 `Std/RVV`）：

| Item | Std Avg (ms/iter) | RVV Avg (ms/iter) | Speedup |
|---|---:|---:|---:|
| Kernel gaussianKernel (3x3, sigma=1.0) | 0.0013 | 0.0009 | 1.41x |
| Kernel gaussianKernel (5x5, sigma=1.0) | 0.0026 | 0.0011 | 2.29x |
| Kernel gaussianKernel (11x11, sigma=2.0) | 0.0095 | 0.0029 | 3.28x |
| Kernel loGKernel (3x3, sigma=1.0) | 0.0012 | 0.0008 | 1.59x |
| Kernel loGKernel (5x5, sigma=1.0) | 0.0034 | 0.0011 | 3.12x |
| Kernel loGKernel (11x11, sigma=2.0) | 0.0100 | 0.0029 | 3.46x |

这些结果与实现结构一致：核越大，指数函数与向量归一化占比越高；核越小，`vsetvl`、地址生成与栈缓冲初始化等固定开销占比更明显。

---

## 5. 总结

Kernel 的 RVV 化把连续数学计算与 AoS 写回解耦：指数/归约/归一化都在 `buf` 上完成，`PointCloud` 的步长写回只在末端出现一次。实现上引入 `MAX_SAFE_VL` 的栈缓冲以压缩固定开销，避免堆分配与隐式状态。板卡基准显示 Gaussian/LoG 在 3×3 到 11×11 的范围内保持正向加速，且核尺寸越大收益越明显。
