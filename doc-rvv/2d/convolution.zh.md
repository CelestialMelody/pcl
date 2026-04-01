# Convolution：RVV 优化实现说明

本文记录 PCL 2D 模块 `pcl::Convolution<PointT>` 在 RVV 1.0（`__RVV10__`）下的实现拆分、与上游的差异点，以及板卡侧的测试与基准数据。对照上游文件：[`2d/include/pcl/2d/impl/convolution.hpp`](https://github.com/PointCloudLibrary/pcl/blob/master/2d/include/pcl/2d/impl/convolution.hpp)。

---

## 1. 背景

`Convolution` 以 `PointCloud<PointT>` 表示二维强度图：每个像素对应一个点，`PointT::intensity` 存储标量强度；卷积核同样以点云表示，使用 `kernel_(l, k).intensity` 作为权值。输出采用“拷贝输入后只更新 `intensity`”的策略：先执行 `output = *input_`，再覆写输出的强度字段，从而保持 `PointT` 的其他字段与输入一致。

这类数据布局对向量化提出了两点直接约束：

- **AoS 步长访问**：强度字段在结构体内有固定偏移，但相邻像素的 `intensity` 之间隔着 `sizeof(PointT)` 字节，因此中心区向量化需要用 RVV 的 strided load/store（`vlse32`/`vsse32`）而不是连续 `vle32`。
- **边界语义不可消掉**：上游支持 `CLAMP`、`MIRROR`、`ZERO_PADDING` 三种边界选项；如果把边界映射写进向量循环，会引入掩码、整数映射与分支，且不同选项的控制流差异较大。

---

## 2. 与上游实现的差异

上游实现基本等价于“对每个输出像素调用一次边界映射 + \(kw \times kh\) 次乘加”。本仓库增加 RVV 版本后，`filter()` 的分流结构如下：在 `__RVV10__` 下进入 `filterRVV()`，否则回退 `filterStandard()`；两条路径都复用相同的 `computePixelIntensity()`，保证边界语义一致。

```cpp
Convolution<PointT>::filter(pcl::PointCloud<PointT>& output)
{
  output = *input_;
#if defined(__RVV10__)
  filterRVV(output);
#else
  filterStandard(output);
#endif
}
```

额外的结构变化来自 `filterRVV()` 对像素集合的拆分：把“卷积核完整落在图像内”的中心区域向量化，边缘环带仍走 `computePixelIntensity()`。这样做的直接结果是：向量循环里不需要执行 `CLAMP/MIRROR/ZERO_PADDING` 的坐标映射；需要映射的像素数量被限制在四个环带内。

---

## 3. 详细实现

### 3.1 标量路径：`computePixelIntensity(i, j, output)`

`computePixelIntensity()` 的职责是：对单个输出像素 `(i, j)` 计算卷积和，并写回 `output(j, i).intensity`。它以 `switch(boundary_options_)` 分出三套坐标处理逻辑：`CLAMP` 做截断，`MIRROR` 做镜像，`ZERO_PADDING` 直接跳过越界采样。

这部分与上游基本保持一致。原因很直接：边界行为要对齐，且边缘环带会复用该函数；如果在这里把二维核展平为一维索引，再在循环里做 `div/rem`，对边缘像素的额外成本会反映到整张图的总耗时。

### 3.2 RVV 路径：中心区向量化

`filterRVV()` 先计算中心区范围：`row_lo = kh_half`、`row_hi = ih - kh_half`，`col_lo = kw_half`、`col_hi = iw - kw_half`。在该矩形内，任意核元素 `(k, l)` 对应的输入坐标都在合法范围内；因此向量循环不需要执行坐标裁剪或镜像。

向量化方向选择为“列方向”：固定输出行 `i`，在 `j0` 起点上按 `vl` 扫描连续的列坐标。输入强度的加载用 `vlse32`（步长 `sizeof(PointT)`），核系数为标量 `kernel_val`，通过 `vfmacc_vf` 做向量乘加累加到 `v_acc`，最后用 `vsse32` 写回输出强度。

```cpp
Convolution<PointT>::filterRVV(pcl::PointCloud<PointT>& output)
{
  const int iw = static_cast<int>(input_->width);
  const int ih = static_cast<int>(input_->height);
  const int kw = static_cast<int>(kernel_.width);
  const int kh = static_cast<int>(kernel_.height);
  const int kh_half = kh / 2;
  const int kw_half = kw / 2;

  const std::size_t point_stride = sizeof(PointT);
  const std::size_t intensity_offset = offsetof(PointT, intensity);
  const std::uint8_t* input_base =
      reinterpret_cast<const std::uint8_t*>(input_->points.data());
  std::uint8_t* output_base = reinterpret_cast<std::uint8_t*>(output.points.data());

  // --- Center (safe) region: kernel fully inside image, vectorized ---
  const int row_lo = kh_half;
  const int row_hi = ih - kh_half;
  const int col_lo = kw_half;
  const int col_hi = iw - kw_half;

  if (row_hi > row_lo && col_hi > col_lo) {
    for (int i = row_lo; i < row_hi; ++i) {
      int j0 = col_lo;
      while (j0 < col_hi) {
        std::size_t vl = __riscv_vsetvl_e32m2(static_cast<std::size_t>(col_hi - j0));
        vfloat32m2_t v_acc = __riscv_vfmv_v_f_f32m2(0.0f, vl);

        for (int k = 0; k < kh; ++k) {
          const int base_row = i + k - kh_half;
          const std::size_t row_offset = static_cast<std::size_t>(base_row) * iw;
          const std::size_t in_start_col = static_cast<std::size_t>(j0 - kw_half);
          const std::uint8_t* in_ptr_byte_base =
              input_base + (row_offset + in_start_col) * point_stride + intensity_offset;
          for (int l = 0; l < kw; ++l) {
            const float kernel_val = kernel_(l, k).intensity;
            const float* in_ptr = reinterpret_cast<const float*>(in_ptr_byte_base);
            vfloat32m2_t v_in = __riscv_vlse32_v_f32m2(in_ptr, point_stride, vl);
            v_acc = __riscv_vfmacc_vf_f32m2(v_acc, kernel_val, v_in, vl);
            in_ptr_byte_base += point_stride; // next l => next input column (+1)
          }
        }

        float* out_ptr = reinterpret_cast<float*>(
            output_base + (static_cast<std::size_t>(i) * iw + j0) * point_stride + intensity_offset);
        __riscv_vsse32_v_f32m2(out_ptr, point_stride, v_acc, vl);
        j0 += static_cast<int>(vl);
      }
    }
  }
```

这里采用 `vlse32`/`vsse32` 是由 AoS 数据布局决定的。若后续把输入强度改为 SoA（单独的 `float` 平面），中心区可以改用连续 `vle32`/`vse32`，对应的地址生成与访存形态会更简单；但这会改变 2D 模块以 `PointCloud<PointT>` 传递图像的接口与调用点，当前实现不做该类结构调整。

### 3.3 边缘环带回退

`filterRVV()` 对中心区之外的像素显式分成四段循环（上/下/左/右），统一回退到 `computePixelIntensity()`。这避免了在中心区向量循环里混入边界判断；代价是边缘像素仍保留标量开销。

```cpp
  // --- Edge region: 4 explicit loops, no branch on center pixels ---
  for (int i = 0; i < row_lo; ++i)
    for (int j = 0; j < iw; ++j)
      computePixelIntensity(i, j, output);
  for (int i = row_hi; i < ih; ++i)
    for (int j = 0; j < iw; ++j)
      computePixelIntensity(i, j, output);
  for (int i = row_lo; i < row_hi; ++i) {
    for (int j = 0; j < col_lo; ++j)
      computePixelIntensity(i, j, output);
    for (int j = col_hi; j < iw; ++j)
      computePixelIntensity(i, j, output);
  }
}
```

---

## 4. 测试与基准

### 4.1 正确性测试

2D 模块的回归测试入口为 `test-rvv/2d/test_2d.cpp`；其中 Convolution 覆盖边界选项与高斯平滑用例。板卡侧的整套输出样例见 `test-rvv/2d/output/run_test.log`，其中 Convolution 两个用例为 `Convolution.borderOptions`、`Convolution.gaussianSmooth`。

### 4.2 基准测试

基准程序为 `test-rvv/2d/bench_2d.cpp`，默认输入为合成强度图 `640 x 480`（307200 像素），卷积核覆盖 `5x5 (sigma=1.0)` 与 `11x11 (sigma=2.0)` 两个规模；迭代次数为 20（由程序固定）。对应的实现片段如下。

```cpp
int main(int argc, char** argv) {
  int width = 640;
  int height = 480;
  int iterations = 20;
  // ...
  generateRandomImage(input, width, height);
  // ...
  {
    pcl::Convolution<PointT> conv;
    conv.setInputCloud(input);
    // 生成一个 5x5 高斯核
    pcl::kernel<PointT> k;
    k.setKernelType(pcl::kernel<PointT>::GAUSSIAN);
    k.setKernelSize(5);
    k.setKernelSigma(1.0f);
    k.fetchKernel(*kernel_cloud);
    conv.setKernel(*kernel_cloud);

    Benchmarker("Convolution (5x5 Gaussian)").run([&](){
      conv.filter(*output);
    }, iterations);
  }
  // ...
}
```

板卡上（Milk-V Jupyter）的日志文件放在 `test-rvv/2d/output/board/`：

- `run_bench_std.log` / `run_bench_rvv.log`：分别为标量与 RVV 版本的原始基准输出；
- `compare.log`：`analyze_bench_compare.py` 解析两份日志后生成的对照表，包含每项 Avg/Total/Speedup 与测试上下文。

### 4.3 结果

以 `test-rvv/2d/output/board/compare.log` 的一组数据为例（`640 x 480`，iterations=20），卷积两项的对比如下。

| Item | Std (ms/iter) | RVV (ms/iter) | Speedup |
|---|---:|---:|---:|
| Convolution (5x5 Gaussian) | 80.508 | 24.707 | 3.26x |
| Convolution (11x11 Gaussian) | 1011.153 | 166.044 | 6.09x |

两项 speedup 的差异符合实现结构：核越大，中心区乘加占比越高，边缘环带回退的影响越小；反之 `5x5` 时，环带比例与 `computePixelIntensity()` 的分支成本更显著。

---

## 5. 总结	

`Convolution` 的 RVV 版本采用“中心区向量化 + 边缘环带回退”的拆分方式：中心区用 `vlse32`/`vfmacc_vf`/`vsse32` 覆盖 AoS 步长访存与乘加累加，边界语义完全复用 `computePixelIntensity()` 的标量实现。该结构在不改变接口与边界行为的前提下，把向量循环内的控制流压到最小，并把与边界选项相关的分支集中到边缘环带。
