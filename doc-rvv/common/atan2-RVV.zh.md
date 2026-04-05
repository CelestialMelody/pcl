# atan2_RVV_f32m2 实现说明

本文描述 `pcl::atan2_RVV_f32m2` 的算法与 RVV 实现细节，并给出板卡侧测试数据。该函数面向批量 `(y, x)` 输入，返回弧度制角度，范围 \((-\pi, \pi]\)。

---

## 1. 接口与语义

接口声明位于 `common/include/pcl/common/common.h`，实现位于 `common/include/pcl/common/impl/common.hpp`，仅在 `__RVV10__` 下可用。`vl` 由调用方通过 `vsetvl_e32m2` 设定，支持 strip-mining 与尾部处理。

```cpp
  inline vfloat32m2_t
  atan2_RVV_f32m2(const vfloat32m2_t& y, const vfloat32m2_t& x, const std::size_t vl);
```

---

## 2. 数学原理与实现

### 2.1 范围约化

目标是把 \(\arctan\) 的输入限制到 \([-1,1]\)，这样多项式逼近的误差更可控。实现里先计算 `abs_x/abs_y`，再生成 `swap_mask = (|x| < |y|)`，并用 `vmerge` 选择分子分母：

- 未交换：`num = y`、`den = x`，得到 \(t = y/x\)
- 交换：`num = x`、`den = y`，得到 \(t = x/y\)

交换后必有 \(|t|\le 1\)。这一步不改变最终角度，只是把计算挪到更适合多项式的区间。

分母的数值保护紧跟在约化之后：实现用 `tiny=1e-20f` 给 `|den|` 设置下限，并用 `vfsgnj` 把 `den` 的符号注入回去，得到 `den_safe`。随后计算 `atan_input = num/den_safe`，并对 `atan_input` 再做一次 \([-1,1]\) 夹取。夹取用于处理“`den` 被 `tiny` 替换后比值可能越界”与浮点舍入带来的轻微超界，避免把区间外值送进多项式。

对应实现片段如下：

```cpp
  const vfloat32m2_t abs_x = __riscv_vfsgnjx_vv_f32m2 (x, x, vl);
  const vfloat32m2_t abs_y = __riscv_vfsgnjx_vv_f32m2 (y, y, vl);
  // swap when |y| > |x|; vmerge(op1, op2, mask) => mask ? op2 : op1 (match atan2.cpp)
  const vbool16_t swap_mask = __riscv_vmflt_vv_f32m2_b16 (abs_x, abs_y, vl);
  const vfloat32m2_t num = __riscv_vmerge_vvm_f32m2 (y, x, swap_mask, vl);
  const vfloat32m2_t den = __riscv_vmerge_vvm_f32m2 (x, y, swap_mask, vl);
  // Preserve sign of den when clamping (scalar: den = (den>=0)? tiny : -tiny)
  const vfloat32m2_t abs_den = __riscv_vfsgnjx_vv_f32m2 (den, den, vl);
  const vfloat32m2_t den_safe = __riscv_vfsgnj_vv_f32m2 (
      __riscv_vfmax_vf_f32m2 (abs_den, tiny, vl), den, vl);
  vfloat32m2_t atan_input = __riscv_vfdiv_vv_f32m2 (num, den_safe, vl);
  // Clamp to [-1,1] so polynomial stays valid (avoids overflow from float noise)
  atan_input = __riscv_vfmin_vf_f32m2 (__riscv_vfmax_vf_f32m2 (atan_input, -1.0f, vl), 1.0f, vl);
```

### 2.2 多项式逼近

在 \(t\in[-1,1]\) 上，用 Hastings 风格的 11 阶奇次多项式近似 \(\arctan(t)\)。系数与 `atan2_test.cpp`、`common.hpp` 实现一致：

- \(a_1=0.99997726\), \(a_3=-0.33262347\), \(a_5=0.19354346\)
- \(a_7=-0.11643287\), \(a_9=0.05265332\), \(a_{11}=-0.01172120\)

实现采用 Horner 形式减少乘法次数。令 \(x2=t^2\)，则：

\[
p = a_1 + x2\cdot(a_3 + x2\cdot(a_5 + x2\cdot(a_7 + x2\cdot(a_9 + x2\cdot a_{11}))))\\
\arctan(t) \approx t\cdot p
\]

对应到代码就是：`x2 = atan_input * atan_input`，`p` 从 `a11` 起按 `vfmacc` 逐级叠加，最后 `result = atan_input * p`。

对应实现片段如下：

```cpp
  const vfloat32m2_t x2 = __riscv_vfmul_vv_f32m2 (atan_input, atan_input, vl);
  vfloat32m2_t p = __riscv_vfmv_v_f_f32m2 (a11, vl);
  p = __riscv_vfmacc_vv_f32m2 (__riscv_vfmv_v_f_f32m2 (a9, vl), x2, p, vl);
  p = __riscv_vfmacc_vv_f32m2 (__riscv_vfmv_v_f_f32m2 (a7, vl), x2, p, vl);
  p = __riscv_vfmacc_vv_f32m2 (__riscv_vfmv_v_f_f32m2 (a5, vl), x2, p, vl);
  p = __riscv_vfmacc_vv_f32m2 (__riscv_vfmv_v_f_f32m2 (a3, vl), x2, p, vl);
  p = __riscv_vfmacc_vv_f32m2 (__riscv_vfmv_v_f_f32m2 (a1, vl), x2, p, vl);
  vfloat32m2_t result = __riscv_vfmul_vv_f32m2 (atan_input, p, vl);
```

### 2.3 重构：交换补偿与象限补偿

约化阶段可能发生交换，因此需要把 “计算的是 \(x/y\) 还是 \(y/x\)” 补回来。

恒等式为：若 \(|y|>|x|\)，则 \(\arctan(y/x)=\mathrm{sgn}(t)\cdot\frac{\pi}{2}-\arctan(t)\)，其中 \(t=x/y\) 。

实现里用 `atan_input>=0` 选择 `adj=+pi/2` 或 `-pi/2`，计算 `adj - result`，再通过 `swap_mask` 合并到最终 `result`。

最后一步恢复象限：当 `x<0` 时，结果位于第二或第三象限，需要在当前结果上加 \(+\pi\)（`y>=0`）或 \(-\pi\)（`y<0`）。实现里通过 `x_lt_zero` 与 `y_ge_zero` 生成 `add_val`，并用 `vmerge` 有条件地执行 `result + add_val`。

对应实现片段如下：

```cpp
  // When swapped: adj = +pi/2 when atan_input>=0 (same sign x,y), else -pi/2
  const vbool16_t atan_ge_zero = __riscv_vmfge_vf_f32m2_b16 (atan_input, 0.0f, vl);
  const vfloat32m2_t pi_2_vec = __riscv_vfmv_v_f_f32m2 (pi_2, vl);
  const vfloat32m2_t neg_pi_2 = __riscv_vfmv_v_f_f32m2 (-pi_2, vl);
  const vfloat32m2_t adj = __riscv_vmerge_vvm_f32m2 (neg_pi_2, pi_2_vec, atan_ge_zero, vl);
  result = __riscv_vmerge_vvm_f32m2 (result, __riscv_vfsub_vv_f32m2 (adj, result, vl), swap_mask, vl);

  const vbool16_t x_lt_zero = __riscv_vmflt_vf_f32m2_b16 (x, 0.0f, vl);
  const vbool16_t y_ge_zero = __riscv_vmfge_vf_f32m2_b16 (y, 0.0f, vl);
  const vfloat32m2_t pi_vec = __riscv_vfmv_v_f_f32m2 (pi, vl);
  const vfloat32m2_t neg_pi = __riscv_vfmv_v_f_f32m2 (-pi, vl);
  const vfloat32m2_t add_val = __riscv_vmerge_vvm_f32m2 (neg_pi, pi_vec, y_ge_zero, vl);
  result = __riscv_vmerge_vvm_f32m2 (result, __riscv_vfadd_vv_f32m2 (result, add_val, vl), x_lt_zero, vl);
```

---

## 3. 测试与结果

测试程序为 `test-rvv/common/common/atan2_test.cpp`。它在 `[-1,1]×[-1,1]` 的 256×256 网格上对比 `std::atan2` 的误差，并对 `std::atan2`、标量近似、RVV 版本分别计时。

```cpp
int main()
{
  const std::size_t n = 256 * 256;
  // ... fill grid and reference ...
  std::printf("=== atan2 approximation vs std::atan2 (n = %zu) ===\n", n);
  // ... compute and print errors ...
  std::printf("\n=== Performance Benchmark (n = %zu, iterations = 100) ===\n", n);
  const int iterations = 100;
  // ... benchmark std, scalar approx, RVV ...
}
```

板卡侧日志位于 `test-rvv/common/common/output/board/atan2_test.log`。其中一组实测数据如下（`n=65536`，`iterations=100`）：

```text
=== atan2 approximation vs std::atan2 (n = 65536) ===
  Scalar approximation (same polynomial as RVV):
    max absolute error:  0.000002 rad  (0.0001 deg)
    mean absolute error: 0.000001 rad
  RVV (pcl::atan2_RVV_f32m2):
    max absolute error:  0.000002 rad  (0.0001 deg)
    mean absolute error: 0.000001 rad
  RVV vs scalar approx max diff: 0.000000e+00 (expect ~0)

=== Performance Benchmark (n = 65536, iterations = 100) ===
  std::atan2:
    time:       808.429 ms (8.084 ms/iter)
    throughput: 8.11 M elements/sec
  Scalar approximation:
    time:       72.373 ms (0.724 ms/iter)
    throughput: 90.55 M elements/sec
    speedup:    11.17x vs std::atan2
  RVV (pcl::atan2_RVV_f32m2):
    time:       60.811 ms (0.608 ms/iter)
    throughput: 107.77 M elements/sec
    speedup:    13.29x vs std::atan2, 1.19x vs scalar
```

---

## 4. 适用范围与限制

该实现以单精度为目标，偏向“吞吐优先”的场景：例如梯度方向、法线夹角、方向量化等后续只需要中等精度的几何计算。它不等价于 `libm` 的全规格 `atan2`：没有覆盖 NaN/Inf 等所有边界输入的逐项语义，也不以最后若干 ulp 的误差为目标。若调用方需要严格遵循标准库的特殊值传播规则，应继续使用 `std::atan2` 或 `atan2f`。

---

## 5. 参考

- [`Speeding up atan2f by 50x`](https://mazzo.li/posts/vectorized-atan2.html)

