# expf_RVV_f32m2 实现说明

本文描述 `pcl::expf_RVV_f32m2` 的算法与 RVV 实现细节，并给出板卡侧测试数据。该函数面向批量 `float` 输入，返回与 `std::expf` 同量级的数值结果；实现以单精度吞吐为目标，输入在向量内夹取到 \([-88,88]\)。

---

## 1. 接口与语义

接口声明位于 `common/include/pcl/common/common.h`，实现位于 `common/include/pcl/common/impl/common.hpp`，仅在 `__RVV10__` 下可用。`vl` 由调用方通过 `vsetvl_e32m2` 设定，支持 strip-mining 与尾部处理。

```205:206:common/include/pcl/common/common.h
  expf_RVV_f32m2(const vfloat32m2_t& x, const std::size_t vl);
```

---

## 2. 数学路径与实现映射（约化 → 多项式 → 重构）

指数函数在远离 0 的区域直接做多项式展开会出现明显的误差与数值问题。这里采用常见的三段式结构，把计算约化到小区间内完成，再恢复到原尺度：

\[
e^x = e^{n\ln 2 + r} = 2^n\cdot e^r
\]

### 2.1 参数来源

本实现依赖三类常量：\(\ln 2\) 的拆分常量（`kExpfLog2Inv/kExpfLog2Hi/kExpfLog2Lo`）、输入夹取区间（`kExpfXMin/kExpfXMax`）以及 Remez 多项式系数（`kExpfRemezC0..C7`）。这些常量在仓库内有对应脚本与 Makefile 入口，便于复现与校验。

- **\(\ln 2\) 拆分与基础常量**：`test-rvv/common/common/script/parms.py` 使用 `decimal` 高精度计算 \(\ln 2\)，再将 `ln2` 的高位部分量化为 float32（脚本输出为 `kLog2Hi`），低位部分为 `kLog2Lo = ln2 - kLog2Hi`，并输出 `kLog2Inv = 1/ln2` 与 `kTwoToMinus127 = 2^-127` 等常量。对应 Makefile 目标为 `parms`。

- **Remez 系数**：`test-rvv/common/common/script/parms_remez_exp.py` 在区间 \([0,\ln 2]\) 上用 Remez（exchange）生成 `exp(r)` 的 degree=7 minimax 多项式，并按 Horner 形式输出 `expf_remez_c0..c7`。对应 Makefile 目标为 `parms_remez_exp`。

参数获取方式如下：

```bash
cd test-rvv/common/common
python3 -m venv .venv
source .venv/bin/activate
pip install numpy
make parms_remez_exp
```

### 2.2 范围约化

实现先将输入夹取到 \([-88,88]\)，避免上溢/下溢；随后计算：

- \(n \approx x/\ln 2\)（实现用 `kExpfLog2Inv = 1/ln2` 乘法得到 `flt_n`）
- `n` 由 `vfcvt_x_f` 转为整数，并再转回 `float` 得到 `flt_n`（用于后续 FMA）
- \(r = x - n\cdot \ln 2\)，其中 `ln2` 被拆成 `kExpfLog2Hi` 与 `kExpfLog2Lo` 两段，减少相减时的舍入损失

对应代码如下：

```cpp
  const float kExpfLog2Inv  = 1.4426950408889634f;     // 1 / ln(2)
  const float kExpfLog2Hi   = 0.6931471824645996f;     // ln(2) 高精度部分
  const float kExpfLog2Lo   = -1.904654290582768e-09f; // ln(2) 低精度部分（补偿）
  const float kExpfXMax     = 88.0f;
  const float kExpfXMin     = -88.0f;
  // ...
  vfloat32m2_t vx = __riscv_vfmin_vf_f32m2 (__riscv_vfmax_vf_f32m2 (x, kExpfXMin, vl), kExpfXMax, vl);
  vfloat32m2_t flt_n = __riscv_vfmul_vf_f32m2 (vx, kExpfLog2Inv, vl);
  vint32m2_t n = __riscv_vfcvt_x_f_v_i32m2 (flt_n, vl);
  flt_n = __riscv_vfcvt_f_x_v_f32m2 (n, vl);
  vfloat32m2_t r = __riscv_vfnmsub_vf_f32m2 (flt_n, kExpfLog2Hi, vx, vl);
  r = __riscv_vfnmsub_vf_f32m2 (flt_n, kExpfLog2Lo, r, vl);
```

### 2.3 多项式逼近

在约化后的 \(r\) 上，使用 7 次 Remez 多项式逼近 \(e^r\)。实现采用 Horner 形式，依次执行 `poly = c_k + r*poly`，最后得到 `exp_r = c0 + r*poly`。系数在 `common.hpp` 里以常量形式给出。

```cpp
  const float kExpfRemezC0  = 9.9999999998e-01f;
  const float kExpfRemezC1  = 1.0000000154e+00f;
  const float kExpfRemezC2  = 4.9999959620e-01f;
  const float kExpfRemezC3  = 1.6667078702e-01f;
  const float kExpfRemezC4  = 4.1645250213e-02f;
  const float kExpfRemezC5  = 8.3952782982e-03f;
  const float kExpfRemezC6  = 1.2887034349e-03f;
  const float kExpfRemezC7  = 2.8147688485e-04f;
  // Horner
  vfloat32m2_t poly = __riscv_vfmv_v_f_f32m2 (kExpfRemezC7, vl);
  poly = __riscv_vfmacc_vv_f32m2 (__riscv_vfmv_v_f_f32m2 (kExpfRemezC6, vl), r, poly, vl);
  poly = __riscv_vfmacc_vv_f32m2 (__riscv_vfmv_v_f_f32m2 (kExpfRemezC5, vl), r, poly, vl);
  poly = __riscv_vfmacc_vv_f32m2 (__riscv_vfmv_v_f_f32m2 (kExpfRemezC4, vl), r, poly, vl);
  poly = __riscv_vfmacc_vv_f32m2 (__riscv_vfmv_v_f_f32m2 (kExpfRemezC3, vl), r, poly, vl);
  poly = __riscv_vfmacc_vv_f32m2 (__riscv_vfmv_v_f_f32m2 (kExpfRemezC2, vl), r, poly, vl);
  poly = __riscv_vfmacc_vv_f32m2 (__riscv_vfmv_v_f_f32m2 (kExpfRemezC1, vl), r, poly, vl);
  vfloat32m2_t exp_r = __riscv_vfmacc_vv_f32m2 (__riscv_vfmv_v_f_f32m2 (kExpfRemezC0, vl), r, poly, vl);
```

### 2.4 重构：构造 \(2^n\) 并相乘

本实现按 IEEE754 `float` 的指数位构造 \(2^n\)：`exp_offset = n + 127`，夹取到 \([0,255]\)，再左移 23 位并 `vreinterpret` 为 `float`。对 `n == -127` 单独合并一个非规格化常量 `kExpfTwoToMinus127`，保持 `2^-127` 的数值形态。

最后返回 `exp_r * two_n`。

```cpp
  vint32m2_t exp_offset = __riscv_vadd_vx_i32m2 (n, 127, vl);
  exp_offset = __riscv_vmax_vx_i32m2 (exp_offset, 0, vl);
  exp_offset = __riscv_vmin_vx_i32m2 (exp_offset, 255, vl);
  vuint32m2_t res_bits =
      __riscv_vsll_vx_u32m2 (__riscv_vreinterpret_v_i32m2_u32m2 (exp_offset), 23, vl);
  vfloat32m2_t two_n_normal = __riscv_vreinterpret_v_u32m2_f32m2 (res_bits);
  const vbool16_t is_n_neg127 = __riscv_vmseq_vx_i32m2_b16 (n, -127, vl);
  vfloat32m2_t two_n_sub = __riscv_vfmv_v_f_f32m2 (kExpfTwoToMinus127, vl);
  vfloat32m2_t two_n = __riscv_vmerge_vvm_f32m2 (two_n_normal, two_n_sub, is_n_neg127, vl);
  return __riscv_vfmul_vv_f32m2 (exp_r, two_n, vl);
```

---

## 3. 测试与结果

### 3.1 expf_test：与 std::expf 对比

板卡侧日志位于 `test-rvv/common/common/output/board/expf_test.log`。该测试对 `n=10000` 的输入向量比较最大相对误差，并进行 100 次迭代计时：

```text
=== expf approximation vs std::expf (n = 10000) ===
  Scalar Remez (aligned with common.hpp constants):
    max relative error:  1.517213e-06
  RVV (pcl::expf_RVV_f32m2 from common.hpp):
    max relative error:  1.517213e-06
  RVV vs scalar Remez max diff: 0.000000e+00

=== Performance (n = 10000, 100 iters) ===
  std::expf:       42.754 ms
  Scalar Remez:   127.095 ms  (speedup: 0.34x vs std)
  RVV common.hpp:   4.634 ms  (speedup: 9.23x vs std, 27.43x vs scalar)
```

`Scalar Remez` 在该测试中慢于 `std::expf`，原因是测试的标量路径使用了 `std::ldexpf` 重构 \(2^n\)，且没有利用向量并行；而 RVV 路径用位构造与向量 FMA 把多数步骤放入向量流水线中。

### 3.2 expf_remez_vs_taylor：Remez 与 Taylor 对比

板卡侧日志位于 `test-rvv/common/common/output/board/expf_remez_vs_taylor.log`。它对比 Remez 与 Taylor（同为 7 次）在误差与耗时上的差异，并区分了 RVV 的两种 \(2^n\) 计算方式（查表/位构造）：

```text
=== expf approximation vs std::expf (n = 10000) ===
  Scalar Remez (degree 7):
    max relative error:  1.517213e-06
  Scalar Taylor (degree 7):
    max relative error:  2.949001e-01
  RVV Remez (table lookup 2^n):
    max relative error:  1.517213e-06
  RVV Remez (construct 2^n):
    max relative error:  1.517213e-06
  RVV Taylor:
    max relative error:  2.949001e-01
  // ...
=== Performance (n = 10000, 100 iters) ===
  std::expf:          42.735 ms
  Scalar Remez:     130.796 ms  (speedup: 0.33x vs std)
  RVV Remez [table 2^n]:         4.720 ms  (speedup: 9.05x vs std, 27.71x vs scalar)
  RVV Remez [construct 2^n]:     4.061 ms  (speedup: 10.52x vs std, 32.21x vs scalar)
```

---

## 4. 适用范围与限制

该实现以单精度为目标，偏向吞吐优先的数值内核：它对输入做 \([-88,88]\) 夹取，且不覆盖 `std::expf` 对 NaN/Inf 的完整逐项语义（测试程序的标量路径会显式处理 NaN/Inf，但 `common.hpp` 的 RVV 版本以夹取后的有限值为主要目标）。在需要严格复现标准库特殊值传播规则的场景，应继续使用 `std::expf`。