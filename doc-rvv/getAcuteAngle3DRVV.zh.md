

# 3D 向量锐角计算 (`getAcuteAngle3D` & `acos`)

## 1. 功能与数学原理

`getAcuteAngle3D`（计算两个向量的锐角）和 `acos`（反余弦函数的快速近似）用于快速计算两个 3D 向量之间的**锐角**夹角。

> `getAcuteAngle3D` & `acos`函数位于 common.hpp 中。

### 1.1 `getAcuteAngle3D` 原理

计算两个向量 $\vec{v_1} = (x_1, y_1, z_1)$ 和 $\vec{v_2} = (x_2, y_2, z_2)$ 夹角的公式为：

$$\theta = \arccos \left( \frac{\vec{v_1} \cdot \vec{v_2}}{|\vec{v_1}| |\vec{v_2}|} \right)$$

**PCL 的特殊优化假设**：

1. **输入已归一化**：输入的向量要求是单位向量（模长为 1），因此分母 $|\vec{v_1}| |\vec{v_2}| = 1$，省略除法。

2. **只求锐角**：向量的点积可能是负数（表示钝角）。为了获得锐角（0 到 90度），我们需要对点积取绝对值：$| \vec{v_1} \cdot \vec{v_2} |$。

3. **数值稳定性 (Clamping)**：由于浮点误差，点积结果可能略微超过 1.0（例如 1.000001），这会导致 `acos` 返回 NaN。因此必须将输入限制在 `min(1.0, |dot|)`。

   > 向量单位化可从代码编写角度，以及测试文件 test_sample_consensus_plane_models.cpp 的使用角度看出。

### 1.2 `pcl::acos` (快速近似) 原理

标准的 `std::acos` 计算非常耗时。PCL 使用了一种基于多项式拟合的近似算法（见代码注释中的 Python 脚本引用），其形式为：

$$\arccos(x) \approx \text{mul\_term}(x) \cdot \sqrt{\text{sqrt\_term}(x)} + \text{add\_term}(x)$$

其中 `mul_term` 和 `add_term` 是关于 $x$ 的二次多项式，这种结构适合 SIMD 并行化。

------

## 2. 代码迁移分析：从 AVX 到 RVV

下面对比分析如何将 x86 AVX 代码转换为 RISC-V RVV 代码。

### 2.1 `pcl::acos` 实现对比

AVX 逻辑 (SIMD 指令堆叠):

AVX 使用显式的 add 和 mul 指令。对于多项式 $a + x(b + xc)$ (Horner 算法)，AVX 需要嵌套调用：

> AVX 也可采用 FMA 融合指令

```cpp
// AVX: 显式的乘法和加法
_mm256_add_ps(a, _mm256_mul_ps(x, _mm256_add_ps(b, _mm256_mul_ps(x, c))))
```

RVV 逻辑 (FMA 融合指令):

RISC-V 的 FMA (Fused Multiply-Accumulate) 指令 vfmacc，可以一条指令完成 a += b * c。这不仅减少了指令数，还提高了精度。

```cpp
// mul_term = a0 + x * (a1 + x * a2)
// 1. 计算内层: tmp = a1 + x * a2
//    RVV: vfmacc(a1, x, a2) -> 累加器是 a1，加上 x*a2
// 2. 计算外层: res = a0 + x * tmp
//    RVV: vfmacc(a0, x, tmp)
vfloat32m2_t mul_term = __riscv_vfmacc_vv_f32m2(a0, x, __riscv_vfmacc_vv_f32m2(a1, x, a2, vl), vl);
```

> **注意**：RISC-V 的 `vfmacc` 定义通常是 `vd = vd + vs1 * vs2`。我们在代码中利用这一点，将常数项 (`a0`, `a1`) 作为累加器的初始值（目标寄存器），从而完美实现多项式求值。

**转换策略**：将 AVX 的 `add(mul(...))` 结构转换为 RVV 的 `vfmacc` 链。

| **操作**       | **AVX (__m256)**                    | **RVV (vfloat32m2_t)**             | **说明**                        |
| -------------- | ----------------------------------- | ---------------------------------- | ------------------------------- |
| **广播常数**   | `_mm256_set1_ps(1.5f)`              | `__riscv_vfmv_v_f_f32m2(1.5f, vl)` | 将标量复制到整个向量寄存器      |
| **多项式计算** | `add(mul(x, add(b, mul(x, c))), a)` | `vfmacc(a, x, vfmacc(b, x, c))`    | RVV 使用 FMA 简化了 Horner 算法 |
| **平方根**     | `_mm256_sqrt_ps`                    | `__riscv_vfsqrt_v_f32m2`           | 直接对应                        |

------

### 2.2 `getAcuteAngle3D` 实现对比

#### A. 点积 (Dot Product)

- **AVX**: 需要分别计算 `x1*x2`, `y1*y2`, `z1*z2` 然后两次 `add`。

  ```cpp
  const __m256 dot_product = _mm256_add_ps (_mm256_add_ps (_mm256_mul_ps (x1, x2), _mm256_mul_ps (y1, y2)), _mm256_mul_ps (z1, z2));
  ```

- **RVV**: 使用 `vfmacc` 级联，无需中间临时变量。

  ```cpp
  // dot = x1*x2 + y1*y2 + z1*z2
  const vfloat32m2_t dot = __riscv_vfmacc_vv_f32m2(
    __riscv_vfmacc_vv_f32m2(
        __riscv_vfmul_vv_f32m2(x1, x2, vl),
        y1, y2, vl),
    z1, z2, vl);
  ```

#### B. 绝对值 (Absolute Value)

- **AVX (Bit Hack)**:

  ```cpp
  // 清除符号位：与 "-0.0f" (0x80000000) 进行 AND NOT 操作
  _mm256_andnot_ps(_mm256_set1_ps(-0.0f), dot_product)
  ```

- RVV (专用指令):

  RISC-V 提供了浮点符号注入指令 vfsgnj (Sign Injection)。

  - `vfsgnjx`: 结果符号 = `src1` 符号 XOR `src2` 符号。
  - **技巧**：如果 src1 和 src2 是同一个数，`Sign XOR Sign` 永远是 0 (正)。

  ```cpp
  // 绝对值：自己异或自己的符号位 -> 正数
  vfloat32m2_t dot_abs = __riscv_vfsgnjx_vv_f32m2(dot, dot, vl);
  ```

#### C. 数值截断 (Clamp to 1.0)

- **AVX**: `_mm256_min_ps(val, 1.0f)`
- **RVV**: `__riscv_vfmin_vf_f32m2(val, 1.0f, vl)` (直接使用向量-标量版本，更简洁)