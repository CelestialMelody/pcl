# RVV 条件判断与掩码控制流

本文说明在 RISC-V Vector（RVV 1.0）上，如何把标量里的 `if / else`、区间判断、分桶（离散化）等逻辑，改写为掩码（mask）驱动的向量代码，并配合仓库中 `common.hpp`、`edge.hpp` 的优化实现加以对照。

---

## 1. 向量里的“分支”处理

标量代码习惯用分支按元素处理不同情况，而在 RVV 里常规做法是：

1. 用比较指令得到 `vbool*_t`（布尔掩码寄存器）；
2. 用掩码逻辑（`vmand`、`vmor` 等）组合复合条件；
3. 用 `vmerge` 做按 lane 选择，或用 `vcompress` 压缩有效元素，或用 带 `_m` 后缀的掩码算术 避免对非法 lane 执行危险运算。

这样整条向量指令流对当前 `vl` 内所有 lane 一致执行，由掩码决定“谁参与、谁被替换、谁保持不动”。

---

## 2. 常见应用

### 2.1 比较生成掩码

浮点可用 `vmflt`、`vmfle`、`vmfgt`、`vmfge`、`vmfeq` 等；整数可用 `vmseq`、`vmsne`、`vmslt` 等。结果写入掩码类型（如 `vbool16_t`，与 LMUL/SEW 组合相关），表示每个 lane 条件是否成立。

### 2.2 复合条件：用掩码逻辑，不用嵌套 `if`

例如闭区间 A \le x \le B：分别生成 `x >= A` 与 `x <= B` 的掩码，再 `vmand_mm` 按位与。标量式的“先判断外再判断内”在向量里应展开为并行比较 + 掩码与/或。

### 2.3 数据选择：`vmerge`

`vmerge` 在寄存器级按掩码从两路源中拼出结果。（书写时务必对照 intrinsic 参数顺序）

### 2.4 稀疏写出：`vcompress` 与 `vcpop`

- `vcompress`：按掩码把“为真”的 lane 紧挨排到向量低位，便于连续写回内存（如索引列表）。
- `vcpop_m`：统计掩码在 `vl` 范围内为 1 的个数；可用于仅在有个数大于 0 时才存储，避免无意义的访存。

### 2.5 危险运算：必须用掩码形式（`_m`）

若先对所有 lane 做除法、`log` 等再 `vmerge`，掩码为 0 的 lane 仍可能执行运算，从而触发异常或产生 NaN 污染。应对“安全子集”使用带掩码参数的算术 intrinsic（名称常含 `_m`），使硬件仅对掩码为 1 的 lane 执行该运算；未参与 lane 的行为受 tail/mask 策略约束（参见仓库内 [Tail-Agnostic / Tail-Undisturbed](Tail-Agnostic-Tail-Undisturbed.zh.md) 等说明）。

### 2.6 块级跳过：`vcpop` 与标量 `if`

若整段 `vl` 内没有任何 lane 满足条件，后续重计算或存储可跳过。比如：

```cpp
if (__riscv_vcpop_m_b16(valid_mask, vl) == 0)
  continue;
```

注意：这与“向量内分支”不同——标量 `if` 只跳过整块 strip，lane 之间仍无分叉；适合粗粒度省带宽与算力。

---

## 3. 实例

### 3.1 `atan2_RVV_f32m2`：比较 + `vmerge` 实现象限与交换

`pcl::atan2_RVV_f32m2` 用 `|y| > |x|` 的掩码交换分子分母，再对交换后的结果做象限修正；全程用比较生成掩码，用 `vmerge` 选择不同分支对应的数值，避免按 lane 分支。

```cpp
  // swap when |y| > |x|; vmerge(op1, op2, mask) => mask ? op2 : op1 (match atan2.cpp)
  const vbool16_t swap_mask = __riscv_vmflt_vv_f32m2_b16 (abs_x, abs_y, vl);
  const vfloat32m2_t num = __riscv_vmerge_vvm_f32m2 (y, x, swap_mask, vl);
  const vfloat32m2_t den = __riscv_vmerge_vvm_f32m2 (x, y, swap_mask, vl);
  // ...
  const vfloat32m2_t adj = __riscv_vmerge_vvm_f32m2 (neg_pi_2, pi_2_vec, atan_ge_zero, vl);
  result = __riscv_vmerge_vvm_f32m2 (result, __riscv_vfsub_vv_f32m2 (adj, result, vl), swap_mask, vl);
  // ...
  const vfloat32m2_t add_val = __riscv_vmerge_vvm_f32m2 (neg_pi, pi_vec, y_ge_zero, vl);
  result = __riscv_vmerge_vvm_f32m2 (result, __riscv_vfadd_vv_f32m2 (result, add_val, vl), x_lt_zero, vl);
```

同一套多项式逼近适用于已交换/未交换的 `atan_input`，象限与 \pi 修正完全由掩码与 `vmerge` 完成。

### 3.2 `expf_RVV_f32m2`：整数相等比较 + `vmerge` 处理特例

构造 2^n 时，正常情况用指数域移位；n = -127 时需落到非规格化常数。这里用 `vmseq` 得到“是否 n=-127”的掩码，再用 `vmerge` 在两路 `2^n` 结果中选一路。

```cpp
  const vbool16_t is_n_neg127 = __riscv_vmseq_vx_i32m2_b16 (n, -127, vl);
  vfloat32m2_t two_n_sub = __riscv_vfmv_v_f_f32m2 (kExpfTwoToMinus127, vl);
  vfloat32m2_t two_n = __riscv_vmerge_vvm_f32m2 (two_n_normal, two_n_sub, is_n_neg127, vl);
  return __riscv_vfmul_vv_f32m2 (exp_r, two_n, vl);  // exp(x) = 2^n * exp(r)
```

标量里会写 `if (n == -127) ...`；向量里改为并行比较 + 合并，两路都算出再选，代价通常仍优于分支。

### 3.3 `getPointsInBoxRVV`：`vmand` 链 + `vcompress` + `vcpop`

轴向盒筛选：每个维度上“≥ min 且 ≤ max”用两次比较与一次 `vmand`，三轴掩码再 `vmand` 得到“点在盒内”掩码。索引用 `vid` 与 `vcompress` 压紧，仅当 `vcpop` > 0 时写入 `indices`，避免空掩码时的多余存储。

```cpp
    vbool16_t in_x = __riscv_vmfge_vf_f32m2_b16 (vx, min_x, vl);
    in_x = __riscv_vmand_mm_b16 (in_x, __riscv_vmfle_vf_f32m2_b16 (vx, max_x, vl), vl);
    vbool16_t in_y = __riscv_vmfge_vf_f32m2_b16 (vy, min_y, vl);
    in_y = __riscv_vmand_mm_b16 (in_y, __riscv_vmfle_vf_f32m2_b16 (vy, max_y, vl), vl);
    vbool16_t in_z = __riscv_vmfge_vf_f32m2_b16 (vz, min_z, vl);
    in_z = __riscv_vmand_mm_b16 (in_z, __riscv_vmfle_vf_f32m2_b16 (vz, max_z, vl), vl);
    vbool16_t mask = __riscv_vmand_mm_b16 (__riscv_vmand_mm_b16 (in_x, in_y, vl), in_z, vl);

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
```

这是“掩码筛选 + 压缩索引”的例子，与仅用 `vmerge` 改写数据值的应用场景互补。

### 3.4 `discretizeAnglesRVV`：区间掩码与链式 `vmerge` 分桶

将弧度转为角度后，负角度先折叠到 [0, 180)（比较 + `vmerge`）；再在 (22.5, 67.5) 等区间上用 `vmand` 组合上下界，最后用多次 `vmerge` 把 0°、45°、90°、135° 离散方向写入（未命中任何区间时保持 0° 基准）。

```cpp
    vbool16_t m_neg = __riscv_vmflt_vf_f32m2_b16(v_deg, 0.0f, vl);
    vfloat32m2_t v_deg_fold =
        __riscv_vfadd_vf_f32m2(v_deg, 180.0f, vl);
    v_deg = __riscv_vmerge_vvm_f32m2(v_deg, v_deg_fold, m_neg, vl);
    // ...
    vbool16_t m45 = __riscv_vmfgt_vf_f32m2_b16(v_deg, 22.5f, vl);
    m45 = __riscv_vmand_mm_b16(m45, __riscv_vmflt_vf_f32m2_b16(v_deg, 67.5f, vl), vl);
    // ... m90, m135 同理 ...
    // vmerge(op1, op2, mask) => mask ? op2 : op1; remainder is 0°
    vfloat32m2_t result = __riscv_vmerge_vvm_f32m2(v_0, v_45, m45, vl);
    result = __riscv_vmerge_vvm_f32m2(result, v_90, m90, vl);
    result = __riscv_vmerge_vvm_f32m2(result, v_135, m135, vl);
```

多个互斥区间在标量里常是 `if / else if`；向量里用各区间掩码 + 链式 `vmerge`，后写的桶覆盖先写的。

### 3.5 区间保留：`vmerge` 置零

```cpp
vbool16_t ge_a = __riscv_vmfge_vf_f32m2_b16(v_data, A, vl);
vbool16_t le_b = __riscv_vmfle_vf_f32m2_b16(v_data, B, vl);
vbool16_t inside = __riscv_vmand_mm_b16(ge_a, le_b, vl);
vfloat32m2_t zero = __riscv_vfmv_v_f_f32m2(0.0f, vl);
vfloat32m2_t out = __riscv_vmerge_vvm_f32m2(zero, v_data, inside, vl);
```

### 3.6 安全除法：掩码除法 intrinsic

```cpp
vbool16_t ok = __riscv_vmfgt_vf_f32m2_b16(v_x, 0.0f, vl);
vfloat32m2_t zero = __riscv_vfmv_v_f_f32m2(0.0f, vl);
vfloat32m2_t inv = __riscv_vfdiv_vf_f32m2_m(ok, zero, 1.0f, v_x, vl);
```

---

## 4. 小结


| 需求           | 常用手段                                  | 本仓库参考                                   |
| ------------ | ------------------------------------- | --------------------------------------- |
| 按条件选值        | 比较 + `vmerge`                         | `atan2_RVV_f32m2`、`discretizeAnglesRVV` |
| 复合条件         | `vmand` / `vmor`                      | `getPointsInBoxRVV`、角度分桶                |
| 筛索引/稀疏写      | `vcompress` + `vcpop`                 | `getPointsInBoxRVV`                     |
| 块内全无命中       | `vcpop == 0` 时 `continue`（标量跳过 strip） | 可与各内核组合使用                               |
| 避免非法 lane 运算 | 带掩码的算术 `_m`                           | 实现时查 intrinsics 手册                      |


