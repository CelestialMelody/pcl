# Tail Agnostic / Undisturbed 机制与跨迭代累加约束

本文档记录了 RISC-V Vector (RVV) 扩展中 TA/TU（尾部无关/尾部保留）机制的工程约束，以及在 Strip-mining 循环中维护向量累加器的设计规范。

## 1. 规范约束与硬件行为差异

RVV 扩展通过 `vsetvli` / `vsetvl` 指令中的 `vta` (Vector Tail Agnostic) 与 `vma` (Vector Mask Agnostic) 标志位，控制非活动通道（Inactive elements）与尾部元素（Tail elements）的硬件写入策略。汇编层需显式声明其行为模式（见 RISC-V “V” Vector Extension 规范 中 [Vector Tail Agnostic and Vector Mask Agnostic](https://github.com/riscvarchive/riscv-v-spec/blob/master/v-spec.adoc#343-vector-tail-agnostic-and-vector-mask-agnostic-vta-and-vma) 一节）。

在应用层代码中，两者的具体表现如下：

- **`ta` (Tail Agnostic)**：索引 $\ge vl$ 的尾部元素内容处于未定义状态。为换取更高的微架构执行效率，硬件可能在尾部通道写入全 1 (All-1s) 或保持无序状态。软件层不可依赖尾部数据。

- **`tu` (Tail Undisturbed)**：硬件严格保持非活动通道和尾部通道的现有寄存器状态。跨迭代期间，原有的寄存器内容可作为有效的中间状态保留。

  *(注：`vma` 的行为与 `vta` 类似；但在掩码操作中，掩码目的寄存器的尾部始终按 Tail Agnostic 规则处理。)*

## 2. Strip-mining 循环中的累加器污染陷阱

在 `vl` 动态变化的循环（例如 `while (remaining) { vl = vsetvl_e32m2(remaining); ... }`）中，若使用单条向量寄存器作为跨迭代的累加状态（如 `v_acc = vfadd(v_acc, chunk)`），存在引发数据损坏的风险。

- **陷阱表现**：若浮点/整数运算使用了默认的 `_ta` 策略（或等价指令），当某次迭代的有效向量长度 $vl < VLMAX$ 时，超出 $vl$ 的尾部通道会被写入未定义值。
- **错误传导**：在后续迭代中，若 $vl$ 重新增大，或者最终浮点归约使用的长度达到 VLMAX（如 `vfredosum` / `vfredusum` 的第三参数为初始化时得到的 `max_vl` 且 $n \ge \text{VLMAX}$），先前被污染的尾部脏数据将进入求和链路，导致规约结果错误。
- **归因**：此行为符合 ISA 规范预期。在 `ta` 策略下，软件系统不应假定尾部通道会继承上一条指令的中间状态。

## 3. RVV 累加器实现建议

**跨迭代累加器使用 `_tu`**：

对于声明周期跨越多次 `vl` 更新的向量累加器，相关算术指令优先使用带 `_tu`（Tail Undisturbed）后缀的内建函数（如 `__riscv_vfadd_vv_f32m2_tu`）。这确保了当 $vl$ 缩小后再次放大时，尾部通道未被破坏，从而保证最终归约操作（Reduction）的数据正确性。

**单次迭代临时变量使用 `_ta`** ：

对于生命周期仅限于当前 `vl` 范围内的临时结果，安全地使用默认的 `_ta` 策略。放宽尾部约束可降低寄存器重命名与状态依赖的开销，赋予底层微架构更大的调度空间。

## 4. 案例：Strip-mining 累加与归约

`calculatePolygonAreaRVV`、`getMeanStdKernelRVV`（位于 `common/include/pcl/common/impl/common.hpp`）与 `sumReduceRVV`（位于 `2d/include/pcl/2d/impl/kernel.hpp`）采用了完全一致的 Strip-mining 归约范式：在 `while` 循环内部利用动态的 `vl` 更新向量累加器（`v_acc*`），循环结束后，采用**进入循环前预先配置的最大向量长度**对累加器执行浮点规约。

| **模块函数**              | **归约指令** | **归约长度参数控制**                                   |
| ------------------------- | ------------ | ------------------------------------------------------ |
| `calculatePolygonAreaRVV` | `vfredusum`  | `vlmax = setvl_e32m2(~0)`（硬件 VLMAX）                |
| `getMeanStdKernelRVV`     | `vfredosum`  | `max_vl = setvl_e32m2(n)`（即 $\min(n,\text{VLMAX})$） |
| `sumReduceRVV`            | `vfredosum`  | 同上                                                   |

若循环体内的累加指令（`vfadd` / `vfmacc`）使用默认的 `_ta` 策略，当某次迭代的 $vl < \text{VLMAX}$ 时，超出 $vl$ 边界的尾部通道（Tail lanes）将被硬件置为未定义状态。由于最终归约指令的覆盖长度可能大于该次迭代的 $vl$（例如 $n \ge \text{VLMAX}$ 时 `max_vl` 等于 VLMAX，或面积计算强制使用 `vlmax` 规约），这些未定义数据将被引入规约计算链路，导致面积分量、统计均值或卷积核总和发生数值偏移。

因此，这三处针对**跨迭代累加器**的更新严格使用了 `_tu` 策略；而仅在当前 $vl$ 生命周期内消费的临时变量（如叉积的各轴分量 `cx`/`cy`/`cz`）则保留默认的 `_ta` 策略以换取微架构调度空间。

**`calculatePolygonAreaRVV`**

累加相邻边的叉积分量。`vfadd` 使用 `_tu` 保护尾部状态，最终通过 `vfredusum(..., vlmax)` 执行全长归约。

```cpp
    v_acc_x = __riscv_vfadd_vv_f32m2_tu (v_acc_x, v_acc_x, cx, vl);
    v_acc_y = __riscv_vfadd_vv_f32m2_tu (v_acc_y, v_acc_y, cy, vl);
    v_acc_z = __riscv_vfadd_vv_f32m2_tu (v_acc_z, v_acc_z, cz, vl);
    i += vl;
  }

  vfloat32m1_t v_zero = __riscv_vfmv_s_f_f32m1 (0.0f, 1);
  float rx = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredusum_vs_f32m2_f32m1 (v_acc_x, v_zero, vlmax));
  float ry = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredusum_vs_f32m2_f32m1 (v_acc_y, v_zero, vlmax));
  float rz = __riscv_vfmv_f_s_f32m1_f32 (__riscv_vfredusum_vs_f32m2_f32m1 (v_acc_z, v_zero, vlmax));
```

**`getMeanStdKernelRVV`**

`v_acc_sum` 与 `v_acc_sq` 承担跨迭代累加任务。`vfadd` 与乘加指令 `vfmacc` 均使用 `_tu`（`vfmacc` 的 `_tu` 与 `_ta` 签名均为四参数，区分在于尾部语义）

```cpp
    v_acc_sum = __riscv_vfadd_vv_f32m2_tu (v_acc_sum, v_acc_sum, v, vl);
    v_acc_sq  = __riscv_vfmacc_vv_f32m2_tu (v_acc_sq, v, v, vl);
    i += vl;
  }
  vfloat32m1_t v_zero = __riscv_vfmv_s_f_f32m1 (0.0f, 1);
  vfloat32m1_t v_sum  = __riscv_vfredosum_vs_f32m2_f32m1 (v_acc_sum, v_zero, max_vl);
  vfloat32m1_t v_sq   = __riscv_vfredosum_vs_f32m2_f32m1 (v_acc_sq,  v_zero, max_vl);
```

**`sumReduceRVV`**

针对紧凑 `float` 缓冲区的归约操作，逻辑结构与上述案例一致。

```cpp
  while (j0 < n) {
    std::size_t vl = __riscv_vsetvl_e32m2(n - j0);
    vfloat32m2_t v_buf = __riscv_vle32_v_f32m2(buf.data() + j0, vl);
    v_acc = __riscv_vfadd_vv_f32m2_tu(v_acc, v_acc, v_buf, vl);
    j0 += vl;
  }
  vfloat32m1_t v_zero = __riscv_vfmv_s_f_f32m1(0.0f, 1);
  vfloat32m1_t v_sum =
      __riscv_vfredosum_vs_f32m2_f32m1(v_acc, v_zero, max_vl);
```
