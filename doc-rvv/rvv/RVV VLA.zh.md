# 向量长度 (Vector Length) 的差异：定长 vs 变长

## AVX：隐含的定长 (Fixed Length)

- **固定宽度**：AVX 指令（如 `_mm256_add_ps`）隐含了操作长度固定为 256 位（即 8 个 float）。
- **尾部处理 (Tail Handling)**：当数据总量 $N$ 不是 8 的倍数时，程序员必须编写额外的**标量循环**（Scalar Loop）或使用掩码（Masking）来处理剩下的 $N  8$ 个元素。

## RVV：显式的变长 (Vector Length Agnostic)

**vl 参数**

RVV Intrinsic 函数（如 `__riscv_vfmacc_..._f32m2`）都接受一个显式的 `vl` 参数。这个参数告诉硬件“这一条指令具体要处理多少个元素”。

**自适应循环 (Strip Mining)**

通过 vsetvl 指令，程序可以动态获取当前硬件支持的向量长度：

- 在循环的中间迭代，`vl` 等于硬件的最大向量长度（例如 8, 16, 32...）。
- **自动处理尾部**：在循环的最后一次迭代中，如果剩余元素（`avl`）不足一个满向量，`vsetvl` 会自动返回剩余的数量。
- **结果**：同一份代码无需任何修改即可运行在不同 VLEN（128位, 256位, 512位等）的硬件上，且不需要专门编写尾部处理代码。

**代码对比示意：**

```cpp
// AVX 风格 (伪代码)
for (int i = 0; i < n - 8; i += 8) { // 必须小心边界
    __m256 v = ...; // 处理 8 个
}
// 必须写额外的 Cleanup Loop 处理剩下的点
for (; i < n; i++) {
    float s = ...;
}

// RVV 风格
for (size_t i = 0; i < n; i += vl) {
    // 自动计算本次处理个数，最后一次可能是 3 或 5
    size_t vl = __riscv_vsetvl_e32m2(n - i);
    vfloat32m2_t v = ...; // 传入 vl 参数
}
```

