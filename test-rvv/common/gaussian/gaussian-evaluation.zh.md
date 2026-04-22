# `common/src/gaussian.cpp`：函数级梳理、筛选评估与 RVV 优先级

本文档是 `common/src/gaussian.cpp` 在 test-rvv/common/gaussian 下的规范说明。

---

## 1. 函数 / 函数族梳理

文件实现 `pcl::GaussianKernel` 的四个非模板成员，均随 `libpcl_common` 链接供调用方使用。

### 1.1 `GaussianKernel::compute`

| 符号（语义） | 功能概要 | 复杂度与访存 |
| ------------ | -------- | ------------ |
| `compute(sigma, kernel, kernel_width)` | 生成一维高斯核：`exp` 对称填充、中心置 1、按相对幅值阈值收缩宽度、平移、`kernel /= sum()`。失败抛 `KernelWidthTooSmallException`。 | 主循环约 O(hw)，`hw = kernel_width/2`；随后 O(g_width) 移位与归一化；分支与 `fabs` 较多。 |
| `compute(sigma, kernel, derivative, kernel_width)` | 在上式基础上同时生成导数核；两路有效宽度可能不同；导数再按一阶矩条件归一化。 | 同上量级，额外向量与一次 O(d_width) 归约。 |

### 1.2 `GaussianKernel::convolveRows` / `convolveCols`（`PointCloud<float>`）

| 符号（语义） | 功能概要 | 复杂度与访存 |
| ------------ | -------- | ------------ |
| `convolveRows(input, kernel, output)` | 将 `PointCloud<float>` 视为 width×height 栅格，对 每一行 做一维卷积；左右各 `radius` 列输出 0；中间列累加 `input(l,j)*kernel[k]`。 | O (height × width × kernel_size)；行向 连续 读 `float`。 |
| `convolveCols(input, kernel, output)` | 对 每一列 做一维卷积；上下各 `radius` 行输出 0。 | O (width × height × kernel_size)；列向 步长 `width` 的 strided 读。 |

---

## 2. 函数级筛选：评估标准说明

对 `gaussian.cpp` 内每个函数族，从下列维度做 低 / 中 / 高 定性，并映射到 §3。

| 维度 | 含义 | 与向量化 / RVV 的关系 |
| ---- | ---- | ---------------------- |
| 循环规模 | 主循环 trip count | 卷积为 width×height×kernel；`compute` 为 kernel_width 量级。 |
| 算术密度 | 乘加、exp、除法占比 | 卷积内层为 FMA；`compute` 以 exp 与归一化为主。 |
| 访存规整性 | 连续 / 固定步长 / 不规则 | 行卷积 连续；列卷积 固定大步长。 |
| 分支复杂度 | 边界段、别名分支、异常 | 卷积外两层分段填零；`input==output` 分支；`compute` 多分支。 |
| 数值语义风险 | 累加顺序、非结合律 | 卷积有限项和，顺序变化可致 浮点细差；需容差验收。 |
| 可测试性 | 固定输入与参考解 | 小图手写参考、`compute` 与上游 `test/common/test_gaussian.cpp` 对齐。 |

---

## 3. 优先级与向量化候选总表

| 优先级 | 状态 | 函数 / 函数族 | 优化方向（语义层） | 主要风险 | 预期收益 | 回退条件 |
| ------ | ---- | --------------- | ------------------- | -------- | -------- | -------- |
| 高 | 已完成 | `convolveRows` / `convolveCols`（`PointCloud<float>`） | 内层乘加、边界分段；行/列访存分别处理 | 列向带宽；累加顺序浮点差 | 大图、长核时 中–高 | bench 无收益或小图则标量 |
| 低 | 暂缓 | `compute`（单核 / 核+导数） | 短循环向量化辅助 | 循环短、`exp` 与分支多 | 低 | profile 非热点则不做 |
| 低 | 暂缓 | 别名分支（整云拷贝） | 结构性（缓冲策略）非 SIMD 核心 | 内存拷贝成本 | 视场景 | 非 RVV 主线 |

---

## 4. 与后续流程的衔接

1. Bench 与 Baseline：在修改 `gaussian.cpp` 前，使用 `test-rvv/common/gaussian` 中 仅调用本文件实现的 API 的 bench，在标量构建下记录 QEMU / 板卡 耗时（`output/qemu/run_bench_std.log` 等）。
2. 编译器诊断：  
   - 整库链接的 bench 对 `fopt-info-vec-missed` 往往只见驱动代码；  
   - 使用 Makefile 目标 `compile_gaussian_cpp`（仅编译 `common/src/gaussian.cpp` 为目标文件）可得到 针对本文件循环 的 missed 日志，便于步骤「自动向量化分析」。
3. RVV 实现：优先卷积两函数；修改后 重编并安装 `pcl_common`，再跑 test/bench（避免与 `.so` 重复符号）。
4. 验证：单测（含 `compute`、卷积、原地别名）→ bench 对比 speedup；无收益则回退。

### 4.1 `libpcl_common` 与 vec-missed 日志

bench 链接已安装 `libpcl_common` 时，仅编译 `bench_gaussian.cpp` 得到的 `-fopt-info-vec-missed` 多半落在驱动代码，不一定包含 `gaussian.cpp` 内卷积循环 的行级诊断。

### 4.2 单独编译 `gaussian.cpp`（`make compile_gaussian_cpp`）的静态结论

在同一套优化选项下 只编译 `common/src/gaussian.cpp` 为目标文件，可在日志中看到 带 `gaussian.cpp` 行号 的 missed 信息。对当前实现，编译器对卷积部分常见表述包括：多层嵌套导致内层无法自动矢量化、列方向访问被判定为复杂/大步长模式、索引类型与目标 ISA 自动向量化限制等；边界与容器相关代码则常因 异常语义、内存屏障或「向量化不划算」 而未矢量化。这些结论用于说明 为何需要手工 RVV 分层，而非指望同一嵌套结构被 GCC 自动展开为向量循环。

---

## 5. 代码与测试索引

| 实现 | 源码 |
| ---- | ---- |
| 本筛选对象 | `common/src/gaussian.cpp` |
| 上游单测 | `test/common/test_gaussian.cpp`（`compute`） |
| test-rvv | `test-rvv/common/gaussian/test_gaussian.cpp`（覆盖 `gaussian.cpp` 中 `compute` 与 float 卷积等） |
| bench | `test-rvv/common/gaussian/bench_gaussian.cpp`（计时 `compute` 与 `convolveRows`/`convolveCols`） |
