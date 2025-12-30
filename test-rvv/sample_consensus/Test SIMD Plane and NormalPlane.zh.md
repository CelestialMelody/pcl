# SampleConsensusModel (SIMD_countWithinDistance) 测试文档

## 1. 如何运行测试

**依赖环境**

- 支持SSE、AVX、RVV等指令集的处理器（或相应模拟器，如 QEMU 用于 RVV 测试）
- CMake构建系统
- 编译器（如GCC、Clang）需支持相应指令集扩展

**构建与运行步骤**

1. **配置CMake项目**：在项目根目录创建 `build`文件夹并进入

   ```bash
   mkdir build && cd build
   ```
   
2. **生成构建文件**：根据需要指定指令集编译选项

   ```bash
   # 基础构建（默认支持检测到的指令集）
   cmake ..
   ```
   
3. **编译并运行测试程序**

   ```bash
   make run
   ```

   > 也可以不使用 cmake 已经提供了 Makefile，直接 make run 即可
   >

## 2. SIMD_countWithinDistance 测试讲解

测试模型（`SampleConsensusModelNormalPlane`、`SampleConsensusModelPlane`、`SampleConsensusModelSphere`、`SampleConsensusModelCircle2D`）均用于验证 `countWithinDistance` 方法的SIMD（单指令多数据）实现与标准实现的一致性，并测试不同指令集（SSE、AVX、RVV）的性能表现。测试通过多次迭代生成随机点云数据，分别使用不同指令集计算符合距离阈值的点数量，并对比结果与性能。

> **说明**：相较于 PCL 原本的测试文件，本测试文件：
>
> - 新增 `total_time_standard`/`total_time_sse`/`total_time_avx`/`total_time_rvv` 四个计时变量
> - 使用 `std::chrono::high_resolution_clock` 对每种实现（Standard/SSE/AVX/RVV）进行毫秒级计时
> - 累加所有迭代的总耗时，最后计算并输出加速比（Speedup）

## 3. RVV移植验证

从代码实现来看，RVV（RISC-V Vector）移植已完成，具体表现为：

1. 代码中存在RVV专用实现的调用：`model.countWithinDistanceRVV(model_coefficients, threshold)`
2. 有完整的正确性验证逻辑：通过 `EXPECT_LE`宏对比RVV实现与标准实现的结果差异，允许最大误差为2
3. 包含RVV性能统计代码：记录总运行时间并计算相对于标准实现的加速比（`Speedup (Std/RVV)`）

若测试通过（即RVV实现与标准实现的结果差异在允许范围内），则可认为RVV移植功能上是成功的。

## 4. QEMU 中 RVV 性能不如标量代码的现象

根据官方Issue [#2137 RISC-V Vector Slowdowns](https://gitlab.com/qemu-project/qemu/-/issues/2137)及社区讨论，在QEMU模拟 RISC-V 架构 时，开启 RVV 向量扩展自动向量化编译的二进制程序，运行速度远慢于非向量化版本；且这类向量优化的程序在真实 RISC-V 硬件上是提速的，仅在 QEMU 模拟环境中出现严重降速。故为正常现象。
