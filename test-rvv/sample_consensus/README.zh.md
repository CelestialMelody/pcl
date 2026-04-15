# 测试说明文档

## 1. 如何运行测试

**依赖环境**

- **本机**：RISC-V 交叉工具链（`riscv64-unknown-linux-gnu-g++` 等）、QEMU user 模式（`qemu-riscv64`），以及与本 Makefile 中 `WORKSPACE`、`RISCV_DEPS` 一致的路径布局（Boost、Eigen-RVV、FLANN、已安装的 PCL 等）。
- **板卡**：与交叉编译产物 ABI 匹配的动态库（见下文 `board.mk`）。

本目录下 `**Makefile` 供开发机使用**：交叉编译后在 **QEMU** 中执行；`**board.mk` 供板卡本机使用**：在已部署二进制与 `.so` 的前提下直接运行，**不在板卡上编译**。

**板卡动态库**

先在仓库 `**test-rvv/`** 目录执行 `**make deploy_lib**`（见 **[test-rvv/README.md](../README.md)**），将精简后的 `.so` 同步到远端（默认 `~/pcl-test/lib`）。随后在板卡上把 `**board.mk`** 中的 `**REMOTE_LIBS_DIR**` 设为与该远端目录一致，再运行测试或基准。

---

### 1.1 本机：`Makefile`（交叉编译 + QEMU）

`plane_models/Makefile` 与 `quadric_models/Makefile` 结构相同：定义 `riscv64-unknown-linux-gnu-g++`、`-D__RVV10__`、Eigen RVV 相关宏、`USE_PCL_SOURCE_HEADERS`、链接 `pcl_sample_consensus` 等依赖，`run_test` / `run_bench` 通过 `qemu-riscv64 -cpu rv64,v=true,vlen=256,elen=64` 并设置 `LD_LIBRARY_PATH` 指向各依赖库的 `lib` 目录。

1. **NormalPlane（`plane_models`）— 功能测试与基准**
  ```bash
   cd test-rvv/sample_consensus/plane_models
   make run_test     # 构建 rvv_sac_plane_test，QEMU 运行；默认传入 pcd/sac_plane_test.pcd
   make run_bench    # 构建 bench_sac_normal_plane，QEMU 运行；同样可传 PCD
  ```
   可选：`make clean` 清理产物；`ARCH` 当前以 `riscv` 为主（与 Makefile 内分支一致）。
2. **Quadric（`quadric_models`）— 仅 GTest**
  ```bash
   cd test-rvv/sample_consensus/quadric_models
   make run_test     # 构建 rvv_sac_quadric_test，QEMU 运行（无额外命令行参数）
  ```

**本机部署到板卡**


| 目标                  | 作用                                                                       |
| ------------------- | ------------------------------------------------------------------------ |
| `make deploy_files` | 仅同步 `pcd/` 下测试用 PCD 到板卡（需配置 `REMOTE_USER` / `REMOTE_IP` / `REMOTE_DIR`）。 |
| `make deploy_test`  | 交叉编译、strip、`rsync` `**rvv_sac_plane_test`** 到板卡。                         |
| `make deploy_bench` | 同上，同步 `**bench_sac_normal_plane**`。                                      |


`quadric_models/Makefile` 仅提供 `**deploy_test**`（同步 `rvv_sac_quadric_test`），不包含 PCD 同步；板卡上需自行保证依赖库可用。

---

### 1.2 板卡：`board.mk`

将对应目录中的 `board.mk` 的内容拷到板卡相应位置的 `Makefile` 文件中使用。脚本只做两件事：设置 `LD_LIBRARY_PATH` 指向板卡上存放依赖 `.so` 的目录，再执行已部署的可执行文件。

`**plane_models/board.mk**`

- `REMOTE_TEST` / `REMOTE_BENCH`：可执行文件名（默认 `rvv_sac_plane_test`、`bench_sac_normal_plane`）。
- `REMOTE_PCD_FILE`：传给程序的 PCD 绝对路径（默认示例为 `/root/pcl-test/...`）。
- `REMOTE_LIBS_DIR`：动态库搜索路径（示例中为 `/root/pcl-test/libs`）。若已用顶层 `**test-rvv/Makefile**` 的 `**deploy_lib**` 部署到 `**~/pcl-test/lib**`，请将该变量改为与之**相同**的目录，避免 `lib` / `libs` 混用导致找不到 `.so`。

若在本机使用 `deploy_files`，PCD 会同步到 `**~/pcl-test/sample_consensus/plane_models/pcd/`**（以 Makefile 中 `REMOTE_DIR` 为准）；请把 `**REMOTE_PCD_FILE**` 改成与真实部署路径一致（含是否放在 `pcd/` 子目录）。

`**quadric_models/board.mk**`

- `REMOTE_TEST`：默认可执行文件 `rvv_sac_quadric_test`。
- `REMOTE_LIBS_DIR`：示例为 `~/pcl-test/libs`；与 `plane_models` 共用一套依赖库时应与 `**deploy_lib**` 目标目录及 `**plane_models/board.mk**` 保持一致。

---

### 1.3 头文件与链接库（本机 Makefile）

> **重要：头文件优先级（是否使用最新源码）**
>
> 当前 `test-rvv` 的 Makefile 默认启用 `USE_PCL_SOURCE_HEADERS=1`：**优先 include** `$(WORKSPACE)/pcl/**/include`（源码树），因此修改 `pcl` 的源文件后，`test-rvv` 下的 `make run_test` 会直接生效；同时仍然链接 `$(WORKSPACE)/riscv/pcl-rvv/lib` 下已安装的 `.so`。
>
> 若只需对齐已安装头文件，可 `**make USE_PCL_SOURCE_HEADERS=0 ...`**。

## 2. SIMD_countWithinDistance 测试说明

测试模型（`SampleConsensusModelNormalPlane`、`SampleConsensusModelPlane`、`SampleConsensusModelSphere`、`SampleConsensusModelCircle2D`）均用于验证 `countWithinDistance` 方法的SIMD（单指令多数据）实现与标准实现的一致性，并测试不同指令集（SSE、AVX、RVV）的性能表现。测试通过多次迭代生成随机点云数据，分别使用不同指令集计算符合距离阈值的点数量，并对比结果与性能。

> **说明**：相较于 PCL 原本的测试文件，本测试文件：
>
> - 新增 `total_time_standard`/`total_time_sse`/`total_time_avx`/`total_time_rvv` 四个计时变量
> - 使用 `std::chrono::high_resolution_clock` 对每种实现（Standard/SSE/AVX/RVV）进行毫秒级计时
> - 累加所有迭代的总耗时，最后计算并输出加速比（Speedup）

## 3. RVV 验证

检查是否有编译 RVV 代码

```bash
cd test-rvv/sample_consensus/plane_models
make run_test   # 或至少 make 以生成 build/rvv_sac_plane_test
riscv64-unknown-linux-gnu-objdump -dC build/rvv_sac_plane_test | rg "vsetvl" | head
riscv64-unknown-linux-gnu-objdump -dC build/rvv_sac_plane_test | rg "countWithinDistanceRVV"
```

若测试通过（即RVV实现与标准实现的结果差异在允许范围内），则可认为RVV移植功能上是成功的。

从代码实现来看，RVV（RISC-V Vector）移植已完成，具体表现为：

1. 代码中存在RVV专用实现的调用：`model.countWithinDistanceRVV(model_coefficients, threshold)`
2. 有正确性验证逻辑：对于随机 smoke 测试，允许极小计数差异（例如 `<= 2`）以避免阈值边界点导致的偶发失败；对于确定性用例可继续使用严格断言
3. 包含RVV性能统计代码：记录总运行时间并计算相对于标准实现的加速比（`Speedup (Std/RVV)`）

## 4. QEMU 中 RVV 性能不如标量代码的现象

根据官方Issue [#2137 RISC-V Vector Slowdowns](https://gitlab.com/qemu-project/qemu/-/issues/2137) 以及 reddit 社区讨论，在QEMU模拟 RISC-V 架构 时，开启 RVV 向量扩展自动向量化编译的二进制程序，运行速度远慢于非向量化版本；且这类向量优化的程序在真实 RISC-V 硬件上是提速的，仅在 QEMU 模拟环境中出现严重降速。故为正常现象。（实际在物理硬件上验证是有明显提升的）
