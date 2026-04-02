# 2d 测试与基准

本目录提供 PCL `2d` 模块相关的 **功能测试**、**基准测试**，并支持：

- **本机 `Makefile`**：交叉编译（RISC-V）后在 **QEMU** 运行；或在 x86 主机上原生运行（`ARCH=x86`）。
- **板卡 `board.mk`**：在板卡上直接运行已部署的二进制，并保存日志、生成对比表（不负责编译）。

板卡运行前建议先在 `test-rvv/` 执行顶层库部署：见 `../README.zh.md` 中的 `make deploy_lib`。

---

## 1. 本机：`Makefile`（交叉编译 + QEMU / x86）

### 1.1 目标（riscv 默认）

在本机（开发机）执行：

```bash
cd test-rvv/2d
make run_test
make run_bench
```

- **`run_test`**：构建并运行 `build/$(ARCH)/test_2d_app`，并将输出 `tee` 到 `output/run_test.log`。运行时会把 `pcd/` 中一组 PCD 按**固定位置参数**传入（顺序不能变）。
- **`run_bench`**：构建并运行 `build/$(ARCH)/$(TARGET_BENCH)`；默认 `TARGET_BENCH=bench_2d_app`，并在运行前自动执行 `generate_vec_report`（见下文）。

### 1.2 Std vs RVV 两套 bench（RISC-V）

该目录用 `USE_PCL_RVV10` 控制是否启用 PCL 手写 RVV 内核：

- **Std**：`USE_PCL_RVV10=0`（关闭 `-D__RVV10__`，用于对比“不开 PCL RVV 内核”的版本）
- **RVV**：`USE_PCL_RVV10=1`（开启 `-D__RVV10__`，使用 PCL RVV10 代码路径）

一键分别跑两套（并将日志保存到 `output/`）：

```bash
cd test-rvv/2d
make run_bench_std
make run_bench_rvv
make analyze_bench_compare
```

- **`run_bench_std`**：子 make 调用 `run_bench`，固定 `USE_PCL_RVV10=0`，且 `TARGET_BENCH=$(TARGET_BENCH_STD)`（默认 `bench_2d_app_std`），输出 `tee` 到 `output/run_bench_std.log`。
- **`run_bench_rvv`**：子 make 调用 `run_bench`，固定 `USE_PCL_RVV10=1`，输出 `tee` 到 `output/run_bench_rvv.log`。
- **`analyze_bench_compare`**：解析两份日志并打印 Std vs RVV 对比表。默认优先使用本目录 `.venv/bin/python`，否则回退 `python3`。

提示：如果你把板卡日志复制回本机，也可以通过覆盖日志路径做对比，例如：

```bash
make BENCH_STD_OUTPUT_FILE=output/board/run_bench_std.log \
     BENCH_RVV_OUTPUT_FILE=output/board/run_bench_rvv.log \
     analyze_bench_compare
```

### 1.3 向量化“missed”日志与分析报告

`Makefile` 默认给编译器加了 `-fopt-info-vec-missed=...`，每次构建会产生一份“向量化未命中”日志到：

- `log/vec_missed_log/vec_missed_<timestamp>.log`

`generate_vec_report` 会：

1. 选取最新一份 vec-missed 日志并链接为 `log/latest_vec_missed.log`
2. 依据 `FOCUS_DIR`（默认 `pcl/2d/include/pcl/2d/impl/edge.hpp`）过滤关心的条目，生成 `log/filtered_report.log`
3. 调用 `script/analyze_vec_log.py` 生成汇总分析 `log/analyzed_vec_report.log`，并把分类输出写入 `log/vec_logs/`

### 1.4 反汇编与 SIMD 指令统计

```bash
cd test-rvv/2d
make dump_bench
```

会生成：

- `build/asm/$(ARCH)/$(TARGET_BENCH).asm`（普通反汇编）
- `build/asm/$(ARCH)/$(TARGET_BENCH).full.asm`（带符号与源码混排）

并在最后基于 `VEC_REGEX_STR` 打印“是否出现向量指令/寄存器”的统计：

- RISC-V：匹配 RVV 指令（例如 `vsetvli`、`vadd.vv` 等）
- x86：匹配 `%xmm/%ymm/%zmm` 等寄存器

### 1.5 x86 原生运行（可选）

如果要在 x86 上跑（不走 QEMU，不用交叉工具链）：

```bash
cd test-rvv/2d
make ARCH=x86 run_bench
```

---

## 2. 板卡：`board.mk`

`board.mk` 约定的部署路径与 `Makefile` 中的 `REMOTE_*` 一致，核心变量：

- **`REMOTE_DIR`**：板卡上二进制所在目录（默认 `/root/pcl-test/2d`）
- **`REMOTE_LIB_DIR`**：板卡上动态库目录（默认 `/root/pcl-test/lib`，建议与 `test-rvv/Makefile` 的 `REMOTE_LIB_DIR` 一致）
- **`REMOTE_PCD_DIR`**：板卡 PCD 目录（默认 `$(REMOTE_DIR)/pcd`）
- **`SCRIPT_DIR`**：板卡脚本目录（默认 `$(REMOTE_DIR)/script`，需包含 `analyze_bench_compare.py`）

### 2.1 板卡端常用命令

在板卡上（含 `board.mk` 内容的 `Makefile`）：

```bash
# 如果是 Makefile: make run_test
make -f board.mk run_test
make -f board.mk run_bench_compare
```

- **`run_bench_std` / `run_bench_rvv`**：分别 `tee` 到 `$(REMOTE_DIR)/output/run_bench_{std,rvv}.log`
- **`analyze_bench_compare`**：读取两份日志，输出对比表；可用 `BENCH_COMPARE_SAVE=...` 另存结果

### 2.2 部署提示

开发机侧 `Makefile` 提供这些部署目标（都会 `strip` 后 `rsync` 到板卡 `REMOTE_DIR`）：

- `deploy_files`：同步 PCD 与 `script/analyze_bench_compare.py` 到板卡
- `deploy_bench_std`：部署 `bench_2d_app_std`
- `deploy_bench_rvv`：部署 `bench_2d_app_rvv`
- `deploy_test`：部署 `test_2d_app`

推荐顺序：

```bash
# 开发机
cd test-rvv/2d
make deploy_files
make deploy_bench_std
make deploy_bench_rvv

# 板卡
cd /root/pcl-test/2d
make -f board.mk run_bench_compare
```

