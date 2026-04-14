# test-rvv/common — distances.h 测试与 Benchmark

本目录用于 `pcl/common/distances.h` 的单元测试与性能基准测试，服务于 RVV 优化前的 baseline 建立与后续对比。

## 构建与运行

- **运行 Benchmark**
  - `make run_bench_std`：不定义 `__RVV10__`，用于标量/自动向量化 baseline
  - `make run_bench_rvv`：定义 `__RVV10__`（后续 RVV 实现落地后用于对比）
  - `make run_bench_compare`：先跑 std/rvv，再输出对比表（读取两份日志）
- **目录结构**
  - 输出路径固定为 `build/$(ARCH)/...`（默认 `ARCH=riscv`）。当前仅实现 RISC-V 配置，但保留该层级以便后续恢复多架构。
- **单元测试**
  - `make run_test_std`：不定义 `__RVV10__`，只跑 Standard 分支
  - `make run_test_rvv`：定义 `__RVV10__`，只跑 RVV 分支
  - `make run_test_compare`：依次运行 std 与 rvv（分别编译/运行）

## bench_distances.cpp 覆盖的 distances.h 接口

| 接口 | 说明 |
| --- | --- |
| `squaredEuclideanDistance` | 点到点平方欧氏距离（PointXYZ / PointXY 特化） |
| `euclideanDistance` | 上述的 `sqrt` 包装 |
| `sqrPointToLineDistance` | 点到线平方距离（点/方向） |
| `getMaxSegment` | 暴力 O(n²) 求点对最远距离（全云 / indices 版本） |

Benchmark 参数：

```bash
./bench_distances [linear_points] [maxseg_points] [iterations]
```

其中 `getMaxSegment` 为 O(n²)，默认点数更小。

## 分析相关

- **向量化诊断日志**：编译时写入 `log/vec_missed_log/vec_missed_*.log`（GCC `-fopt-info-vec-missed`）。
- **过滤报告**：`make generate_vec_report` 生成 `log/filtered_distances.log` 与 `log/analyze_distances.log`。
- **反汇编**：`make dump_bench_std` / `make dump_bench_rvv` 生成 `build/asm/$(ARCH)/*.full.asm` 便于统计 RVV 指令与差异。

