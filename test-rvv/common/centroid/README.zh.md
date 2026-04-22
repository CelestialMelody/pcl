# test-rvv/common/centroid — centroid.hpp 测试与 Benchmark

## 构建与运行

- `make run_test_std`：不定义 `__RVV10__`，跑标量路径单测
- `make run_test_rvv`：定义 `__RVV10__`，跑 RVV 构建单测
- `make run_bench_std`：标量 baseline
- `make run_bench_rvv`：RVV 构建
- `make run_bench_compare`：自动对比 std/rvv 日志

## 向量化诊断

- 编译期日志：`log/vec_missed_log/vec_missed_*.log`（GCC `-fopt-info-vec-missed`）
- 过滤与分析：`make generate_vec_report`
- 反汇编：`make dump_bench_std` / `make dump_bench_rvv`

