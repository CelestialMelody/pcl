# test-rvv/common — common.hpp 测试与 Benchmark

## 构建与运行

- **运行 Benchmark**  
  `make run_bench_rvv` 与 `make run_bench_std` （或者直接 `make run_bench_compare`）用于性能分析与 RVV 优化前后对比。
- **架构切换**
  - RISC-V（默认）：`make ARCH=riscv`
  - x86：`make ARCH=x86`
- **单元测试**
需 GTest 与 PCL test 头文件。`make run_test` 运行。

## bench_common.cpp 覆盖的 common.hpp 接口


| 接口                                     | 说明                                         |
| -------------------------------------- | ------------------------------------------ |
| getMeanStd                             | 对 `vector<float>` 求均值与标准差（单遍 sum + sq_sum） |
| getPointsInBox                         | AABB 内点索引（dense 点云，6 比较/点）                 |
| getMaxDistance (cloud, pivot)          | 到 pivot 最远点                                |
| getMaxDistance (cloud, indices, pivot) | 同上，带 indices                               |
| getMinMax3D (cloud, Eigen)             | AABB 最小/最大 xyz                             |
| getMinMax3D (cloud, indices, Eigen)    | 同上，带 indices                               |
| getMinMax3D (cloud, PointT)            | 同上，输出 PointT                               |
| getAngle3D                             | 两向量夹角（标量，多对调用作基线）                          |
| getCircumcircleRadius                  | 三点外接圆半径                                    |
| calculatePolygonArea                   | 多边形面积（顶点循环）                                |
| getMinMax-like                         | 直方图 min/max 循环（等价的简单循环）                    |
| computeMedian                          | 中位数（nth_element）                           |


默认点云规模约 20 万点、向量 50 万，可用参数调整：

```bash
./bench_common [cloud_size] [vector_size] [iterations]
```

## 分析相关

- **向量化日志**：编译时会生成 `logs/vec_missed_*.log`，可配合 `-fopt-info-vec-missed` 分析未向量化循环。
- **过滤 common 相关**：`make generate_clean_report` 会生成 `logs/filtered_common.log`，仅含 `common.hpp` 相关条目。
- **反汇编**：`make dump_bench` 生成 `bench_common_app.asm`，可统计 RVV 指令（ARCH=riscv）或 SSE/AVX（ARCH=x86）。

