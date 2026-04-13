# RVV Load / Store 微基准（隔离测试）

本目录提供一组**尽量隔离**的 RVV 访存微基准，用来回答类似问题：

- `vluxei32` vs `vluxseg3ei32`（indexed gather）
- `vlse32` vs `vlsseg3e32`（strided load）
- `vlseg3e32` 在 `xyzxyz...` 紧密交错布局下是否更有优势（contiguous segment load）
- `vsse32` / `vssseg4e32`、`vse32` / `vsseg4e32`、`vsuxei32` / `vsuxseg4ei32`（store / scatter）

与 `test-rvv/sample_consensus/...`、`test-rvv/common/...` 等**算法级 bench** 的区别：

- **这里**：减少计算、分支与复杂临时变量，结果更贴近「load/store 形态 + 地址生成」的差异。
- **算法级**：更接近真实负载，但会受计算链、寄存器压力、mask/reduction 等影响。

---

## 目录与文件

| 文件 | 说明 |
| --- | --- |
| `bench_rvv_load_compare.cpp` | load 微基准（按 Stride → Contiguous → Indexed 顺序输出） |
| `bench_rvv_store_compare.cpp` | store 微基准（同上） |
| `Makefile` | 开发机交叉编译、QEMU 运行、`deploy_*` 到板卡 |
| `board.mk` | 板卡上运行，日志写入 `output/` |
| `output/qemu/run_load.log`、`run_store.log` | 开发机 QEMU 运行产物（`make run_*`） |
| `output/board/load_store.log` | 板卡汇总类记录 |

封装头文件（由 bench 直接 `#include`）位于 `common/include/pcl/common/`：

- `rvv_point_load.h` / `impl/rvv_point_load.hpp`
- `rvv_point_store.h` / `impl/rvv_point_store.hpp`

`bench_*_compare.cpp` 内曾保留对本目录同名 `rvv_point_*.hpp` 的注释引用，当前构建以 `<pcl/common/rvv_point_*.h>` 为准。

---

## 访存模式

### Load（`bench_rvv_load_compare`）

- **Strided AoS**：顺序遍历 AoS，`x/y/z` 固定 stride（`sizeof(PointT)`）  
  - A：`3× vlse32`  
  - B：`vlsseg3e32`
- **Contiguous（Packed xyz）**：`xyzxyz...`，stride = 12  
  - C：`3× vlse32`  
  - D：`vlseg3e32`
- **Indexed AoS gather**：随机索引间接读  
  - E：`3× vluxei32`  
  - F：`vluxseg3ei32`（不可用则回退到多条 `vluxei32`）

### Store（`bench_rvv_store_compare`）

- **Strided AoS**：stride = `sizeof(Edge4f)`  
  - A：`4× vsse32`  
  - B：`vssseg4e32`（或回退）
- **Contiguous**：四路连续数组（SoA）  
  - C：`4× vse32`  
  - D：`vsseg4e32`
- **Indexed scatter**：随机索引间接写  
  - E：`4× vsuxei32`  
  - F：`vsuxseg4ei32`（或回退）

---

## 如何运行

### 开发机（QEMU）

```bash
cd test-rvv/rvv/load_store
make run_load
make run_store
```

默认参数见 `Makefile`：`N_POINTS=262144`，`ITERS=50`，`WARMUP=5`。覆盖示例：

```bash
make run_load  N_POINTS=262144 ITERS=50 WARMUP=5
make run_store N_POINTS=262144 ITERS=50 WARMUP=5
```

日志：`output/qemu/run_load.log`、`output/qemu/run_store.log`。

### 部署到板卡

```bash
cd test-rvv/rvv/load_store
make deploy_load
make deploy_store
```

### 板卡侧运行

```bash
cd /root/pcl-test/rvv/load_store
make -f board.mk run_load
make -f board.mk run_store
```

`board.mk` 将输出 tee 到：

- `/root/pcl-test/rvv/load_store/output/run_load.log`
- `/root/pcl-test/rvv/load_store/output/run_store.log`

同样支持：

```bash
make -f board.mk run_load  N_POINTS=524288 ITERS=100 WARMUP=10
make -f board.mk run_store N_POINTS=524288 ITERS=100 WARMUP=10
```

---

## 备注

- 为避免循环被优化掉，bench 将结果规约到 `sink`（是否打印由代码决定）。
- 微基准用于观察「指令形态 / 访存模式」趋势；是否替换生产代码默认策略，需结合真实算法路径做回归与板卡数据。
