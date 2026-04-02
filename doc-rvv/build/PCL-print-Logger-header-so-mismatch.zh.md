# PCL `print` / `Logger` 头文件与安装库不一致问题

本文记录一次在 `test-rvv/2d` 编译 `bench_2d` 时出现的链接错误及其原因与处理方式，便于在 **rebase / 升级 PCL 源码** 后遇到同类现象时快速对照。

## 现象

交叉编译 RISC-V 的 `bench_2d` 时，链接阶段报错（示例）：

```text
undefined reference to `pcl::console::Logger::getInstance()'
undefined reference to `pcl::console::Logger::print(pcl::console::LogRecord const&)'
```

## 直接原因

- `**test-rvv/2d/Makefile**`（及 `common/common/Makefile`）中，`-I$(PCL_SOURCE_ROOT)/common/include` 通常排在 **安装前缀** `$(PCL_INSTALL_ROOT)/include/pcl-1.15` **之前**，因此编译时优先使用 **当前源码树** 里的 `pcl/console/print.h`。
- 若本地 **已安装的** `libpcl_common.so` 仍是 **较早版本** 编出来的，其中 **没有** 上述 `Logger` 相关符号的实现，就会出现「新头文件声明/展开的 API + 旧动态库」的错配。

二者必须 **同源**（同一次构建或至少同一套 `print.h` / `print.cpp` 约定）。

## 与上游改动的关系

上游在引入「可重定向日志 / `Logger`」等改动时，同时修改了 `common/include/pcl/console/print.h` 与 `common/src/print.cpp`。典型提交（fork 上的镜像）：

- [CelestialMelody/pcl@da4c408](https://github.com/CelestialMelody/pcl/commit/da4c40873dfbc9f4d27c477cbf63d42cb18b249a)（说明里对应 PointCloudLibrary/pcl [#6244](https://github.com/PointCloudLibrary/pcl/pull/6244)）

**在源码里 `git pull` / `rebase` 拉到包含该改动的分支后**，若未 **按同一源码重新安装** RISC-V 版 PCL，就很容易触发本节所述链接错误。

触发链路与「是否在 `bench_2d.cpp` 里手写日志无关」：`bench_2d` → `pcl/2d/convolution.h` → `pcl/filters/filter.h` → `pcl/common/io.h` → `pcl/common/impl/io.hpp` 中的 `PCL_ERROR` 等宏，会在 **新** `print.h` 下展开为对 `Logger` 的调用。

## 处理

1. 使用 **当前 PCL 源码** 重新 **配置、编译并安装** 到 Makefile 中使用的 `PCL_INSTALL_ROOT`（例如 `riscv/pcl-rvv`），保证头文件与 `libpcl_common.so` 一致。
2. 安装完成后，对测试目标执行清理再编，例如：
  - `make -C test-rvv/2d clean_bench`（或删除对应 `build/$(ARCH)/bench_`*），避免沿用旧 `.o`。

按上述步骤处理后，**无需** 改动 `test-rvv/2d/Makefile` 的包含顺序即可正常链接。

**验证（2026-04-02）**：在重装 PCL 后的环境中执行 `make -C test-rvv/2d clean_bench && make -C test-rvv/2d build/riscv/bench_2d`，链接成功。

## 相关文档

- [RISC-V PCL Cross-Compilation Guide.zh.md](./RISC-V%20PCL%20Cross-Compilation%20Guide.zh.md) — 交叉编译与安装流程
- [RISCV Environment Setup.zh.md](./RISCV%20Environment%20Setup.zh.md) — 环境准备

