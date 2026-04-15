# test-rvv 说明

本目录存放面向 RISC-V RVV 的独立测试与基准：各子目录自带 `Makefile`（交叉编译 + 本机 QEMU 运行或部署到板卡），不依赖 PCL 顶层 CMake 测试目标。

## 顶层 `Makefile`：向板卡同步精简动态库

目标：`deploy_lib`

在本机收集交叉编译测试所需的 `.so`，`strip` 后通过 `rsync` 同步到远端板卡，便于在硬件上直接运行已部署的可执行文件（配合各子目录的 `board.mk`）：

1. 在 `./slim_lib` 中汇总（从本机 `WORKSPACE` 布局复制）：
  - `$(PCL_INSTALL_ROOT)/lib` 下 PCL 相关 `lib*.so*`
  - 依赖：`boost`、`lz4`、`hdf5`、`flann`、`libpng`、`zlib`、`gtest` 各 `lib` 目录
  - 工具链 sysroot 中的基础库：`libstdc++`、`libc`、`libm`、`libgcc_s`
2. 对 `slim_lib/*.so*` 执行 `riscv64-unknown-linux-gnu-strip -s`
3. `ssh` 在远端创建 `REMOTE_LIB_DIR`，再用 `rsync -avzP` 将 `slim_lib/` 内容同步到该目录
4. 删除本地临时目录 `./slim_lib`

### 使用前需修改的变量


| 变量               | 含义                                            |
| ---------------- | --------------------------------------------- |
| `REMOTE_USER`    | SSH 登录用户（默认 `root`）                           |
| `REMOTE_IP`      | 板卡 IP                                         |
| `REMOTE_LIB_DIR` | 远端存放动态库的目录（默认 `~/pcl-test/lib`）               |
| `WORKSPACE`      | 本机工作区根路径，用于推导 `PCL_INSTALL_ROOT`、`RISCV_DEPS` |


工具链前缀固定为 `riscv64-unknown-linux-gnu-gcc` / `g++`；sysroot 由 `$(CC) -print-sysroot` 自动获取。

### 命令示例

```bash
cd test-rvv
make deploy_lib
```

依赖：ssh、rsync、交叉工具链（含 `strip`），且本机已按 `Makefile` 中的路径安装好 PCL 与各 `RISCV_DEPS` 库。

### 与子目录 `board.mk` 的配合

板卡上运行测试时，`LD_LIBRARY_PATH` 必须包含上述同步目录。

## 其他说明

### QEMU 中 RVV 性能不如标量代码的现象

根据官方Issue [#2137 RISC-V Vector Slowdowns](https://gitlab.com/qemu-project/qemu/-/issues/2137) 以及 reddit 社区讨论，在QEMU模拟 RISC-V 架构 时，开启 RVV 向量扩展自动向量化编译的二进制程序，运行速度远慢于非向量化版本；且这类向量优化的程序在真实 RISC-V 硬件上是提速的，仅在 QEMU 模拟环境中出现严重降速。故为正常现象。（实际在物理硬件上验证是有明显提升的）
