# RISC-V 开发环境构建指南

为了进行 RISC-V 版本的 PCL 交叉编译与 RVV 算法验证，我们需要构建一套包含交叉编译工具链（GCC/G++）和模拟器（QEMU/Spike）的开发环境。

推荐的构建方案：

1. **本地构建方案**：适合希望直接在宿主机（Ubuntu/Arch Linux）上开发的用户。
2. **Docker 构建方案**：使用本项目提供的配置，确保环境一致性，避免污染宿主机。

如果您希望在现有的 Linux 发行版上直接安装工具链，请根据您的发行版选择对应的命令。

### 1. 依赖包安装

> **说明**：该部分内容参考 [riscv-gnu-toolchain](https://github.com/riscv-collab/riscv-gnu-toolchain) 提供的构建方式（以下内容可能有一些多余的包）

Ubuntu / Debian 系

```bash
sudo apt update
sudo apt install -y \
    git curl wget python3 python3-pip python3-tomli build-essential ninja-build pkg-config \
    autoconf automake autotools-dev libmpc-dev libmpfr-dev libgmp-dev gawk bison flex \
    texinfo gperf libtool patchutils bc zlib1g-dev libexpat-dev libslirp-dev libncurses-dev \
    libglib2.0-dev libpixman-1-dev device-tree-compiler libboost-regex-dev libboost-system-dev \
    cmake vim tmux usbutils udev zsh gdb-multiarch openocd
```

Arch Linux 系

```bash
sudo pacman -Syu
sudo pacman -S --noconfirm \
    git curl wget python python-pip base-devel ninja pkgconf \
    autoconf automake mpc mpfr gmp gawk bison flex \
    texinfo gperf libtool patchutils bc zlib expat \
    glib2 pixman dtc cmake boost libslirp ncurses \
    python-tomli vim tmux usbutils zsh gdb openocd
```

### 2.编译 RISC-V GNU Toolchain

构建支持 RVV 的交叉编译工具链。安装路径建议为 `/usr/local/riscv` 或 `/opt/riscv`

> **注意**：编译 GCC 非常耗时（取决于 CPU 核心数，可能需要 30分钟 ~ 2小时），请预留足够的磁盘空间（约 15GB）。

```bash
# 1. 设置目标路径
export RISCV=/opt/riscv
sudo mkdir -p $RISCV
sudo chown -R $USER:$USER $RISCV

# 2. 克隆源码
git clone https://github.com/riscv-collab/riscv-gnu-toolchain
cd riscv-gnu-toolchain

# 3. 检出子模块 (Submodules)
# 这一步会下载 gcc, binutils, glibc 等大量源码，请确保网络通畅
git submodule update --init --recursive

# 4. 配置构建选项
# --prefix: 安装路径
# --enable-multilib: 启用多库支持（允许编译 32位/64位 软浮点/硬浮点等不同变体）
./configure --prefix=$RISCV --enable-multilib

# 5. 编译 (构建 Linux 版本工具链，包含 glibc)
# -j$(nproc) 表示使用所有 CPU 核心并行编译
make -j$(nproc)
make linux -j$(nproc)
```

### 3. 编译 QEMU 模拟器

编译支持 RISC-V 的 QEMU，用于运行交叉编译后的程序。

```bash
# 1. 下载源码 (参考 Dockerfile 使用的版本)
wget https://download.qemu.org/qemu-9.0.0.tar.xz
tar xvJf qemu-9.0.0.tar.xz
cd qemu-9.0.0

# 2. 配置
# target-list: 仅构建 riscv64 的系统模式和用户模式，节省时间
./configure --target-list=riscv64-softmmu,riscv64-linux-user --prefix=$RISCV

# 3. 编译安装
make -j$(nproc)
make install
```

---

### 4. 编译 Spike 模拟器 (可选)

Spike 是 RISC-V 的指令集模拟器 (ISA Simulator)

```bash
# 1. 克隆源码
git clone https://github.com/riscv-software-src/riscv-isa-sim.git
cd riscv-isa-sim
mkdir build && cd build

# 2. 配置与编译
# 这里的 --prefix 可以指向我们统一的 RISCV 目录，也可以安装到 /usr/local
../configure --prefix=$RISCV
make -j$(nproc)
make install

# 3. 编译 pk (Proxy Kernel) - Spike 运行程序需要 pk 支持
cd ../..
git clone https://github.com/riscv-software-src/riscv-pk.git
cd riscv-pk
mkdir build && cd build
../configure --prefix=$RISCV --host=riscv64-unknown-linux-gnu
make -j$(nproc)
make install
```

### 5. 配置环境变量

编译完成后，需要将二进制文件路径添加到环境变量中。

在您的 `~/.bashrc` 或 `~/.zshrc` 文件末尾添加：

```bash
export RISCV=/opt/riscv
export PATH=$RISCV/bin:$PATH
```

执行 `source ~/.bashrc` 使配置生效。

### 6.验证安装

执行以下命令检查版本，确保安装成功：

```bash
# 检查 GCC 版本 (应显示 13.x, 14.x 或 15.x)
riscv64-unknown-linux-gnu-gcc --version

# 检查 QEMU 版本
qemu-riscv64 --version
```

---

## 环境验证

### 1. 检查编译器版本

```bash
# 检查 GCC 版本及支持的架构
riscv64-unknown-linux-gnu-g++ --version
# 或 (本地安装方案)
riscv64-linux-gnu-g++ --version
```

*输出示例：应当看到 gcc version x.x.x*

### 2. 验证 RVV 支持

创建一个简单的测试文件 `test_rvv.c`：

```c
#include <riscv_vector.h>
#include <stdio.h>

int main() {
    size_t vl = __riscv_vsetvl_e32m2(10);
    printf("RVV Vector Length set to: %zu\n", vl);
    return 0;
}
```

**编译与运行：**

```bash
# 编译 (开启 RVV 扩展)
riscv64-unknown-linux-gnu-gcc -march=rv64gcv -o test_rvv test_rvv.c

# 使用 QEMU 运行
qemu-riscv64 -cpu rv64,v=true ./test_rvv
```

如果输出 `RVV Vector Length set to: ...`，则说明编译环境与模拟器环境均已正确配置。