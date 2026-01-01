# PCL RISC-V 交叉编译指南

在阅读本文之前，请先阅读 RISCV Environment Setup.md，完成 RISCV  开发环境的搭建。本教程假设您使用 Docker 容器，且 RISC-V 工具链安装在 `/usr/local/riscv` 。

> 说明：1. 对于不使用 Docker 环境也能参考；2. 你可以在交叉编译时，更改为你想要安装的路径

## 1. 环境准备与变量配置（后续环境配置请看这里）

为了简化后续命令，我们先定义统一的环境变量。请将 `{YOUR_RISCV_ENV_PATH}` 替换为您希望安装库的实际路径（例如 `/workspace/riscv`）。

> 说明：1. 编译时若存在 "Too many open files" 问题，请确保宿主机与容器内的 ulimit 足够大且一致 ；2. 编译时若存在 OOM (Out of Memory) 问题，可以减小编译并行度，如采用 make -j8

```bash
# === 环境变量配置（可以选择不配置，后续文件中相关变量直接替换为相应的值即可） ===
# 定义安装目录 (请根据需要修改)
export INSTALL_DIR="{YOUR_RISCV_ENV_PATH}"
# 定义工具链路径（请根据需要修改)
export RISCV_TOOLCHAIN="/usr/local/riscv"

# 创建目录
mkdir -p ${INSTALL_DIR}

# 编译器标志
export CC="${RISCV_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc"
export CXX="${RISCV_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++"
export AR="${RISCV_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-ar"
export RANLIB="${RISCV_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-ranlib"

# 架构优化标志 (RV64GCV)
export ARCH_FLAGS="-march=rv64gcv -mabi=lp64d"

# 验证环境变量
echo "Install Dir: ${INSTALL_DIR}"
echo "Compiler: ${CC}"
```

## 2. 编译 zlib

测试时依赖 [zlib](https://www.zlib.net/) （libz.so）

``` bash
# 下载源码
wget https://www.zlib.net/zlib-1.3.1.tar.gz
tar -xvf zlib-1.3.1.tar.gz
cd zlib-1.3.1

# 配置安装路径
./configure --prefix=${INSTALL_DIR}/zlib

# 编译并安装
make -j$(nproc) && make install
```

## 3.编译 LZ4 (压缩库)

[LZ4](https://github.com/lz4/lz4/releases) 是 PCL 和 FLANN 的依赖库。

```bash
# 下载并解压
wget https://github.com/lz4/lz4/releases/download/v1.10.0/lz4-1.10.0.tar.gz
tar -zxvf lz4-1.10.0.tar.gz
cd lz4-1.10.0

# 编译安装
make CC=$CC AR=$AR RANLIB=$RANLIB \
     CFLAGS="$ARCH_FLAGS" \
     PREFIX=${INSTALL_DIR}/lz4 \
     install

# 验证
ls ${INSTALL_DIR}/lz4/lib
```

## 4.编译 HDF5

FLANN 需要 [HDF5](https://github.com/HDFGroup/hdf5/releases) 支持。

```bash
wget https://github.com/HDFGroup/hdf5/releases/download/hdf5_1.14.6/hdf5-1.14.6.tar.gz
tar -zxvf hdf5-1.14.6.tar.gz
cd hdf5-1.14.6

mkdir -p build && cd build

# 配置 (启用动态库)
../configure \
    --host=riscv64-unknown-linux-gnu \
    --prefix=${INSTALL_DIR}/hdf5 \
    --enable-shared \
    --disable-static \
    --with-zlib=${INSTALL_DIR}/zlib \
    CC=$CC CXX=$CXX \
    CFLAGS="$ARCH_FLAGS" CXXFLAGS="$ARCH_FLAGS"

# 编译安装 (使用多核编译)
make -j$(nproc) && make install
```

## 5.编译 FLANN (近似最近邻搜索)

[FLANN](https://github.com/flann-lib/flann) 需要链接前面编译好的 LZ4 和 HDF5。

```bash
# 这里选择从源码仓库编译
git clone https://github.com/flann-lib/flann.git
cd flann
mkdir build && cd build

# 设置 PKG_CONFIG 路径以防万一
export PKG_CONFIG_PATH=${INSTALL_DIR}/lz4/lib/pkgconfig:$PKG_CONFIG_PATH

# CMake 配置
cmake .. \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
    -DCMAKE_C_COMPILER=$CC \
    -DCMAKE_CXX_COMPILER=$CXX \
    -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
    -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
    -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY \
    -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY \
    -DCMAKE_C_FLAGS="-I${INSTALL_DIR}/lz4/include -I${INSTALL_DIR}/hdf5/include $ARCH_FLAGS" \
    -DCMAKE_CXX_FLAGS="-I${INSTALL_DIR}/lz4/include -I${INSTALL_DIR}/hdf5/include $ARCH_FLAGS" \
    -DCMAKE_EXE_LINKER_FLAGS="-L${INSTALL_DIR}/lz4/lib -L${INSTALL_DIR}/hdf5/lib -llz4 -lhdf5" \
    -DCMAKE_SHARED_LINKER_FLAGS="-L${INSTALL_DIR}/lz4/lib -L${INSTALL_DIR}/hdf5/lib -llz4 -lhdf5" \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/flann \
    -DHDF5_ROOT=${INSTALL_DIR}/hdf5 \
    -DBUILD_PYTHON_BINDINGS=OFF \
    -DBUILD_MATLAB_BINDINGS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_TESTS=OFF

make -j$(nproc) && make install
```

## 6. 编译 Boost (1.88.0)

[Boost](https://www.boost.org/releases/1.88.0/) 采用1.88.0版本，后续版本似乎存在 [libboost_system 丢失问题](https://github.com/boostorg/boost/issues/1071)。使用 `b2` 构建系统并指定配置文件。

```bash
# 下载并解压 Boost 1.88.0
wget https://archives.boost.io/release/1.88.0/source/boost_1_88_0.tar.gz
tar -zxvf boost_1_88_0.tar.gz
cd boost_1_88_0

# 引导构建系统
./bootstrap.sh

# 创建交叉编译配置文件
cp tools/build/example/user-config.jam .
# 找到 "using gcc" 这一行进行修改，若不存在则添加（这里 $CXX 可能需要替换为具体路径）
echo "using gcc : riscv64 : $CXX ;" > user-config.jam

# 编译安装
./b2 -j$(nproc) \
    toolset=gcc-riscv64 \
    --user-config=./user-config.jam \
    --build-dir=./build-riscv \
    --prefix=${INSTALL_DIR}/boost \
    cxxflags="$ARCH_FLAGS" \
    address-model=64 \
    architecture=riscv \
    target-os=linux \
    abi=sysv \
    link=shared \
    install
```

## 7.编译 Eigen (3.3.9)

目前，Eigen 5.x 版本不兼容，需使用 3.3.9。(PS：eigen 源码仓库最新分支上是支持 RISCV RVV 的)

```bash
wget https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.gz
tar -zxvf eigen-3.3.9.tar.gz
cd eigen-3.3.9
mkdir build && cd build

cmake .. \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
  -DCMAKE_C_COMPILER=$CC \
  -DCMAKE_CXX_COMPILER=$CXX \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/eigen \
  -DCMAKE_CXX_FLAGS="$ARCH_FLAGS"

make install -j$(nproc)
```

## 8.编译 PCL (最终目标)

链接所有依赖项。

```bash
git clone https://github.com/PointCloudLibrary/pcl.git
cd pcl
mkdir build && cd build

# 1. 导出必要的环境变量
export CMAKE_PREFIX_PATH="${INSTALL_DIR}/zlib;${INSTALL_DIR}/lz4;${INSTALL_DIR}/hdf5;${INSTALL_DIR}/flann;${INSTALL_DIR}/eigen;${INSTALL_DIR}/boost"

# -----------------------------------------------------------------
# [选项 A] 若未使用本仓库提供的 RVV 函数 (标准编译)
# -----------------------------------------------------------------
cmake .. \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
  -DCMAKE_C_COMPILER=$CC \
  -DCMAKE_CXX_COMPILER=$CXX \
  -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
  -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
  -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY \
  -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/pcl \
  -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" \
  -DCMAKE_CXX_FLAGS="$ARCH_FLAGS -I${INSTALL_DIR}/lz4/include -I${INSTALL_DIR}/hdf5/include" \
  -DCMAKE_EXE_LINKER_FLAGS="-L${INSTALL_DIR}/lz4/lib -L${INSTALL_DIR}/hdf5/lib -Wl,-rpath-link=${INSTALL_DIR}/hdf5/lib" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L${INSTALL_DIR}/lz4/lib -L${INSTALL_DIR}/hdf5/lib -Wl,-rpath-link=${INSTALL_DIR}/hdf5/lib" \
  -DWITH_CUDA=OFF -DWITH_OPENGL=OFF -DWITH_LIBUSB=OFF -DWITH_PCAP=OFF -DWITH_QT=OFF -DWITH_VTK=OFF \
  -DCMAKE_POLICY_DEFAULT_CMP0144=NEW

# -----------------------------------------------------------------
# [选项 B] 若使用本仓库 RVV 函数
# -----------------------------------------------------------------
cmake .. \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
  -DCMAKE_C_COMPILER=$CC \
  -DCMAKE_CXX_COMPILER=$CXX \
  -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
  -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
  -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY \
  -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/pcl-rvv \
  -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" \
  -DCMAKE_CXX_FLAGS="$ARCH_FLAGS -D__RVV10__ -O3 -I${INSTALL_DIR}/lz4/include -I${INSTALL_DIR}/hdf5/include" \
  -DCMAKE_EXE_LINKER_FLAGS="-L${INSTALL_DIR}/lz4/lib -L${INSTALL_DIR}/hdf5/lib -Wl,-rpath-link=${INSTALL_DIR}/hdf5/lib" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L${INSTALL_DIR}/lz4/lib -L${INSTALL_DIR}/hdf5/lib -Wl,-rpath-link=${INSTALL_DIR}/hdf5/lib" \
  -DPCL_ENABLE_SSE=OFF -DPCL_ENABLE_AVX=OFF \
  -DWITH_CUDA=OFF -DWITH_OPENGL=OFF -DWITH_LIBUSB=OFF -DWITH_PCAP=OFF -DWITH_QT=OFF -DWITH_VTK=OFF \
  -DCMAKE_POLICY_DEFAULT_CMP0144=NEW

# 编译与安装
make -j$(nproc) && make install
```

## 9.其他相关依赖

提供的测试程序依赖 [gtest](https://github.com/google/googletest.git)，但 PCL 库编译本身不依赖

```bash
git clone https://github.com/google/googletest.git
cd googletest && mkdir -p build && cd build

# CMake 配置与编译
cmake .. \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
    -DCMAKE_C_COMPILER=$CC \
    -DCMAKE_CXX_COMPILER=$CXX \
    -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
    -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
    -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/gtest \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_CXX_FLAGS="$ARCH_FLAGS -fPIC" \
    -DCMAKE_C_FLAGS="$ARCH_FLAGS -fPIC"

# 编译并安装
make -j$(nproc) && make install
```
