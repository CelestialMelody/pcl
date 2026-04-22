# PCL RISC-V 交叉编译指南

在阅读本文之前，请先阅读 [RISCV Environment Setup.zh.md](./RISCV Environment Setup.zh.md)，完成 RISCV  开发环境的搭建。

## 1. 环境准备与变量配置说明

**后续环境配置请看这里**

为了简化后续命令，我们先定义统一的环境变量。请将 `{YOUR_RISCV_ENV_PATH}` 替换为您希望安装库的实际路径（例如 `/workspace/riscv`）。

> 若在容器内编译，一些说明：
>
> 1. 编译时若存在 "Too many open files" 问题，请确保宿主机与容器内的 ulimit 足够大且一致 ；
> 2. 编译时若存在 OOM (Out of Memory) 问题，可以减小编译并行度，如采用 make -j8

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

# 参考硬件 Milk-V Jupyter
export VLEN=256
export DEFAULT_LMUL=2
# EIGEN RVV 支持需要使用的编译选项参数, https://gitlab.com/libeigen/eigen/-/merge_requests/1687
export EIGEN_ARCH_FLAGS="-mrvv-vector-bits=zvl -DEIGEN_RISCV64_USE_RVV10 -DEIGEN_RISCV64_DEFAULT_LMUL=${DEFAULT_LMUL}"
# 架构优化标志 (RV64GCV)
export ARCH_FLAGS="-march=rv64gcv_zvl${VLEN}b -mabi=lp64d -fPIC -O3"

# 验证环境变量
echo "Install Dir: ${INSTALL_DIR}"
echo "Compiler: ${CC}"
```

## 2. 编译 zlib-1.3.1

测试时依赖 [zlib](https://www.zlib.net/) （libz.so）

``` bash
# === 1. 环境准备 ===
export RV_INSTALL_DIR=
export RISCV_TOOLCHAIN=

# 编译器及标志
export RV_CC="${RISCV_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc"
export RV_AR="${RISCV_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-ar"
export RV_RANLIB="${RISCV_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-ranlib"
# 架构优化标志 (RV64GCV)
export VLEN=256
export ARCH_FLAGS="-march=rv64gcv_zvl${VLEN}b -mabi=lp64d -fPIC -O3"

# === 2. 下载与解压 ===
wget https://www.zlib.net/zlib-1.3.1.tar.gz
tar -xvf zlib-1.3.1.tar.gz
cd zlib-1.3.1

# === 3. 配置阶段 ===
export CC="${RV_CC}"
export AR="${RV_AR}"
export RANLIB="${RV_RANLIB}"
export CFLAGS="${ARCH_FLAGS}"

# 显式指定前缀，并开启共享库编译
./configure \
    --prefix=${RV_INSTALL_DIR}/zlib \
    --shared

# === 4. 编译与安装 ===
# 再次确保 CFLAGS 传递到位，利用多核编译加速
make -j8
make install

# === 5. 验证成果 ===
echo "--------------------------------------"
echo "验证编译出的库文件格式："
file ${RV_INSTALL_DIR}/zlib/lib/libz.so.1.3.1
echo "--------------------------------------"
```

## 3.编译 LZ4-1.10.0

[LZ4](https://github.com/lz4/lz4/releases) 是 PCL 和 FLANN 的依赖库。

```bash
# === 1. 环境准备 ===
export RV_INSTALL_DIR=
export RISCV_TOOLCHAIN=

export RV_CC="${RISCV_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc"
export RV_AR="${RISCV_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-ar"
export RV_RANLIB="${RISCV_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-ranlib"
export VLEN=256
export ARCH_FLAGS="-march=rv64gcv_zvl${VLEN}b -mabi=lp64d -fPIC -O3"

# === 2. 下载与解压 ===
wget https://github.com/lz4/lz4/releases/download/v1.10.0/lz4-1.10.0.tar.gz
tar -zxvf lz4-1.10.0.tar.gz
cd lz4-1.10.0

# === 3. 编译与安装 ===
# lz4 的 Makefile 直接接受 CC, AR 等变量
make -j8 CC="$RV_CC" \
         AR="$RV_AR" \
         RANLIB="$RV_RANLIB" \
         CFLAGS="$ARCH_FLAGS"

make PREFIX="${RV_INSTALL_DIR}/lz4" install

# === 4. 验证 ===
echo "--------------------------------------"
echo "验证 lz4 库文件格式："
file ${RV_INSTALL_DIR}/lz4/lib/liblz4.so.1.10.0
echo "--------------------------------------"
```

## 4.编译 HDF5-2.1.0

FLANN 需要 [HDF5](https://github.com/HDFGroup/hdf5/releases) 支持。

```bash
# === 1. 环境准备 ===
export RV_INSTALL_DIR=
export RISCV_TOOLCHAIN=

export RV_CC="${RISCV_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc"
export RV_CXX="${RISCV_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++"
export VLEN=256
export ARCH_FLAGS="-march=rv64gcv_zvl${VLEN}b -mabi=lp64d -fPIC -O3"

# === 2. 下载与解压 ===
wget "https://github.com/HDFGroup/hdf5/releases/download/2.1.0/hdf5-2.1.0.tar.gz"
tar -zxvf hdf5-2.1.0.tar.gz
cd hdf5-2.1.0

# 建议在源码外构建
rm -rf build && mkdir build && cd build

# === 3. 配置阶段 ===
cmake .. \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
  -DCMAKE_C_COMPILER="$RV_CC" \
  -DCMAKE_CXX_COMPILER="$RV_CXX" \
  -DCMAKE_C_FLAGS="$ARCH_FLAGS" \
  -DCMAKE_CXX_FLAGS="$ARCH_FLAGS" \
  -DCMAKE_INSTALL_PREFIX="${RV_INSTALL_DIR}/hdf5" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${RV_INSTALL_DIR}/zlib" \
  -DCMAKE_EXE_LINKER_FLAGS="-L${RV_INSTALL_DIR}/zlib/lib -Wl,-rpath-link,${RV_INSTALL_DIR}/zlib/lib" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L${RV_INSTALL_DIR}/zlib/lib -Wl,-rpath-link,${RV_INSTALL_DIR}/zlib/lib" \
  -DHDF5_BUILD_CPP_LIB=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DHDF5_BUILD_FORTRAN=OFF \
  -DHDF5_BUILD_EXAMPLES=OFF \
  -DBUILD_TESTING=OFF \
  -DHDF5_ENABLE_ZLIB_SUPPORT=ON \
  -DHDF5_ENABLE_SZIP_SUPPORT=OFF \
  -DHDF5_ENABLE_SZIP_ENCODING=OFF

# === 4. 编译与安装 ===
make -j8
make install

# === 5. 验证 ===
echo "--------------------------------------"
echo "验证 HDF5 C++ 库文件格式："
file ${RV_INSTALL_DIR}/hdf5/lib/libhdf5_cpp.so.320.1.0
echo "--------------------------------------"
```

## 5.编译 FLANN

[FLANN](https://github.com/flann-lib/flann) 需要链接前面编译好的 LZ4 和 HDF5。

```bash
# === 1. 环境准备 ===
export RV_INSTALL_DIR=
export RISCV_TOOLCHAIN=

export RV_CC="${RISCV_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc"
export RV_CXX="${RISCV_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++"
export VLEN=256
export ARCH_FLAGS="-march=rv64gcv_zvl${VLEN}b -mabi=lp64d -fPIC -O3"

export ZLIB_LIB="${RV_INSTALL_DIR}/zlib/lib"
export ZLIB_INC="${RV_INSTALL_DIR}/zlib/include"

# === 2. 进入源码目录 ===
# 这里选择从源码仓库编译
git clone https://github.com/flann-lib/flann.git
cd flann

# 预处理：修改 CMakeLists.txt 以兼容 CMake
sed -i 's/cmake_minimum_required(VERSION [0-9.]*)/cmake_minimum_required(VERSION 3.5)/' CMakeLists.txt

# 清理旧的构建目录
rm -rf build && mkdir build && cd build

# === 3. CMake 配置 ===
# 提示：将路径直接加入编译器标志，确保链接器能准确找到 lz4 和 hdf5
cmake .. \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
    -DCMAKE_C_COMPILER="$RV_CC" \
    -DCMAKE_CXX_COMPILER="$RV_CXX" \
    -DCMAKE_FIND_ROOT_PATH="${RV_INSTALL_DIR}" \
    -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
    -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
    -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY \
    -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY \
    -DCMAKE_C_FLAGS="$ARCH_FLAGS -I${ZLIB_INC} -I${RV_INSTALL_DIR}/lz4/include -I${RV_INSTALL_DIR}/hdf5/include" \
    -DCMAKE_CXX_FLAGS="$ARCH_FLAGS -I${ZLIB_INC} -I${RV_INSTALL_DIR}/lz4/include -I${RV_INSTALL_DIR}/hdf5/include" \
    -DCMAKE_EXE_LINKER_FLAGS="-L${ZLIB_LIB} -L${RV_INSTALL_DIR}/lz4/lib -L${RV_INSTALL_DIR}/hdf5/lib -lz -llz4 -lhdf5" \
    -DCMAKE_SHARED_LINKER_FLAGS="-L${ZLIB_LIB} -L${RV_INSTALL_DIR}/lz4/lib -L${RV_INSTALL_DIR}/hdf5/lib -lz -llz4 -lhdf5" \
    -DCMAKE_INSTALL_PREFIX="${RV_INSTALL_DIR}/flann" \
    -DHDF5_ROOT="${RV_INSTALL_DIR}/hdf5" \
    -DBUILD_PYTHON_BINDINGS=OFF \
    -DBUILD_MATLAB_BINDINGS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_DOC=OFF

# === 4. 编译与安装 ===
make -j8
make install

# === 5. 验证与清理 ===
echo "--------------------------------------"
echo "验证 FLANN 库文件格式："
file ${RV_INSTALL_DIR}/flann/lib/libflann_cpp.so.1.9.2
echo "--------------------------------------"
```

## 6. 编译 Boost -1.88.0

[Boost](https://www.boost.org/releases/1.88.0/) 采用1.88.0版本，后续版本似乎存在 [libboost_system 丢失问题](https://github.com/boostorg/boost/issues/1071)。使用 `b2` 构建系统并指定配置文件。

```bash
# === 1. 环境准备 ===
export RV_INSTALL_DIR=
export RISCV_TOOLCHAIN=

export RV_CXX="${RISCV_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++"
export VLEN=256
export ARCH_FLAGS="-march=rv64gcv_zvl${VLEN}b -mabi=lp64d -fPIC -O3"

# === 2. 下载与解压 ===
wget https://archives.boost.io/release/1.88.0/source/boost_1_88_0.tar.gz
tar -zxvf boost_1_88_0.tar.gz
cd boost_1_88_0

# === 3. 引导构建系统 (生成 b2 引擎) ===
./bootstrap.sh --prefix=${RV_INSTALL_DIR}/boost

# === 4. 创建关键配置文件 user-config.jam ===
# 注意：using gcc 后的空格必须严格遵守语法，末尾的分号前必须有空格
cat << EOF > user-config.jam
using gcc : riscv64 : $RV_CXX : <cflags>"$ARCH_FLAGS" <cxxflags>"$ARCH_FLAGS" ;
EOF

# === 5. 编译与安装 ===
# 1. 必须通过 -s 指定 zlib 路径，否则 boost.iostreams 会跳过 zlib 支持
# 2. 显式指定 architecture=riscv 和 abi=sysv
./b2 -j8 \
    --user-config=./user-config.jam \
    --build-dir=./build \
    toolset=gcc-riscv64 \
    --prefix=${RV_INSTALL_DIR}/boost \
    architecture=riscv \
    abi=sysv \
    address-model=64 \
    target-os=linux \
    link=shared \
    threading=multi \
    runtime-link=shared \
    -sZLIB_INCLUDE="${RV_INSTALL_DIR}/zlib/include" \
    -sZLIB_LIBPATH="${RV_INSTALL_DIR}/zlib/lib" \
    -sZLIB_BINARY=z \
    install

# === 6. 验证 ===
echo "--------------------------------------"
echo "验证 Boost 库文件格式："
file ${RV_INSTALL_DIR}/boost/lib/libboost_system.so.1.88.0
echo "--------------------------------------"
```

## 7.编译 Eigen

eigen 源码仓库最新分支上是支持 RISCV RVV 的，[#2842: RISC-V RVV1.0 support](https://gitlab.com/libeigen/eigen/-/issues/2842)

```bash
# === 1. 环境准备 ===
export RV_INSTALL_DIR=
export RISCV_TOOLCHAIN=

export RV_CC="${RISCV_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc"
export RV_CXX="${RISCV_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++"
# milk-v jupyter vlen = 256
export VLEN=256
export DEFAULT_LMUL=2
# 包含 zfh (半精度浮点) 和 zvfh (向量半精度浮点)
export MARCH="rv64gcv_zvl${VLEN}b_zfh_zvfh"
export EIGEN_ARCH_FLAGS="-march=${MARCH} -mabi=lp64d -O3 -mrvv-vector-bits=zvl -DEIGEN_RISCV64_USE_RVV10 -DEIGEN_RISCV64_DEFAULT_LMUL=${DEFAULT_LMUL}"

# === 2. 下载与构建 ===
git clone https://gitlab.com/libeigen/eigen.git
cd eigen
rm -rf build && mkdir build && cd build

# 根据需求更改 CMAKE_INSTALL_PREFIX
cmake .. \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
  -DCMAKE_C_COMPILER=$RV_CC \
  -DCMAKE_CXX_COMPILER=$RV_CXX \
  -DCMAKE_INSTALL_PREFIX=${RV_INSTALL_DIR}/eigen-rvv \
  -DCMAKE_CXX_FLAGS="$EIGEN_ARCH_FLAGS" \
  -DBUILD_TESTING=OFF

make install -j8

echo "--------------------------------------"
echo "Eigen RVV (VLEN=256) 安装完成！"
echo "路径: ${RV_INSTALL_DIR}/eigen-rvv"
echo "--------------------------------------"
```

## 8.其他相关依赖

### 编译 gtest

提供的测试程序依赖 [gtest](https://github.com/google/googletest.git)，但 PCL 库编译本身不依赖

```bash
# === 1. 环境准备 ===
export RV_INSTALL_DIR=
export RISCV_TOOLCHAIN=

# 编译器路径
export RV_CC="${RISCV_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc"
export RV_CXX="${RISCV_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++"

# 标志位：googletest 建议开启 -fPIC
export VLEN=256
export ARCH_FLAGS="-march=rv64gcv_zvl${VLEN}b -mabi=lp64d -O3 -fPIC"

# === 2. 下载与源码进入 ===
git clone https://github.com/google/googletest.git
cd googletest
rm -rf build && mkdir -p build && cd build

# === 3. 配置阶段 ===
# 注意：gtest 交叉编译时通常不需要额外的前缀路径，因为它不依赖第三方库
cmake .. \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
    -DCMAKE_C_COMPILER="$RV_CC" \
    -DCMAKE_CXX_COMPILER="$RV_CXX" \
    -DCMAKE_INSTALL_PREFIX="${RV_INSTALL_DIR}/gtest" \
    -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
    -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
    -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_CXX_FLAGS="$ARCH_FLAGS" \
    -DCMAKE_C_FLAGS="$ARCH_FLAGS" \
    -Dgtest_disable_pthreads=OFF

# === 4. 编译与安装 ===
make -j8
make install

# === 5. 验证成果 ===
echo "--------------------------------------"
echo "验证 GTest 库文件格式："
file ${RV_INSTALL_DIR}/gtest/lib/libgtest.so
echo "--------------------------------------"
```

### 编译 libpng

可能存在对 libpng 的依赖

```bash
# === 1. 环境准备 ===
export RV_INSTALL_DIR=
export RISCV_TOOLCHAIN=

# 编译器及工具链路径
export RV_CC="${RISCV_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc"
export RV_CXX="${RISCV_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++"
export RV_AR="${RISCV_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-ar"
export RV_RANLIB="${RISCV_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-ranlib"

# 标志位：注入 RVV 256-bit 优化及 fPIC
export VLEN=256
export ARCH_FLAGS="-march=rv64gcv_zvl${VLEN}b -mabi=lp64d -O3 -fPIC"

# === 2. 下载与解压 ===
wget http://prdownloads.sourceforge.net/libpng/libpng-1.6.54.tar.gz
tar -xzf libpng-1.6.54.tar.gz
cd libpng-1.6.54

# === 3. 配置阶段 ===
# 注意：libpng 的 configure 需要显式指定 zlib 的头文件和库路径
# 我们通过环境变量直接传递给 configure 脚本
echo "正在配置 libpng (依赖 zlib: ${RV_INSTALL_DIR}/zlib)..."

./configure \
    --host=riscv64-unknown-linux-gnu \
    --prefix=${RV_INSTALL_DIR}/libpng \
    --enable-shared \
    --disable-static \
    CC="$RV_CC" \
    CXX="$RV_CXX" \
    AR="$RV_AR" \
    RANLIB="$RV_RANLIB" \
    CFLAGS="$ARCH_FLAGS" \
    CXXFLAGS="$ARCH_FLAGS" \
    CPPFLAGS="-I${RV_INSTALL_DIR}/zlib/include" \
    LDFLAGS="-L${RV_INSTALL_DIR}/zlib/lib -Wl,-rpath-link=${RV_INSTALL_DIR}/zlib/lib"

# === 4. 编译与安装 ===
make -j8
make install

# === 5. 验证成果 ===
echo "--------------------------------------"
echo "验证 libpng 库文件格式及依赖："
file ${RV_INSTALL_DIR}/libpng/lib/libpng16.so.16.*
ls -l ${RV_INSTALL_DIR}/libpng/lib
echo "--------------------------------------"

```



## 9.编译 PCL

链接所有依赖项。这里加上了 libpng，考虑到可能存在对其的依赖。

```bash
# === 1. 环境准备 ===
export RV_INSTALL_DIR=
export RISCV_TOOLCHAIN=
export PCL_WORKSPACE=

export RV_CC="${RISCV_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc"
export RV_CXX="${RISCV_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++"
export VLEN=256
export DEFAULT_LMUL=2

export EIGEN_ARCH_FLAGS="-mrvv-vector-bits=zvl -DEIGEN_RISCV64_USE_RVV10 -DEIGEN_RISCV64_DEFAULT_LMUL=${DEFAULT_LMUL}"
# -D__RVV10__ 进入 common/src/gaussian.cpp 等 TU 的 RVV 分支；去掉则卷积走标量路径。对比两套库时需各编一次。
export ARCH_FLAGS="-march=rv64gcv_zvl${VLEN}b -mabi=lp64d -O3 -D__RVV10__"
export EXTRA_INCLUDES="-I${RV_INSTALL_DIR}/boost/include \
                       -I${RV_INSTALL_DIR}/eigen-rvv/include/eigen3 \
                       -I${RV_INSTALL_DIR}/flann/include \
                       -I${RV_INSTALL_DIR}/lz4/include \
                       -I${RV_INSTALL_DIR}/hdf5/include \
                       -I${RV_INSTALL_DIR}/libpng/include"

# === 2. 进入源码目录 ===
cd ${PCL_WORKSPACE}
rm -rf build && mkdir build && cd build

# === 3. 配置阶段 ===
# 根据需求更改 CMAKE_INSTALL_PREFIX
cmake .. \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
  -DCMAKE_C_COMPILER=${RV_CC} \
  -DCMAKE_CXX_COMPILER=${RV_CXX} \
  -DCMAKE_INSTALL_PREFIX=${RV_INSTALL_DIR}/pcl-rvv \
  -DCMAKE_PREFIX_PATH="${RV_INSTALL_DIR}/zlib;${RV_INSTALL_DIR}/libpng;${RV_INSTALL_DIR}/lz4;${RV_INSTALL_DIR}/hdf5;${RV_INSTALL_DIR}/flann;${RV_INSTALL_DIR}/eigen-rvv;${RV_INSTALL_DIR}/boost" \
  -DCMAKE_LIBRARY_PATH="${RV_INSTALL_DIR}/lz4/lib;${RV_INSTALL_DIR}/zlib/lib;${RV_INSTALL_DIR}/hdf5/lib;${RV_INSTALL_DIR}/boost/lib;${RV_INSTALL_DIR}/flann/lib" \
  -DCMAKE_FIND_ROOT_PATH="${RV_INSTALL_DIR}/boost;${RV_INSTALL_DIR}/eigen-rvv;${RV_INSTALL_DIR}/flann;${RV_INSTALL_DIR}/lz4;${RV_INSTALL_DIR}/hdf5" \
  -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
  -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=BOTH \
  -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=BOTH \
  -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH \
  -DBOOST_ROOT=${RV_INSTALL_DIR}/boost \
  -DEigen3_DIR=${RV_INSTALL_DIR}/eigen-rvv/share/eigen3/cmake \
  -DCMAKE_CXX_FLAGS="$ARCH_FLAGS $EIGEN_ARCH_FLAGS $EXTRA_INCLUDES" \
  -DCMAKE_EXE_LINKER_FLAGS="-L${RV_INSTALL_DIR}/boost/lib -L${RV_INSTALL_DIR}/lz4/lib -L${RV_INSTALL_DIR}/hdf5/lib -L${RV_INSTALL_DIR}/libpng/lib -L${RV_INSTALL_DIR}/zlib/lib -L${RV_INSTALL_DIR}/flann/lib \
    -Wl,-rpath-link=${RV_INSTALL_DIR}/boost/lib \
    -Wl,-rpath-link=${RV_INSTALL_DIR}/lz4/lib \
    -Wl,-rpath-link=${RV_INSTALL_DIR}/hdf5/lib \
    -Wl,-rpath-link=${RV_INSTALL_DIR}/flann/lib" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L${RV_INSTALL_DIR}/boost/lib -L${RV_INSTALL_DIR}/lz4/lib -L${RV_INSTALL_DIR}/hdf5/lib -L${RV_INSTALL_DIR}/libpng/lib -L${RV_INSTALL_DIR}/zlib/lib -L${RV_INSTALL_DIR}/flann/lib \
    -Wl,-rpath-link=${RV_INSTALL_DIR}/boost/lib \
    -Wl,-rpath-link=${RV_INSTALL_DIR}/lz4/lib \
    -Wl,-rpath-link=${RV_INSTALL_DIR}/hdf5/lib \
    -Wl,-rpath-link=${RV_INSTALL_DIR}/flann/lib" \
  -DPCL_ENABLE_SSE=OFF -DPCL_ENABLE_AVX=OFF \
  -DWITH_CUDA=OFF -DWITH_OPENGL=OFF -DWITH_LIBUSB=OFF -DWITH_PCAP=OFF -DWITH_QT=OFF -DWITH_VTK=OFF \
  -DCMAKE_POLICY_DEFAULT_CMP0144=NEW

# === 4. 编译与安装 ===
make -j8
make install

# === 5. 验证成果 ===
echo "--------------------------------------"
echo "验证 PCL 核心库格式："
file ${RV_INSTALL_DIR}/pcl-rvv/lib/libpcl_common.so.1.15.1.99
echo "--------------------------------------"

```

## 10. 增量重编 pcl 模块

以 `pcl_common` 为例，仅修改 `common` 下个别源文件（例如 `common/src/gaussian.cpp`）时，可以只构建目标 `pcl_common`，并把生成的 `libpcl_common.so*` 拷贝到安装前缀的 `lib` 目录，从而避免 `make install` 依赖整棵安装树或触发不必要的工作。

首次在本机建立 `build` 目录时仍需执行与上一节一致的 `cmake`；之后若未改 CMake 选项，可在 `build` 目录内直接 `make pcl_common`，无需每次重新配置。

下面脚本与上一节使用相同的交叉工具链、`__RVV10__`、Eigen RVV 与依赖路径变量；请将 `RV_INSTALL_DIR`、`RISCV_TOOLCHAIN` 以及 `cd` 到 PCL 源码的路径改成你的环境。

若 `file` 检查的 `.so` 版本号与当前 PCL 不一致，请按 `ls ${RV_INSTALL_DIR}/pcl-rvv/lib/libpcl_common.so*` 实际名称修改。

```bash
#!/usr/bin/env bash
# 增量重编 PCL

set -euo pipefail

# === 1. 环境准备 ===
export RV_INSTALL_DIR=
export RISCV_TOOLCHAIN=
export PCL_WORKSPACE=

export RV_CC="${RISCV_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc"
export RV_CXX="${RISCV_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++"
export VLEN=256
export DEFAULT_LMUL=2

export EIGEN_ARCH_FLAGS="-mrvv-vector-bits=zvl -DEIGEN_RISCV64_USE_RVV10 -DEIGEN_RISCV64_DEFAULT_LMUL=${DEFAULT_LMUL}"
# 与第 9 节相同：保留 -D__RVV10__ 则 gaussian.cpp 卷积进 RVV；去掉后重装/拷贝的 lib 内为标量卷积。改 ARCH_FLAGS 后须重新 cmake 或清缓存再配置，再 make pcl_common。
export ARCH_FLAGS="-march=rv64gcv_zvl${VLEN}b -mabi=lp64d -O3 -D__RVV10__"
# 标量卷积库示例（仅此注释，使用时替换上一行）： export ARCH_FLAGS="-march=rv64gcv_zvl${VLEN}b -mabi=lp64d -O3"
export EXTRA_INCLUDES="-I${RV_INSTALL_DIR}/boost/include \
                       -I${RV_INSTALL_DIR}/eigen-rvv/include/eigen3 \
                       -I${RV_INSTALL_DIR}/flann/include \
                       -I${RV_INSTALL_DIR}/lz4/include \
                       -I${RV_INSTALL_DIR}/hdf5/include \
                       -I${RV_INSTALL_DIR}/libpng/include"

# === 2. 构建目录 ===
cd ${PCL_WORKSPACE}
mkdir -p build && cd build

# === 3. 配置 ===
cmake .. \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
  -DCMAKE_C_COMPILER="${RV_CC}" \
  -DCMAKE_CXX_COMPILER="${RV_CXX}" \
  -DCMAKE_INSTALL_PREFIX="${RV_INSTALL_DIR}/pcl-rvv" \
  -DCMAKE_PREFIX_PATH="${RV_INSTALL_DIR}/zlib;${RV_INSTALL_DIR}/libpng;${RV_INSTALL_DIR}/lz4;${RV_INSTALL_DIR}/hdf5;${RV_INSTALL_DIR}/flann;${RV_INSTALL_DIR}/eigen-rvv;${RV_INSTALL_DIR}/boost" \
  -DCMAKE_LIBRARY_PATH="${RV_INSTALL_DIR}/lz4/lib;${RV_INSTALL_DIR}/zlib/lib;${RV_INSTALL_DIR}/hdf5/lib;${RV_INSTALL_DIR}/boost/lib;${RV_INSTALL_DIR}/flann/lib" \
  -DCMAKE_FIND_ROOT_PATH="${RV_INSTALL_DIR}/boost;${RV_INSTALL_DIR}/eigen-rvv;${RV_INSTALL_DIR}/flann;${RV_INSTALL_DIR}/lz4;${RV_INSTALL_DIR}/hdf5" \
  -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
  -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=BOTH \
  -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=BOTH \
  -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH \
  -DBOOST_ROOT="${RV_INSTALL_DIR}/boost" \
  -DEigen3_DIR="${RV_INSTALL_DIR}/eigen-rvv/share/eigen3/cmake" \
  -DCMAKE_CXX_FLAGS="${ARCH_FLAGS} ${EIGEN_ARCH_FLAGS} ${EXTRA_INCLUDES}" \
  -DCMAKE_EXE_LINKER_FLAGS="-L${RV_INSTALL_DIR}/boost/lib -L${RV_INSTALL_DIR}/lz4/lib -L${RV_INSTALL_DIR}/hdf5/lib -L${RV_INSTALL_DIR}/libpng/lib -L${RV_INSTALL_DIR}/zlib/lib -L${RV_INSTALL_DIR}/flann/lib \
    -Wl,-rpath-link=${RV_INSTALL_DIR}/boost/lib \
    -Wl,-rpath-link=${RV_INSTALL_DIR}/lz4/lib \
    -Wl,-rpath-link=${RV_INSTALL_DIR}/hdf5/lib \
    -Wl,-rpath-link=${RV_INSTALL_DIR}/flann/lib" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L${RV_INSTALL_DIR}/boost/lib -L${RV_INSTALL_DIR}/lz4/lib -L${RV_INSTALL_DIR}/hdf5/lib -L${RV_INSTALL_DIR}/libpng/lib -L${RV_INSTALL_DIR}/zlib/lib -L${RV_INSTALL_DIR}/flann/lib \
    -Wl,-rpath-link=${RV_INSTALL_DIR}/boost/lib \
    -Wl,-rpath-link=${RV_INSTALL_DIR}/lz4/lib \
    -Wl,-rpath-link=${RV_INSTALL_DIR}/hdf5/lib \
    -Wl,-rpath-link=${RV_INSTALL_DIR}/flann/lib" \
  -DPCL_ENABLE_SSE=OFF -DPCL_ENABLE_AVX=OFF \
  -DWITH_CUDA=OFF -DWITH_OPENGL=OFF -DWITH_LIBUSB=OFF -DWITH_PCAP=OFF -DWITH_QT=OFF -DWITH_VTK=OFF \
  -DCMAKE_POLICY_DEFAULT_CMP0144=NEW

# === 4. 编译与安装 ===
make -j16 pcl_common
# 勿用 make install：会依赖整棵安装树并触发全量编译
install -d "${RV_INSTALL_DIR}/pcl-rvv/lib"
cp -a lib/libpcl_common.so* "${RV_INSTALL_DIR}/pcl-rvv/lib/"
echo "[INFO] 已仅更新 ${RV_INSTALL_DIR}/pcl-rvv/lib/libpcl_common.so*"

# === 5. 验证 ===
echo "--------------------------------------"
echo "验证 PCL 核心库格式："
file "${RV_INSTALL_DIR}/pcl-rvv/lib/libpcl_common.so.1.15.1.99"
echo "--------------------------------------"
```

