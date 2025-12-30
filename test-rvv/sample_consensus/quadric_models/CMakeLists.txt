cmake_minimum_required(VERSION 3.10)

# ==============================================================================
# 1. 交叉编译工具链配置 (必须在 project() 之前)
# ==============================================================================
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# 指定交叉编译器 (确保这些命令在你的 PATH 中，或者写绝对路径)
set(CMAKE_C_COMPILER "riscv64-unknown-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "riscv64-unknown-linux-gnu-g++")

# 指定查找规则：
# 1. 不去宿主机(/usr/lib等)找库，只在指定位置找
# 2. 程序(如 make, bash)依然使用宿主机的
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# ==============================================================================
# 2. 项目定义
# ==============================================================================
project(rvv_sac_quadric_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==============================================================================
# 3. 路径变量配置 (根据你的环境修改)
# ==============================================================================
set(RISCV_ROOT       "/workspace/riscv")
set(PCL_INSTALL_PATH "${RISCV_ROOT}/pcl-rvv")
set(PCL_SOURCE_PATH  "/workspace/pcl")

# 设置 CMAKE_PREFIX_PATH，让 find_package 能自动找到这些路径下的库
list(APPEND CMAKE_PREFIX_PATH
    "${PCL_INSTALL_PATH}"
    "${RISCV_ROOT}/boost"
    "${RISCV_ROOT}/eigen"
    "${RISCV_ROOT}/flann"
    "${RISCV_ROOT}/lz4"
    "${RISCV_ROOT}/hdf5"
    "${RISCV_ROOT}/zlib"
    "${RISCV_ROOT}/gtest"
)

# ==============================================================================
# 4. 查找依赖
# ==============================================================================
# 强制使用新编译的 PCL
set(PCL_DIR "${PCL_INSTALL_PATH}/share/pcl-1.15")
find_package(PCL 1.15 REQUIRED COMPONENTS common sample_consensus io search kdtree)

find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)

# ==============================================================================
# 5. 编译目标设置
# ==============================================================================
add_executable(rvv_sac_quadric_test test_sample_consensus_quadric_models.cpp)

# --- 头文件路径 ---
target_include_directories(rvv_sac_quadric_test PRIVATE
    # 解决 pcl/test/gtest.h 找不到的问题
    "${PCL_SOURCE_PATH}/test/include"

    # 依赖库头文件
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${GTEST_INCLUDE_DIRS}

    # 手动补充可能找不到的头文件 (LZ4, HDF5, FLANN)
    "${RISCV_ROOT}/lz4/include"
    "${RISCV_ROOT}/hdf5/include"
    "${RISCV_ROOT}/flann/include"
)

# --- 编译选项 (RVV关键配置) ---
target_compile_options(rvv_sac_quadric_test PRIVATE
    -march=rv64gcv
    -mabi=lp64d
    -D__RVV10__
    -O3
    -DPCL_NO_PRECOMPILE
)

# --- 链接库路径 ---
# 显式告诉链接器去哪里找 .so 文件 (防止找不到 -lz, -lhdf5, -lflann_cpp)
target_link_directories(rvv_sac_quadric_test PRIVATE
    "${RISCV_ROOT}/zlib/lib"
    "${RISCV_ROOT}/lz4/lib"
    "${RISCV_ROOT}/hdf5/lib"
    "${RISCV_ROOT}/flann/lib"
    "${RISCV_ROOT}/boost/lib"
)

# --- 链接库 ---
target_link_libraries(rvv_sac_quadric_test PRIVATE
    ${PCL_LIBRARIES}
    GTest::GTest
    GTest::Main
    pthread
    m
    z     # zlib
    hdf5  # hdf5
)

# ==============================================================================
# 6. QEMU 运行目标 (make run)
# ==============================================================================
# 自动获取 Sysroot
execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} -print-sysroot
    OUTPUT_VARIABLE SYSROOT_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# 拼接 LD_LIBRARY_PATH (包含所有依赖库的 lib 目录)
set(RUN_LIB_PATHS
    "${PCL_INSTALL_PATH}/lib"
    "${RISCV_ROOT}/boost/lib"
    "${RISCV_ROOT}/eigen/lib"
    "${RISCV_ROOT}/flann/lib"
    "${RISCV_ROOT}/lz4/lib"
    "${RISCV_ROOT}/hdf5/lib"
    "${RISCV_ROOT}/gtest/lib"
    "${RISCV_ROOT}/zlib/lib"
)
# 将列表转换为冒号分隔的字符串
string(REPLACE ";" ":" LIB_PATH_STR "${RUN_LIB_PATHS}")

add_custom_target(run
    COMMAND ${CMAKE_COMMAND} -E echo "=========================================="
    COMMAND ${CMAKE_COMMAND} -E echo "Running RVV Test on QEMU..."
    COMMAND ${CMAKE_COMMAND} -E echo "Sysroot: ${SYSROOT_PATH}"
    COMMAND ${CMAKE_COMMAND} -E echo "=========================================="
    # 设置环境变量并启动 QEMU
    COMMAND ${CMAKE_COMMAND} -E env "LD_LIBRARY_PATH=${LIB_PATH_STR}:$ENV{LD_LIBRARY_PATH}"
            qemu-riscv64 -L "${SYSROOT_PATH}" -cpu rv64,v=true ./rvv_sac_quadric_test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS rvv_sac_quadric_test
    USES_TERMINAL
)
