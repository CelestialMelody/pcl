# 交叉编译器
CC = riscv64-unknown-linux-gnu-gcc
CXX = riscv64-unknown-linux-gnu-g++

# ================= 核心路径配置 =================
# 1. 新安装的 PCL-RVV 路径
PCL_INSTALL_ROOT = /workspace/riscv/pcl-rvv

# 2. PCL 源码路径 (用于 pcl/test/gtest.h)
PCL_SOURCE_ROOT = /workspace/pcl

# 3. 其他依赖库路径
RISCV_DEPS = /workspace/riscv
BOOST_ROOT = $(RISCV_DEPS)/boost
EIGEN_ROOT = $(RISCV_DEPS)/eigen
FLANN_ROOT = $(RISCV_DEPS)/flann
LZ4_ROOT   = $(RISCV_DEPS)/lz4
HDF5_ROOT  = $(RISCV_DEPS)/hdf5
GTEST_ROOT = $(RISCV_DEPS)/gtest
ZLIB_ROOT  = $(RISCV_DEPS)/zlib

# 自动获取 sysroot
RISCV_SYSROOT := $(shell $(CC) -print-sysroot)

# RISCV_LIB_PATH := /workspace/riscv/pcl-rvv/lib
# RISCV_LIB_PATH := $(RISCV_LIB_PATH):/workspace/riscv/boost/lib
# RISCV_LIB_PATH := $(RISCV_LIB_PATH):/workspace/riscv/eigen/lib
# RISCV_LIB_PATH := $(RISCV_LIB_PATH):/workspace/riscv/flann/lib
# RISCV_LIB_PATH := $(RISCV_LIB_PATH):/workspace/riscv/lz4/lib
# RISCV_LIB_PATH := $(RISCV_LIB_PATH):/workspace/riscv/hdf5/lib
# RISCV_LIB_PATH := $(RISCV_LIB_PATH):/workspace/riscv/gtest/lib
# RISCV_LIB_PATH := $(RISCV_LIB_PATH):/workspace/riscv/zlib/lib

# 定义为空格分隔的列表
LIB_LIST = \
	/workspace/riscv/pcl-rvv/lib \
	/workspace/riscv/boost/lib \
	/workspace/riscv/eigen/lib \
	/workspace/riscv/flann/lib \
	/workspace/riscv/lz4/lib \
	/workspace/riscv/hdf5/lib \
	/workspace/riscv/gtest/lib \
	/workspace/riscv/zlib/lib

# 定义空格变量
empty :=
space := $(empty) $(empty)

# 将空格替换为冒号
RISCV_LIB_PATH := $(subst $(space),:,$(LIB_LIST))


# 目标
TARGET = rvv_sac_test
SRCS = test_sample_consensus_plane_models.cpp

# ================= 编译选项 (CXXFLAGS) =================
CXXFLAGS = -std=c++17 -O2 -g \
	-march=rv64gcv -mabi=lp64d \
	-D__RVV10__ \
	-DPCL_NO_PRECOMPILE \
	-I$(PCL_SOURCE_ROOT)/test/include \
	-I$(PCL_INSTALL_ROOT)/include/pcl-1.15 \
	-I$(BOOST_ROOT)/include \
	-I$(EIGEN_ROOT)/include/eigen3 \
	-I$(FLANN_ROOT)/include \
	-I$(LZ4_ROOT)/include \
	-I$(HDF5_ROOT)/include \
	-I$(GTEST_ROOT)/include

# ================= 链接选项 (LDFLAGS) =================
# PCL 和依赖库（如 FLANN, HDF5 等）大多是动态库（.so），系统中并没有对应的静态库（.a）
# 1. 去掉 -static
# 2. 添加 -Wl,-rpath-link，告诉链接器去哪里找依赖库的 .so 文件 (解决找不到 -lz, -lhdf5 等问题)
LDFLAGS = \
	-L$(PCL_INSTALL_ROOT)/lib \
	-L$(BOOST_ROOT)/lib \
	-L$(FLANN_ROOT)/lib \
	-L$(LZ4_ROOT)/lib \
	-L$(HDF5_ROOT)/lib \
	-L$(GTEST_ROOT)/lib \
	-L$(ZLIB_ROOT)/lib \
	-Wl,-rpath-link=$(PCL_INSTALL_ROOT)/lib \
	-Wl,-rpath-link=$(BOOST_ROOT)/lib \
	-Wl,-rpath-link=$(FLANN_ROOT)/lib \
	-Wl,-rpath-link=$(LZ4_ROOT)/lib \
	-Wl,-rpath-link=$(HDF5_ROOT)/lib \
	-Wl,-rpath-link=$(GTEST_ROOT)/lib \
	-Wl,-rpath-link=$(ZLIB_ROOT)/lib

# ================= 链接库 (LIBS) =================
# 注意库的顺序：SampleConsensus -> Common -> 依赖
LIBS = \
	-lpcl_sample_consensus \
	-lpcl_common \
	-lpcl_io \
	-lpcl_search \
	-lpcl_kdtree \
	-lflann_cpp \
	-lgtest -lgtest_main -lpthread \
	-lboost_filesystem -lboost_iostreams -lboost_system \
	-lm \
	-lz -lhdf5 # 可能需要显式链接 lz4 和 hdf5

# ================= 构建规则 =================
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) $(LDFLAGS) $(LIBS) -o $(TARGET)

clean:
	rm -f $(TARGET)

# 使用行内环境变量设置 (Inline Env Var)，比 export 更安全
# 注意 $$LD_LIBRARY_PATH 用来保留系统原有的路径
# @echo "Sysroot is: $(RISCV_SYSROOT)"
run: $(TARGET)
	LD_LIBRARY_PATH=$(RISCV_LIB_PATH):$$LD_LIBRARY_PATH \
	qemu-riscv64 -L $(RISCV_SYSROOT) -cpu rv64,v=true ./$(TARGET) sac_plane_test.pcd