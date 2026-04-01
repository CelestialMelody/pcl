ARCH ?= riscv

# =============================================================================
# Sample Consensus / Quadric Models (RVV)
# =============================================================================

ifeq ($(ARCH), riscv)
CC  = riscv64-unknown-linux-gnu-gcc
CXX = riscv64-unknown-linux-gnu-g++

# =============================================================================
#  Paths
# =============================================================================
WORKSPACE               = /home/zoomin/codes/RISCV/workspace
PCL_INSTALL_ROOT = $(WORKSPACE)/riscv/pcl-rvv
PCL_SOURCE_ROOT  = $(WORKSPACE)/pcl
RISCV_DEPS       = $(WORKSPACE)/riscv
BOOST_ROOT  = $(RISCV_DEPS)/boost
EIGEN_ROOT  = $(RISCV_DEPS)/eigen-rvv
FLANN_ROOT  = $(RISCV_DEPS)/flann
LZ4_ROOT    = $(RISCV_DEPS)/lz4
HDF5_ROOT   = $(RISCV_DEPS)/hdf5
GTEST_ROOT  = $(RISCV_DEPS)/gtest
ZLIB_ROOT   = $(RISCV_DEPS)/zlib
LIBPNG_ROOT = $(RISCV_DEPS)/libpng

# 自动获取 sysroot
RISCV_SYSROOT := $(shell $(CC) -print-sysroot)

LIB_LIST = \
	$(PCL_INSTALL_ROOT)/lib \
	$(BOOST_ROOT)/lib \
	$(FLANN_ROOT)/lib \
	$(LZ4_ROOT)/lib \
	$(HDF5_ROOT)/lib \
	$(GTEST_ROOT)/lib \
	$(ZLIB_ROOT)/lib \
	$(LIBPNG_ROOT)/lib

empty :=
space := $(empty) $(empty)
RISCV_LIB_PATH := $(subst $(space),:,$(LIB_LIST))

endif

# =============================================================================
#  Targets
# =============================================================================
TARGET = rvv_sac_quadric_test
SRCS = test_sample_consensus_quadric_models.cpp
BUILD_DIR = build
TARGET_BIN = $(BUILD_DIR)/$(TARGET)

EIGEN_RVV_FLAGS ?= -DEIGEN_RISCV64_USE_RVV10 -DEIGEN_RISCV64_DEFAULT_LMUL=2 -march=rv64gcv_zvl256b -mrvv-vector-bits=zvl

# =============================================================================
#  Include Strategy (use patched headers)
# =============================================================================
# Prefer headers from the PCL source tree (no reinstall needed)
# USE_PCL_SOURCE_HEADERS=1  -> include $(PCL_SOURCE_ROOT)/**/include before installed headers
# USE_PCL_SOURCE_HEADERS=0  -> use only installed headers under $(PCL_INSTALL_ROOT)/include
USE_PCL_SOURCE_HEADERS ?= 1
# 这些文件都是我们修改过的，所以需要从源码中获取
PCL_SOURCE_INCLUDES = \
	-I$(PCL_SOURCE_ROOT)/common/include \
	-I$(PCL_SOURCE_ROOT)/test/include \
	-I$(PCL_SOURCE_ROOT)/sample_consensus/include

ifeq ($(USE_PCL_SOURCE_HEADERS),1)
PCL_INCLUDE = $(PCL_SOURCE_INCLUDES) -I$(PCL_INSTALL_ROOT)/include/pcl-1.15
else
PCL_INCLUDE = -I$(PCL_INSTALL_ROOT)/include/pcl-1.15
endif

# =============================================================================
#  Build Flags
# =============================================================================
# ================= 编译选项 (CXXFLAGS) =================
# Optional: add -Wno-deprecated-declarations to hide Eigen jacobiSvd deprecation
# from PCL sac_model_torus.hpp until PCL headers are reinstalled.
CXXFLAGS = -std=c++17 -O3 -g \
	$(EIGEN_RVV_FLAGS) \
	-mabi=lp64d \
	-D__RVV10__ \
	-DPCL_NO_PRECOMPILE \
	$(PCL_INCLUDE) \
	-I$(BOOST_ROOT)/include \
	-I$(EIGEN_ROOT)/include/eigen3 \
	-I$(FLANN_ROOT)/include \
	-I$(LZ4_ROOT)/include \
	-I$(HDF5_ROOT)/include \
	-I$(GTEST_ROOT)/include \
	-I$(ZLIB_ROOT)/include \
	-I$(LIBPNG_ROOT)/include

# ================= 链接选项 (LDFLAGS) =================
# 添加 -Wl,-rpath-link，告诉链接器去哪里找依赖库的 .so 文件 (解决找不到 -lz, -lhdf5 等问题)
LDFLAGS = \
	-L$(PCL_INSTALL_ROOT)/lib \
	-L$(BOOST_ROOT)/lib \
	-L$(FLANN_ROOT)/lib \
	-L$(LZ4_ROOT)/lib \
	-L$(HDF5_ROOT)/lib \
	-L$(GTEST_ROOT)/lib \
	-L$(ZLIB_ROOT)/lib \
	-L$(LIBPNG_ROOT)/lib \
	-Wl,-rpath-link=$(PCL_INSTALL_ROOT)/lib \
	-Wl,-rpath-link=$(BOOST_ROOT)/lib \
	-Wl,-rpath-link=$(FLANN_ROOT)/lib \
	-Wl,-rpath-link=$(LZ4_ROOT)/lib \
	-Wl,-rpath-link=$(HDF5_ROOT)/lib \
	-Wl,-rpath-link=$(GTEST_ROOT)/lib \
	-Wl,-rpath-link=$(ZLIB_ROOT)/lib \
	-Wl,-rpath-link=$(LIBPNG_ROOT)/lib

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
	-lz -lhdf5 -lpng # 可能需要显式链接 lz4, hdf5, libpng

# =============================================================================
#  Rules
# =============================================================================
all: $(TARGET_BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET_BIN): $(SRCS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SRCS) $(LDFLAGS) $(LIBS) -o $(TARGET)
	mv -f $(TARGET) $(TARGET_BIN)

clean:
	rm -f $(TARGET) $(TARGET_BIN)

# 使用行内环境变量设置 (Inline Env Var)，比 export 更安全
# 注意 $$LD_LIBRARY_PATH 用来保留系统原有的路径
run_test: clean $(TARGET_BIN)
	LD_LIBRARY_PATH=$(RISCV_LIB_PATH):$$LD_LIBRARY_PATH \
	qemu-riscv64 -L $(RISCV_SYSROOT) -cpu rv64,v=true,vlen=256,elen=64 ./$(TARGET_BIN)

# =============================================================================
#  Remote Deployment
# =============================================================================
REMOTE_USER = root
REMOTE_IP   = 172.16.89.126
REMOTE_DIR  = ~/pcl-test/sample_consensus/quadric_models

deploy_test: $(TARGET_BIN)
	@echo "[DEPLOY] Stripping and Syncing $(TARGET)..."
	@riscv64-unknown-linux-gnu-strip -s ./$(TARGET_BIN) -o ./$(TARGET)_stripped
	@rsync -avzP ./$(TARGET)_stripped $(REMOTE_USER)@$(REMOTE_IP):$(REMOTE_DIR)/$(TARGET)
	@rm -f ./$(TARGET)_stripped
	@echo "[DONE] $(TARGET) deployed."
