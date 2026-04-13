ARCH ?= riscv

# =============================================================================
#  Sample Consensus / Plane Models (RVV)
# =============================================================================

ifeq ($(ARCH), riscv)
	CC  = riscv64-unknown-linux-gnu-gcc
	CXX = riscv64-unknown-linux-gnu-g++

# -------------------------------------------------------------------------
# 路径配置
# -------------------------------------------------------------------------
	WORKSPACE        = /home/zoomin/codes/RISCV/workspace
	INSTALL_DIR      = $(WORKSPACE)/riscv
	PCL_INSTALL_ROOT = $(INSTALL_DIR)/pcl-rvv
	PCL_SOURCE_ROOT  = $(WORKSPACE)/pcl
	RISCV_DEPS       = $(INSTALL_DIR)

	BOOST_ROOT       = $(RISCV_DEPS)/boost
	EIGEN_ROOT       = $(RISCV_DEPS)/eigen-rvv
	FLANN_ROOT       = $(RISCV_DEPS)/flann
	LZ4_ROOT         = $(RISCV_DEPS)/lz4
	HDF5_ROOT        = $(RISCV_DEPS)/hdf5
	GTEST_ROOT       = $(RISCV_DEPS)/gtest
	ZLIB_ROOT        = $(RISCV_DEPS)/zlib
	LIBPNG_ROOT      = $(RISCV_DEPS)/libpng

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
# PCD Files
# =============================================================================
PCD_FILES_DIR = $(WORKSPACE)/pcl/test-rvv/sample_consensus/plane_models/pcd
PCD_FILES = $(PCD_FILES_DIR)/sac_plane_test.pcd

# =============================================================================
#  Targets
# =============================================================================

# 目标（GTest 功能测试）
TARGET_TEST = rvv_sac_plane_test
SRCS_TEST   = test_sample_consensus_plane_models.cpp

# 目标（纯 bench，测 RVV vs 标量）
TARGET_BENCH = bench_sac_normal_plane
SRCS_BENCH   = bench_sac_normal_plane.cpp

# 目标（对比两种 RVV 加载策略：vluxei32 vs vluxseg3ei32）
TARGET_BENCH_LOAD = bench_sac_normal_plane_load_compare
SRCS_BENCH_LOAD   = bench_sac_normal_plane_load_compare.cpp

# 编译产物目录
BUILD_DIR = build
TARGET_TEST_BIN  = $(BUILD_DIR)/$(TARGET_TEST)
TARGET_BENCH_BIN = $(BUILD_DIR)/$(TARGET_BENCH)
TARGET_BENCH_LOAD_BIN = $(BUILD_DIR)/$(TARGET_BENCH_LOAD)

# =============================================================================
#  Build Flags
# =============================================================================

EIGEN_RVV_FLAGS ?= -DEIGEN_RISCV64_USE_RVV10 -DEIGEN_RISCV64_DEFAULT_LMUL=2 -march=rv64gcv_zvl256b -mrvv-vector-bits=zvl

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

# ================= 编译选项 (CXXFLAGS) =================
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
	-lz -lhdf5 -llz4 -lpng # 显式链接 lz4, hdf5, libpng

# =============================================================================
#  Build Rules
# =============================================================================

all: $(TARGET_TEST_BIN) $(TARGET_BENCH_BIN) $(TARGET_BENCH_LOAD_BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET_TEST_BIN): $(SRCS_TEST) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SRCS_TEST) $(LDFLAGS) $(LIBS) -o $(TARGET_TEST)
	mv -f $(TARGET_TEST) $(TARGET_TEST_BIN)

$(TARGET_BENCH_BIN): $(SRCS_BENCH) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SRCS_BENCH) $(LDFLAGS) $(LIBS) -o $(TARGET_BENCH)
	mv -f $(TARGET_BENCH) $(TARGET_BENCH_BIN)

$(TARGET_BENCH_LOAD_BIN): $(SRCS_BENCH_LOAD) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SRCS_BENCH_LOAD) $(LDFLAGS) $(LIBS) -o $(TARGET_BENCH_LOAD)
	mv -f $(TARGET_BENCH_LOAD) $(TARGET_BENCH_LOAD_BIN)

clean:
	rm -f $(TARGET_TEST) $(TARGET_BENCH) $(TARGET_BENCH_LOAD) \
		$(TARGET_TEST_BIN) $(TARGET_BENCH_BIN) $(TARGET_BENCH_LOAD_BIN)

# =============================================================================
#  Run Rules
# =============================================================================

# 使用行内环境变量设置 (Inline Env Var)
# 注意 $$LD_LIBRARY_PATH 用来保留系统原有的路径
run_test: clean $(TARGET_TEST_BIN)
	LD_LIBRARY_PATH=$(RISCV_LIB_PATH):$$LD_LIBRARY_PATH \
	qemu-riscv64 -L $(RISCV_SYSROOT) -cpu rv64,v=true,vlen=256,elen=64 ./$(TARGET_TEST_BIN) $(PCD_FILES)

run_bench: clean $(TARGET_BENCH_BIN)
	LD_LIBRARY_PATH=$(RISCV_LIB_PATH):$$LD_LIBRARY_PATH \
	qemu-riscv64 -L $(RISCV_SYSROOT) -cpu rv64,v=true,vlen=256,elen=64 ./$(TARGET_BENCH_BIN) $(PCD_FILES)

run_bench_load: clean $(TARGET_BENCH_LOAD_BIN)
	LD_LIBRARY_PATH=$(RISCV_LIB_PATH):$$LD_LIBRARY_PATH \
	qemu-riscv64 -L $(RISCV_SYSROOT) -cpu rv64,v=true,vlen=256,elen=64 ./$(TARGET_BENCH_LOAD_BIN) $(PCD_FILES) 200

# =============================================================================
#  Remote Deployment (RISC-V 板卡)
# =============================================================================

REMOTE_USER = root
REMOTE_IP   = 172.16.89.126
REMOTE_DIR  = ~/pcl-test/sample_consensus/plane_models
REMOTE_PCD_FILES_DIR = $(REMOTE_DIR)/pcd
REMOTE_PCD_FILES = $(notdir $(PCD_FILES))

# 需要一并拷贝到板卡的 PCD 测试数据
PCD_FILE ?= sac_plane_test.pcd
PCD_FILE_PATH := $(PCD_FILES_DIR)/$(PCD_FILE)

deploy_files: $(PCD_FILES)
	@echo "[DEPLOY] Syncing PCD test data $(PCD_FILES) only..."
	@ssh $(REMOTE_USER)@$(REMOTE_IP) "mkdir -p $(REMOTE_PCD_FILES_DIR)"
	@rsync -avzP $(PCD_FILES) $(REMOTE_USER)@$(REMOTE_IP):$(REMOTE_PCD_FILES_DIR)/
	@echo "[DONE] PCD test data deployed."

deploy_bench: clean $(TARGET_BENCH_BIN) $(PCD_FILE_PATH) deploy_files
	@echo "[DEPLOY] Stripping and Syncing $(TARGET_BENCH)..."
	@riscv64-unknown-linux-gnu-strip -s ./$(TARGET_BENCH_BIN) -o ./$(TARGET_BENCH)_stripped
	@rsync -avzP ./$(TARGET_BENCH)_stripped $(REMOTE_USER)@$(REMOTE_IP):$(REMOTE_DIR)/$(TARGET_BENCH)
	@rm -f ./$(TARGET_BENCH)_stripped
	@echo "[DONE] $(TARGET_BENCH) deployed."

deploy_bench_load: clean $(TARGET_BENCH_LOAD_BIN) $(PCD_FILE_PATH) deploy_files
	@echo "[DEPLOY] Stripping and Syncing $(TARGET_BENCH_LOAD)..."
	@riscv64-unknown-linux-gnu-strip -s ./$(TARGET_BENCH_LOAD_BIN) -o ./$(TARGET_BENCH_LOAD)_stripped
	@rsync -avzP ./$(TARGET_BENCH_LOAD)_stripped $(REMOTE_USER)@$(REMOTE_IP):$(REMOTE_DIR)/$(TARGET_BENCH_LOAD)
	@rm -f ./$(TARGET_BENCH_LOAD)_stripped
	@echo "[DONE] $(TARGET_BENCH_LOAD) deployed."

deploy_test: clean $(TARGET_TEST_BIN) $(PCD_FILE_PATH) deploy_files
	@echo "[DEPLOY] Stripping and Syncing $(TARGET_TEST)..."
	@riscv64-unknown-linux-gnu-strip -s ./$(TARGET_TEST_BIN) -o ./$(TARGET_TEST)_stripped
	@rsync -avzP ./$(TARGET_TEST)_stripped $(REMOTE_USER)@$(REMOTE_IP):$(REMOTE_DIR)/$(TARGET_TEST)
	@rm -f ./$(TARGET_TEST)_stripped
	@echo "[DONE] $(TARGET_TEST) deployed."
