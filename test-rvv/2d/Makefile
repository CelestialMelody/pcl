# =============================================================================
#  PCL Unified Makefile (RISC-V & x86)
#
#  Usage:
# make run_bench        -> 同 common：clean + 编当前 TARGET_BENCH + generate_vec_report，再运行
# make run_bench_std    -> 子 make：TARGET_BENCH=$(TARGET_BENCH_STD)，USE_PCL_RVV10=0
# make run_bench_rvv    -> 子 make：TARGET_BENCH=bench_2d_rvv，USE_PCL_RVV10=1
# make analyze_bench_compare -> 解析 output/run_bench_{std,rvv}.log，打印 Std vs RVV 对比表（.venv 优先）
#   板卡另存日志： make BENCH_STD_OUTPUT_FILE=output/board/run_bench_std.log \
#                  BENCH_RVV_OUTPUT_FILE=output/board/run_bench_rvv.log analyze_bench_compare
# make ARCH=x86 run_bench -> 在 x86 上跑 benchmark
# =============================================================================

# ================================================
# 日志与报告配置
# ================================================
LOG_DIR               = log
VEC_LOGS_DIR          = $(LOG_DIR)/vec_logs
LOG_VEC_MISS_DIR      = $(LOG_DIR)/vec_missed_log
SCRIPT_DIR            = script

TIMESTAMP            := $(shell date +%Y-%m-%d_%H-%M-%S)

LOG_FILE              = $(LOG_VEC_MISS_DIR)/vec_missed_$(TIMESTAMP).log
LATEST_VEC_LOG        = $(LOG_DIR)/latest_vec_missed.log
FILTER_REPORT         = $(LOG_DIR)/filtered_report.log
ANALYZE_REPORT        = $(LOG_DIR)/analyzed_vec_report.log

OUTPUT_DIR            = output
OUTPUT_DIR_QEMU       = $(OUTPUT_DIR)/qemu
OUTPUT_DIR_BOARD      = $(OUTPUT_DIR)/board
BENCH_STD_OUTPUT_FILE = $(OUTPUT_DIR_QEMU)/run_bench_std.log
BENCH_RVV_OUTPUT_FILE = $(OUTPUT_DIR_QEMU)/run_bench_rvv.log
TEST_OUTPUT_FILE      = $(OUTPUT_DIR_QEMU)/run_test.log
BENCH_COMPARE_OUTPUT_FILE = $(OUTPUT_DIR_QEMU)/analyze_bench_compare.log

# 与 test-rvv/common 等目录共用（单份维护）
TEST_RVV_SHARED_SCRIPT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../script)

ANALYZE_VEC_SCRIPT    = $(TEST_RVV_SHARED_SCRIPT_DIR)/analyze_vec_log.py
BENCH_COMPARE_SCRIPT  = $(TEST_RVV_SHARED_SCRIPT_DIR)/analyze_bench_compare.py
# 优先使用本目录 .venv（与 `source .venv/bin/activate` 等价）；否则退回系统 python3
VENV_PYTHON           := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/.venv/bin/python)
PYTHON_BENCH_COMPARE  := $(shell test -x '$(VENV_PYTHON)' && echo '$(VENV_PYTHON)' || echo python3)

# ================================================
# 目标定义
# ================================================
ARCH ?= riscv
BUILD_DIR             = build
ASM_DIR               = $(BUILD_DIR)/asm

# Std: USE_PCL_RVV10=0（关闭 PCL 手写 RVV 内核），但依旧保留 EIGEN_RVV_FLAGS
# RVV: USE_PCL_RVV10=1
# 与 test-rvv/common/Makefile 一致：仅 TARGET_BENCH / TARGET_BENCH_BIN；默认即为带 RVV10 手写内核的二进制名。
TARGET_BENCH_STD ?= bench_2d_std
TARGET_BENCH ?= bench_2d

SRCS_BENCH            = bench_2d.cpp
TARGET_BENCH_STD_BIN = $(BUILD_DIR)/$(ARCH)/$(TARGET_BENCH_STD)
TARGET_BENCH_BIN = $(BUILD_DIR)/$(ARCH)/$(TARGET_BENCH)

TARGET_BENCH_STD_ASM = $(ASM_DIR)/$(ARCH)/$(TARGET_BENCH_STD).asm
TARGET_BENCH_ASM = $(ASM_DIR)/$(ARCH)/$(TARGET_BENCH).asm

TARGET_BENCH_STD_FULL_ASM = $(ASM_DIR)/$(ARCH)/$(TARGET_BENCH_STD).full.asm
TARGET_BENCH_FULL_ASM = $(ASM_DIR)/$(ARCH)/$(TARGET_BENCH).full.asm

SRCS_TEST             = test_2d.cpp
TARGET_TEST           = test_2d
TARGET_TEST_BIN       = $(BUILD_DIR)/$(ARCH)/$(TARGET_TEST)

# Edge store micro-benchmark (independent from existing test_2d)
SRCS_EDGE_STORE_TEST  = test_edge_store_bench.cpp
TARGET_EDGE_STORE_TEST = test_edge_store_bench
TARGET_EDGE_STORE_TEST_BIN = $(BUILD_DIR)/$(ARCH)/$(TARGET_EDGE_STORE_TEST)

# 数学函数相关测试
TARGET_ATAN2_TEST     = atan2_test
SRCS_ATAN2_TEST       = atan2_test.cpp
TARGET_EXPF_TEST      = expf_test
SRCS_EXPF_TEST        = expf_test.cpp
TARGET_EXPF_REMEZ_VS_TAYLOR = expf_remez_vs_taylor
SRCS_EXPF_REMEZ_VS_TAYLOR   = expf_remez_vs_taylor.cpp
TARGET_ATAN2_TEST_BIN = $(BUILD_DIR)/$(ARCH)/$(TARGET_ATAN2_TEST)
TARGET_EXPF_TEST_BIN  = $(BUILD_DIR)/$(ARCH)/$(TARGET_EXPF_TEST)
TARGET_EXPF_REMEZ_VS_TAYLOR_BIN = $(BUILD_DIR)/$(ARCH)/$(TARGET_EXPF_REMEZ_VS_TAYLOR)

# 性能分析相关
PERF_DIR              = $(BUILD_DIR)/perf/$(ARCH)
PERF_DATA             = $(PERF_DIR)/perf.data
PERF_DATA_JIT         = $(PERF_DIR)/perf.data.jitted
PERF_SCRIPT           = $(PERF_DIR)/perf.script

# 优化关注的代码路径
FOCUS_DIR             = pcl/2d/include/pcl/2d/impl/edge.hpp

# 其他路径
WORKSPACE             = /home/zoomin/codes/RISCV/workspace
PCD_FILES_DIR         = $(WORKSPACE)/pcl/test-rvv/2d/pcd
# 命令行参数是按位置固定的输入，顺序是固定的
# lena.pcd gauss_smooth.pcd erosion.pcd dilation.pcd opening.pcd closing.pcd \
# erosion_binary.pcd dilation_binary.pcd opening_binary.pcd closing_binary.pcd canny.pcd
PCD_FILES             = $(PCD_FILES_DIR)/lena.pcd \
	$(PCD_FILES_DIR)/gauss_smooth.pcd \
	$(PCD_FILES_DIR)/erosion.pcd \
	$(PCD_FILES_DIR)/dilation.pcd \
	$(PCD_FILES_DIR)/opening.pcd \
	$(PCD_FILES_DIR)/closing.pcd \
	$(PCD_FILES_DIR)/erosion_binary.pcd \
	$(PCD_FILES_DIR)/dilation_binary.pcd \
	$(PCD_FILES_DIR)/opening_binary.pcd \
	$(PCD_FILES_DIR)/closing_binary.pcd \
	$(PCD_FILES_DIR)/canny.pcd

# 用于将空格列表转换为冒号分隔 (LD_LIBRARY_PATH)
empty :=
space := $(empty) $(empty)

# ================================================
# 架构相关配置 (RISC-V, x86)
# ================================================

ifeq ($(ARCH), riscv)
# ================= 基础工具配置 =================
	CC            = riscv64-unknown-linux-gnu-gcc
	CXX           = riscv64-unknown-linux-gnu-g++

# 依赖库根路径配置
	PCL_INSTALL_ROOT  = $(WORKSPACE)/riscv/pcl-rvv
	PCL_SOURCE_ROOT   = $(WORKSPACE)/pcl
	RISCV_DEPS        = $(WORKSPACE)/riscv
	RISCV_SYSROOT    := $(shell $(CC) -print-sysroot)

# ================= 各个依赖库的具体路径 =================
	BOOST_ROOT  = $(RISCV_DEPS)/boost
	EIGEN_ROOT  = $(RISCV_DEPS)/eigen-rvv
	FLANN_ROOT  = $(RISCV_DEPS)/flann
	LZ4_ROOT    = $(RISCV_DEPS)/lz4
	HDF5_ROOT   = $(RISCV_DEPS)/hdf5
	GTEST_ROOT  = $(RISCV_DEPS)/gtest
	ZLIB_ROOT   = $(RISCV_DEPS)/zlib
	LIBPNG_ROOT = $(RISCV_DEPS)/libpng

# 是否启用 PCL 手写 RVV 内核 -D__RVV10__（仅 riscv 时生效）
# 说明：即使关闭 __RVV10__，也建议保留 EIGEN_RVV_FLAGS 以与 PCL 安装时 Eigen 配置一致，避免 ABI / aligned malloc warning。
	USE_PCL_RVV10 ?= 1

# Eigen RVV1.0 支持所需的编译参数（与 PCL 安装时的 Eigen 配置一致）
# 固定向量长度 256bit，与当前 PCL 构建一致
	EIGEN_RVV_FLAGS ?= -DEIGEN_RISCV64_USE_RVV10 -DEIGEN_RISCV64_DEFAULT_LMUL=2 -march=rv64gcv_zvl256b -mrvv-vector-bits=zvl

# ================= 核心路径管理 =================
	LIB_DIRS_LIST = \
		$(PCL_INSTALL_ROOT)/lib \
		$(BOOST_ROOT)/lib \
		$(FLANN_ROOT)/lib \
		$(LZ4_ROOT)/lib \
		$(HDF5_ROOT)/lib \
		$(GTEST_ROOT)/lib \
		$(ZLIB_ROOT)/lib \
		$(LIBPNG_ROOT)/lib

# 转换为 LD_LIBRARY_PATH 格式
	LIB_PATH_VAL := $(subst $(space),:,$(LIB_DIRS_LIST))

# ================= 编译器标志 =================
	CXXFLAGS_ARCH = -mabi=lp64d \
		-fopt-info-vec-missed=$(LOG_FILE) \
		-DPCL_SILENCE_MALLOC_WARNING=1
	CXXFLAGS_ARCH += $(EIGEN_RVV_FLAGS)
	ifeq ($(USE_PCL_RVV10),1)
	CXXFLAGS_ARCH += -D__RVV10__
	endif

# ================= 链接器标志 =================
	RUN_CMD = LD_LIBRARY_PATH=$(LIB_PATH_VAL):$$LD_LIBRARY_PATH qemu-riscv64 -L $(RISCV_SYSROOT) -cpu rv64,v=true,vlen=256,elen=64

	PERF_TARGET_CMD_STR = qemu-riscv64 -perfmap -L $(RISCV_SYSROOT) -cpu rv64,v=true ./$(TARGET_BENCH_BIN)
	PERF_NEED_JIT = true

# ================= 反汇编工具 =================
	OBJDUMP_CMD = riscv64-unknown-linux-gnu-objdump

# ================= 向量化检查配置 (RVV) =================
# 检查以 v 开头且带 .v 后缀的指令 (如 vadd.vv)，或者 vsetvli
	VEC_TPYE_NAME = RVV (RISC-V Vector)
# 1. [[:space:]]+ 匹配指令前的空格
# 2. v[a-z0-9]+ 匹配 v 开头的指令名（如 vadd, vsetvli）
# 3. (\.[a-z0-9]+)* 匹配可选的后缀（如 .vv, .v）
# 4. [[:space:]]+ 确保指令后有空格（区分于以 v 开头的标量寄存器名，如果存在的话
	VEC_REGEX_STR := "[[:space:]]+v[a-z0-9]+(\.[a-z0-9]+)*[[:space:]]+"
endif

ifeq ($(ARCH), x86)
	CC  = gcc
	CXX = g++
	X86_INSTALL_DIR = /home/zoomin/workspace/x86
	WORKSPACE_X86              = /home/zoomin/workspace
	PCL_SOURCE_ROOT = $(WORKSPACE_X86)/pcl

# ================= 依赖库路径 =================
	PCL_INSTALL_ROOT = $(X86_INSTALL_DIR)/pcl
	EIGEN_ROOT       = $(X86_INSTALL_DIR)/eigen-3.3.9
	BOOST_ROOT       = $(X86_INSTALL_DIR)/boost-1.88.0
# 下面这些如果 x86 是用系统库，可以留空或指向系统路径；这里假设也是安装在 X86_INSTALL_DIR 或用系统默认
# 如果您也在该目录下编译了这些库，请对应修改；否则 x86 通常能自动找到 /usr/lib 下的库
# 这里我们添加一个通用的 lib 搜索以防万一

# ================= 库搜索路径列表 =================
	LIB_DIRS_LIST = $(PCL_INSTALL_ROOT)/lib $(BOOST_ROOT)/lib

# ================= 编译标志 (AVX/Native开启) =================
	CXXFLAGS_ARCH = -march=native -mtune=native \
		-fopt-info-vec-missed=$(LOG_FILE)

# 这里的 $$LD_LIBRARY_PATH 确保系统原来的库也能找到
	LIB_PATH_VAL := $(subst $(space),:,$(LIB_DIRS_LIST))
	RUN_CMD = LD_LIBRARY_PATH=$(LIB_PATH_VAL):$$LD_LIBRARY_PATH

	PERF_TARGET_CMD_STR = ./$(TARGET_BENCH_BIN)
	PERF_NEED_JIT = false

# ================= 反汇编工具 (使用宿主机默认 objdump) =================
	OBJDUMP_CMD = objdump

# ================= 向量化检查配置 (AVX/SSE) =================
# x86 很难通过指令名通配 (指令太多: addps, vmulpd...)
# 最准确的方法是检查是否使用了向量寄存器: %xmm (SSE), %ymm (AVX), %zmm (AVX-512)
	VEC_TPYE_NAME = AVX/SSE (x86 SIMD)
	VEC_REGEX_STR = %[xyz]mm
endif

# ================================================
# 共享编译与链接标志
# ================================================
# 包含路径：
# 1. 优先包含 PCL_SOURCE_ROOT 中的 2d 模块 (确保我们测试的是修改后的代码)
# 2. 包含 PCL 源码中的通用头文件 (common, io 等)
# 3. 包含 安装目录下的 pcl (用于获取生成的 pcl_config.h 等)
# 4. 包含 依赖库 (Eigen, Boost 等)
INCLUDES = \
    -I$(PCL_SOURCE_ROOT)/2d/include \
    -I$(PCL_SOURCE_ROOT)/common/include \
    -I$(PCL_SOURCE_ROOT)/test/include \
    -I$(PCL_INSTALL_ROOT)/include/pcl-1.15 \
    -I$(EIGEN_ROOT)/include/eigen3 \
    -I$(BOOST_ROOT)/include

# 如果是 RISC-V，还需要添加这些特定的 include
ifeq ($(ARCH), riscv)
    INCLUDES += -I$(FLANN_ROOT)/include -I$(LZ4_ROOT)/include \
                -I$(HDF5_ROOT)/include -I$(GTEST_ROOT)/include \
                -I$(ZLIB_ROOT)/include -I$(LIBPNG_ROOT)/include
endif

ifeq ($(ARCH), x86)
    # x86 GTest 通常在系统目录
    # 如果 x86 编译报错找不到 gtest，可以在这里添加路径
    # INCLUDES += -I/path/to/x86/gtest/include
endif

CXXFLAGS = -std=c++17 -O3 -g \
           $(CXXFLAGS_ARCH) \
           -DPCL_NO_PRECOMPILE \
           $(INCLUDES)

# =================== 链接标志 =================
# 自动为所有 LIB_DIRS_LIST 添加 -L 和 -rpath-link
LDFLAGS = $(foreach dir,$(LIB_DIRS_LIST),-L$(dir) -Wl,-rpath-link=$(dir))

# =================== 库标志 =================
LIBS = \
	-lpcl_filters \
	-lpcl_io \
	-lpcl_common \
	-lgtest -lgtest_main -lpthread \
	-lboost_filesystem -lboost_iostreams -lboost_system \
	-lm

# RISC-V 下特定的库
ifeq ($(ARCH), riscv)
    LIBS += -lz -lhdf5 -lpng -llz4
endif

# ================= 构建规则 (Build Rules) =================
# 打印配置信息
info:
	@echo "Architecture    : $(ARCH)"
	@echo "Compiler        : $(CXX)"
	@echo "PCL Source      : $(PCL_SOURCE_ROOT)"
	@echo "PCL Install     : $(PCL_INSTALL_ROOT)"
	@echo "Eigen Root      : $(EIGEN_ROOT)"
	@echo "Boost Root      : $(BOOST_ROOT)"
	@echo "Run Command     : $(RUN_CMD)"

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(LOG_DIR):
	mkdir -p $(LOG_DIR)

$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

$(LOG_VEC_MISS_DIR): $(LOG_DIR)
	mkdir -p $(LOG_VEC_MISS_DIR)

$(VEC_LOGS_DIR): $(LOG_DIR)
	mkdir -p $(VEC_LOGS_DIR)

$(PERF_DIR): $(BUILD_DIR)/$(ARCH)
	mkdir -p $(PERF_DIR)

$(ASM_DIR): $(BUILD_DIR)/$(ARCH)
	mkdir -p $(ASM_DIR)

$(ASM_DIR)/$(ARCH): $(ASM_DIR)
	mkdir -p $(ASM_DIR)/$(ARCH)

$(BUILD_DIR)/$(ARCH): $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/$(ARCH)

$(OUTPUT_DIR_QEMU): $(OUTPUT_DIR)
	mkdir -p $(OUTPUT_DIR_QEMU)

$(OUTPUT_DIR_BOARD): $(OUTPUT_DIR)
	mkdir -p $(OUTPUT_DIR_BOARD)

make_dirs: $(BUILD_DIR)/$(ARCH) $(LOG_DIR) $(LOG_VEC_MISS_DIR) $(VEC_LOGS_DIR) $(PERF_DIR) $(ASM_DIR)/$(ARCH) $(OUTPUT_DIR)

# ================= common.hpp 数学函数测试（迁移到 common/common） =================
# 为避免 2d 与 common 双份维护，这里仅保留转调入口；
# 实际实现位于 test-rvv/common/common/Makefile。
COMMON_COMMON_DIR := ../common/common

parms:
	@$(MAKE) -C $(COMMON_COMMON_DIR) ARCH=$(ARCH) parms

parms_remez_exp:
	@$(MAKE) -C $(COMMON_COMMON_DIR) ARCH=$(ARCH) parms_remez_exp

parms_remez_atan:
	@$(MAKE) -C $(COMMON_COMMON_DIR) ARCH=$(ARCH) parms_remez_atan

run_atan2_test:
	@$(MAKE) -C $(COMMON_COMMON_DIR) ARCH=$(ARCH) run_atan2_test

run_expf_test:
	@$(MAKE) -C $(COMMON_COMMON_DIR) ARCH=$(ARCH) run_expf_test

run_expf_remez_vs_taylor:
	@$(MAKE) -C $(COMMON_COMMON_DIR) ARCH=$(ARCH) run_expf_remez_vs_taylor

# ========= 2D test & bench =========
$(TARGET_TEST_BIN): $(SRCS_TEST) | $(BUILD_DIR)/$(ARCH) $(LOG_VEC_MISS_DIR)
	@echo "[BUILD] Compiling test_2d ($(ARCH))..."
	$(CXX) $(CXXFLAGS) $(SRCS_TEST) $(LDFLAGS) $(LIBS) -o $(TARGET_TEST_BIN)

$(TARGET_EDGE_STORE_TEST_BIN): $(SRCS_EDGE_STORE_TEST) | $(BUILD_DIR)/$(ARCH) $(LOG_VEC_MISS_DIR)
	@echo "[BUILD] Compiling $(TARGET_EDGE_STORE_TEST) ($(ARCH))..."
	$(CXX) $(CXXFLAGS) $(SRCS_EDGE_STORE_TEST) $(LDFLAGS) $(LIBS) -o $(TARGET_EDGE_STORE_TEST_BIN)

$(TARGET_BENCH_BIN): $(SRCS_BENCH) | $(BUILD_DIR)/$(ARCH) $(LOG_VEC_MISS_DIR)
	@echo "[BUILD] Compiling $(TARGET_BENCH) ($(ARCH)) USE_PCL_RVV10=$(USE_PCL_RVV10)..."
	$(CXX) $(CXXFLAGS) $(SRCS_BENCH) $(LDFLAGS) $(LIBS) -o $(TARGET_BENCH_BIN)

run_test: clean_test $(TARGET_TEST_BIN) | $(BUILD_DIR)/$(ARCH) $(OUTPUT_DIR_QEMU)
	@echo "[RUN] Executing Unit Test on $(ARCH)..."
	$(RUN_CMD) ./$(TARGET_TEST_BIN) $(PCD_FILES) 2>&1 | tee $(TEST_OUTPUT_FILE)

run_edge_store_test: clean_edge_store_test $(TARGET_EDGE_STORE_TEST_BIN) | $(BUILD_DIR)/$(ARCH)
	@echo "[RUN] Executing Edge Store Micro-bench on $(ARCH)..."
	$(RUN_CMD) ./$(TARGET_EDGE_STORE_TEST_BIN)

run_bench: clean_bench $(TARGET_BENCH_BIN) generate_vec_report | $(BUILD_DIR)/$(ARCH)
	@echo "[RUN] Executing Benchmark on $(ARCH) USE_PCL_RVV10=$(USE_PCL_RVV10)..."
	$(RUN_CMD) ./$(TARGET_BENCH_BIN)

run_bench_std: clean_bench_std | $(BUILD_DIR)/$(ARCH) $(OUTPUT_DIR_QEMU)
	@$(MAKE) ARCH=$(ARCH) run_bench USE_PCL_RVV10=0 TARGET_BENCH=$(TARGET_BENCH_STD) 2>&1 | tee $(BENCH_STD_OUTPUT_FILE)

run_bench_rvv: clean_bench_rvv | $(BUILD_DIR)/$(ARCH) $(OUTPUT_DIR_QEMU)
	@$(MAKE) ARCH=$(ARCH) run_bench USE_PCL_RVV10=1 TARGET_BENCH=$(TARGET_BENCH) 2>&1 | tee $(BENCH_RVV_OUTPUT_FILE)

# 读取 $(BENCH_STD_OUTPUT_FILE) 与 $(BENCH_RVV_OUTPUT_FILE)，打印 Std vs RVV 对比表（需先跑 run_bench_std / run_bench_rvv）
analyze_bench_compare: $(BENCH_STD_OUTPUT_FILE) $(BENCH_RVV_OUTPUT_FILE) | $(OUTPUT_DIR_QEMU)
	@$(PYTHON_BENCH_COMPARE) $(BENCH_COMPARE_SCRIPT) \
		--std-log $(BENCH_STD_OUTPUT_FILE) --rvv-log $(BENCH_RVV_OUTPUT_FILE) 2>&1 | tee $(BENCH_COMPARE_OUTPUT_FILE)

# ================= 日志分析相关 =================
# 构建 RVV bench -> 选最新日志 -> 过滤 -> 分析
# 1. $(LOG_VEC_MISS_DIR) 目录下通过 ls -1t 寻找最近修改（最新）的 .log 文件
# 2. 如果找到了日志，则执行 ln -sf "$$latest_file" "$(LATEST_VEC_LOG)"
# 3. 定向过滤：使用 $(FOCUS_DIR)（通常是你当前关注的源代码目录）来过滤日志。-A 1 表示匹配行的后一行也要抓取（通常向量化日志一行写位置，下一行写原因）。
# 4. 如果过滤后的日志为空，则提取前 80 行（默认最大行数）的 "missed:" 行。
# 5. 分析：使用 python3 "$(ANALYZE_VEC_SCRIPT)" "$(FILTER_REPORT)" -o "$(ANALYZE_REPORT)" -s "$(VEC_LOGS_DIR)" 生成分析报告。
# 6. 保存分析报告到 $(ANALYZE_REPORT)
generate_vec_report: $(TARGET_BENCH_BIN) | $(LOG_DIR) $(LOG_VEC_MISS_DIR)
	@latest_file=""; \
	if ls "$(LOG_VEC_MISS_DIR)"/vec_missed_*.log >/dev/null 2>&1; then \
		latest_file=$$(realpath "$$(ls -1t "$(LOG_VEC_MISS_DIR)"/vec_missed_*.log | head -n 1)"); \
	else \
		echo "[WARN] No vec-missed log found. Using empty log for report generation."; \
		: > "$(LATEST_VEC_LOG)"; \
	fi; \
	if [ -n "$$latest_file" ]; then \
		ln -sf "$$latest_file" "$(LATEST_VEC_LOG)"; \
	fi; \
	echo "[INFO]  Latest vectorization log linked: $(LATEST_VEC_LOG)"; \
	echo "[INFO]  Filtering report for keywords: '$(FOCUS_DIR)'"; \
	grep '$(FOCUS_DIR)' '$(LATEST_VEC_LOG)' -A 1 > '$(FILTER_REPORT)' || true; \
	if [ ! -s "$(FILTER_REPORT)" ]; then \
		echo "[WARN]  No entries matched FOCUS_DIR. Fallback: extract top missed diagnostics."; \
		grep 'missed:' '$(LATEST_VEC_LOG)' | head -n 80 > '$(FILTER_REPORT)' || true; \
	fi; \
	echo "[INFO]  Clean report generated: $(FILTER_REPORT)"; \
	echo "[ANALYZE] Generating Vectorization Analysis Report..."; \
	python3 "$(ANALYZE_VEC_SCRIPT)" "$(FILTER_REPORT)" -o "$(ANALYZE_REPORT)" -s "$(VEC_LOGS_DIR)"; \
	echo "[DONE] Analysis report saved to: $(ANALYZE_REPORT)"; \
	echo "-------------------------------------------------------"; \
	echo "Check optimization hints:"; \
	echo "  cat $(FILTER_REPORT)"; \
	echo "-------------------------------------------------------"

########################################################
# 反汇编
########################################################

dump_bench: $(TARGET_BENCH_BIN) generate_vec_report | $(ASM_DIR)/$(ARCH)
	@echo "[DUMP] Generating Advanced Disassembly for $(ARCH)..."
	$(OBJDUMP_CMD) -d ./$(TARGET_BENCH_BIN) > $(TARGET_BENCH_ASM)
	@echo "[DONE] Disassembly saved to $(TARGET_BENCH_ASM)"
	@echo "-------------------------------------------------------"
# 参数详解:
# -d: 反汇编可执行段
# -C: 解析 C++ 符号
# -S: 混合源代码 (需要 -g 编译)
# -l: 显示行号
# --no-show-raw-insn: 隐藏机器码，只看汇编
	$(OBJDUMP_CMD) -d -C -S -l --no-show-raw-insn --wide ./$(TARGET_BENCH_BIN) > $(TARGET_BENCH_FULL_ASM)
	@echo "[DONE] Full disassembly with source saved to $(TARGET_BENCH_FULL_ASM)"
	@echo "-------------------------------------------------------"
	@echo "[CHECK] SIMD Instruction Statistics ($(VEC_TPYE_NAME)):"
# 统计匹配到的行数
	@echo "  Total Vector Instructions/Registers found: $$(grep -E $(VEC_REGEX_STR) $(TARGET_BENCH_ASM) | wc -l)"
	@echo "  Sample usage found:"
# 打印前 5 行匹配项供人工检查
	@grep -E $(VEC_REGEX_STR) $(TARGET_BENCH_ASM) | head -n 5 | sed 's/^/    /' || echo "  (None detected)"
	@echo "-------------------------------------------------------"

# =======================================================================
# 清理
# =======================================================================
clean_all: clean_build clean_log clean_perf clean_atan2_test clean_expf_test clean_expf_remez_vs_taylor \
clean_test clean_bench_std clean_bench_rvv

clean_build:
	@echo "[CLEAN] Removing build directory..."
	rm -rf $(BUILD_DIR)
	@echo "[DONE] Build directory cleaned."
	@echo "-------------------------------------------------------"

clean_log:
	@echo "[CLEAN] Removing all log files..."
	rm -rf $(LOG_DIR)/* ${VEC_LOGS_DIR} ${FILTER_REPORT} ${ANALYZE_REPORT} $(LATEST_VEC_LOG)
	@echo "[DONE] All logs cleaned."
	@echo "-------------------------------------------------------"

clean_perf:
	@echo "[CLEAN] Removing perf data..."
	rm -rf $(PERF_DIR)
	@echo "[DONE] Perf data cleaned."
	@echo "-------------------------------------------------------"

clean_atan2_test:
	@echo "[CLEAN] Removing atan2 test..."
	rm -rf $(TARGET_ATAN2_TEST) $(TARGET_ATAN2_TEST_BIN)
	@echo "[DONE] Atan2 test cleaned."
	@echo "-------------------------------------------------------"

clean_expf_test:
	@echo "[CLEAN] Removing expf test..."
	rm -rf $(TARGET_EXPF_TEST) $(TARGET_EXPF_TEST_BIN)
	@echo "[DONE] Expf test cleaned."
	@echo "-------------------------------------------------------"

clean_expf_remez_vs_taylor:
	@echo "[CLEAN] Removing expf remez test..."
	rm -rf $(TARGET_EXPF_REMEZ_VS_TAYLOR) $(TARGET_EXPF_REMEZ_VS_TAYLOR_BIN)

clean_test:
	@echo "[CLEAN] Removing test..."
	rm -rf $(TARGET_TEST) $(TARGET_TEST_BIN)
	@echo "[DONE] Test cleaned."
	@echo "-------------------------------------------------------"

clean_edge_store_test:
	@echo "[CLEAN] Removing edge store micro-bench..."
	rm -rf $(TARGET_EDGE_STORE_TEST) $(TARGET_EDGE_STORE_TEST_BIN)
	@echo "[DONE] Edge store micro-bench cleaned."
	@echo "-------------------------------------------------------"

clean_bench_std:
	@echo "[CLEAN] Removing bench std..."
	rm -rf $(TARGET_BENCH_STD) $(TARGET_BENCH_STD_BIN)
	@echo "[DONE] Bench std cleaned."
	@echo "-------------------------------------------------------"

clean_bench_rvv:
	@echo "[CLEAN] Removing bench bench_2d_rvv..."
	rm -rf bench_2d_rvv $(BUILD_DIR)/$(ARCH)/bench_2d_rvv
	@echo "[DONE] Bench rvv cleaned."
	@echo "-------------------------------------------------------"

clean_bench:
	@echo "[CLEAN] Removing bench $(TARGET_BENCH)..."
	rm -rf $(TARGET_BENCH) $(TARGET_BENCH_BIN)
	@echo "[DONE] Bench cleaned."
	@echo "-------------------------------------------------------"

# =============================================================================
#  Remote Deployment (RISC-V 板卡)
# =============================================================================

REMOTE_USER = root
REMOTE_IP   = 172.16.89.126
REMOTE_DIR  = /root/pcl-test/2d
# 仅位于 $(PCD_FILES_DIR) 目录下的文件，不包含路径
REMOTE_PCD_FILES_DIR = $(REMOTE_DIR)/pcds
REMOTE_PCD_FILES = $(notdir $(PCD_FILES))

deploy_files:
	@echo "[DEPLOY] Syncing PCD files to $(REMOTE_USER)@$(REMOTE_IP):$(REMOTE_DIR)..."
	@ssh $(REMOTE_USER)@$(REMOTE_IP) "mkdir -p $(REMOTE_PCD_FILES_DIR)"
	@echo "[DEPLOY] Syncing analyze_bench_compare.py to $(REMOTE_USER)@$(REMOTE_IP):$(REMOTE_DIR)/script/..."
	@ssh $(REMOTE_USER)@$(REMOTE_IP) "mkdir -p $(REMOTE_DIR)/script"
	@rsync -avzP $(BENCH_COMPARE_SCRIPT) $(REMOTE_USER)@$(REMOTE_IP):$(REMOTE_DIR)/script/
	@echo "[DONE] analyze_bench_compare.py deployed."
	@echo "-------------------------------------------------------"
	@rsync -avzP $(PCD_FILES) $(REMOTE_USER)@$(REMOTE_IP):$(REMOTE_PCD_FILES_DIR)/
	@echo "[DONE] PCD files deployed: $(REMOTE_PCD_FILES)"
	@echo "-------------------------------------------------------"

deploy_bench_std: deploy_files clean_bench_std
	@$(MAKE) --no-print-directory ARCH=$(ARCH) TARGET_BENCH=$(TARGET_BENCH_STD) USE_PCL_RVV10=0 $(TARGET_BENCH_STD_BIN)
	@echo "[DEPLOY] Stripping and Syncing $(TARGET_BENCH_STD)..."
	@riscv64-unknown-linux-gnu-strip -s ./$(TARGET_BENCH_STD_BIN) -o ./$(TARGET_BENCH_STD)_stripped
	@rsync -avzP ./$(TARGET_BENCH_STD)_stripped $(REMOTE_USER)@$(REMOTE_IP):$(REMOTE_DIR)/$(TARGET_BENCH_STD)
	@rm -f ./$(TARGET_BENCH_STD)_stripped
	@echo "[DONE] $(TARGET_BENCH_STD) deployed."

deploy_bench_rvv: deploy_files clean_bench_rvv
	@$(MAKE) --no-print-directory ARCH=$(ARCH) TARGET_BENCH=bench_2d_rvv USE_PCL_RVV10=1 $(BUILD_DIR)/$(ARCH)/bench_2d_rvv
	@echo "[DEPLOY] Stripping and Syncing bench_2d_rvv..."
	@riscv64-unknown-linux-gnu-strip -s ./$(BUILD_DIR)/$(ARCH)/bench_2d_rvv -o ./bench_2d_rvv_stripped
	@rsync -avzP ./bench_2d_rvv_stripped $(REMOTE_USER)@$(REMOTE_IP):$(REMOTE_DIR)/bench_2d_rvv
	@rm -f ./bench_2d_rvv_stripped
	@echo "[DONE] bench_2d_rvv deployed."

deploy_test: $(TARGET_TEST_BIN) deploy_files
	@echo "[DEPLOY] Stripping and Syncing $(TARGET_TEST)..."
	@riscv64-unknown-linux-gnu-strip -s ./$(TARGET_TEST_BIN) -o ./$(TARGET_TEST)_stripped
	@rsync -avzP ./$(TARGET_TEST)_stripped $(REMOTE_USER)@$(REMOTE_IP):$(REMOTE_DIR)/$(TARGET_TEST)
	@rm -f ./$(TARGET_TEST)_stripped
	@echo "[DONE] $(TARGET_TEST) deployed."

deploy_edge_store_test: $(TARGET_EDGE_STORE_TEST_BIN)
	@echo "[DEPLOY] Ensuring remote dir exists: $(REMOTE_DIR)"
	@ssh $(REMOTE_USER)@$(REMOTE_IP) "mkdir -p $(REMOTE_DIR)"
	@echo "[DEPLOY] Stripping and Syncing $(TARGET_EDGE_STORE_TEST)..."
	@riscv64-unknown-linux-gnu-strip -s ./$(TARGET_EDGE_STORE_TEST_BIN) -o ./$(TARGET_EDGE_STORE_TEST)_stripped
	@rsync -avzP ./$(TARGET_EDGE_STORE_TEST)_stripped $(REMOTE_USER)@$(REMOTE_IP):$(REMOTE_DIR)/$(TARGET_EDGE_STORE_TEST)
	@rm -f ./$(TARGET_EDGE_STORE_TEST)_stripped
	@echo "[DONE] $(TARGET_EDGE_STORE_TEST) deployed."

deploy_atan2_test: $(TARGET_ATAN2_TEST_BIN) deploy_files
	@echo "[DEPLOY] Stripping and Syncing $(TARGET_ATAN2_TEST)..."
	@riscv64-unknown-linux-gnu-strip -s ./$(TARGET_ATAN2_TEST_BIN) -o ./$(TARGET_ATAN2_TEST)_stripped
	@rsync -avzP ./$(TARGET_ATAN2_TEST)_stripped $(REMOTE_USER)@$(REMOTE_IP):$(REMOTE_DIR)/$(TARGET_ATAN2_TEST)
	@rm -f ./$(TARGET_ATAN2_TEST)_stripped
	@echo "[DONE] $(TARGET_ATAN2_TEST) deployed."

deploy_expf_test: $(TARGET_EXPF_TEST_BIN) deploy_files
	@echo "[DEPLOY] Stripping and Syncing $(TARGET_EXPF_TEST)..."
	@riscv64-unknown-linux-gnu-strip -s ./$(TARGET_EXPF_TEST_BIN) -o ./$(TARGET_EXPF_TEST)_stripped
	@rsync -avzP ./$(TARGET_EXPF_TEST)_stripped $(REMOTE_USER)@$(REMOTE_IP):$(REMOTE_DIR)/$(TARGET_EXPF_TEST)
	@rm -f ./$(TARGET_EXPF_TEST)_stripped
	@echo "[DONE] $(TARGET_EXPF_TEST) deployed."

deploy_expf_remez_vs_taylor: $(TARGET_EXPF_REMEZ_VS_TAYLOR_BIN) deploy_files
	@echo "[DEPLOY] Stripping and Syncing $(TARGET_EXPF_REMEZ_VS_TAYLOR)..."
	@riscv64-unknown-linux-gnu-strip -s ./$(TARGET_EXPF_REMEZ_VS_TAYLOR_BIN) -o ./$(TARGET_EXPF_REMEZ_VS_TAYLOR)_stripped
	@rsync -avzP ./$(TARGET_EXPF_REMEZ_VS_TAYLOR)_stripped $(REMOTE_USER)@$(REMOTE_IP):$(REMOTE_DIR)/$(TARGET_EXPF_REMEZ_VS_TAYLOR)
	@rm -f ./$(TARGET_EXPF_REMEZ_VS_TAYLOR)_stripped
	@echo "[DONE] $(TARGET_EXPF_REMEZ_VS_TAYLOR) deployed."

.PHONY: all clean clean_build clean_logs clean_perf clean_atan2_test clean_expf_test clean_expf_remez_test \
		clean_test clean_edge_store_test clean_bench clean_bench_std clean_bench_rvv run_bench run_bench_std run_bench_rvv \
		run_edge_store_test analyze_bench_compare deploy_files deploy_bench_std deploy_bench_rvv deploy_test deploy_edge_store_test \
		deploy_analyze_bench_compare_script

# ================= (已丢弃) 性能分析规则 (Perf Profiling) =================
profile_bench: $(TARGET_BENCH_BIN) generate_vec_report | $(PERF_DIR)
	@echo "[PERF] Cleaning old perf data..."
	@rm -f $(PERF_DATA) $(PERF_DATA_JIT) $(PERF_SCRIPT) perf-*.map /tmp/perf-*.map

	@echo "[PERF] Step 1: Recording on $(ARCH)..."
# 使用 LD_LIBRARY_PATH 确保能找到库
# --call-graph dwarf 对 C++ 调试非常重要
	LD_LIBRARY_PATH=$(LIB_PATH_VAL):$$LD_LIBRARY_PATH \
	perf record \
		-e cycles:u \
		-k 1 \
		--call-graph dwarf \
		--output=$(PERF_DATA) \
		-- \
		$(PERF_TARGET_CMD_STR)

ifeq ($(PERF_NEED_JIT), true)
	@echo "[PERF] Step 2: (RISC-V/QEMU) Retrieving JIT map..."
# QEMU 将 map 文件生成在 /tmp 下，需要移过来
	@mv /tmp/perf-*.map $(PERF_DIR) 2>/dev/null || echo "[WARN] No JIT map file found. QEMU might not have run long enough or -perfmap missing."
	@echo "[PERF] Step 3: (RISC-V/QEMU) Injecting JIT symbols..."
# 将 map 文件的符号注入到 perf.data 中
	perf inject -j -i $(PERF_DATA) -o $(PERF_DATA_JIT)
	@echo "[PERF] Step 4: Generating Firefox Profiler Script..."
	perf script -i $(PERF_DATA_JIT) > $(PERF_SCRIPT)
else
	@echo "[PERF] Step 2: (x86/Native) Skipping JIT injection (Standard ELF symbols used)..."
# x86 原生程序不需要 inject 步骤，直接使用原始数据即可
# 但为了统一输出文件名，我们将 perf.data 复制一份或直接生成脚本
	@echo "[PERF] Step 3: Generating Firefox Profiler Script..."
	perf script -i $(PERF_DATA) > $(PERF_SCRIPT)
endif
	@echo "-------------------------------------------------------"
	@echo "[DONE] Profiling Pipeline Finished."
	@echo "Analyze results with:"
	@echo "  1. Firefox Profiler (Recommended): Upload $(PERF_SCRIPT) to profiler.firefox.com"
	@echo "  2. CLI Report: perf report -i $(if $(filter true,$(PERF_NEED_JIT)),$(PERF_DATA_JIT),$(PERF_DATA))"
	@echo "-------------------------------------------------------"
