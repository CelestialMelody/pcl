# =============================================================================
#  PCL common/src/gaussian.cpp — test-rvv（bench / 单测 / 单独 TU 向量化诊断）
# =============================================================================
#
# 说明：
# - bench / 常规单测 通过链接 libpcl_common 调用库内 gaussian 实现；改 common 后需重装库。
# - test_gaussian_convolve_compare：使用本目录 **gaussian_convolve_float_local.hpp**（与 **common/src/gaussian.cpp**
#   中卷积须人工同步），**不链 libpcl_common**，只链 gtest；同进程内标量与 RVV 对拍，与装哪套 PCL 动态库无关。
# - compile_gaussian_cpp：仅将 gaussian.cpp 编译为 .o，做 vec-missed 诊断，无需整库重链。

LOG_DIR           = log
TIMESTAMP        := $(shell date +%Y-%m-%d_%H-%M-%S)
LOG_FILE          = $(LOG_VEC_MISS_DIR)/vec_missed_$(TIMESTAMP).log
LATEST_VEC_LOG    = $(LOG_DIR)/latest_vec_missed.log
FOCUS_DIR         = gaussian.cpp
FILTER_REPORT     = $(LOG_DIR)/filtered_gaussian.log
VEC_LOGS_DIR      = $(LOG_DIR)/vec_logs
LOG_VEC_MISS_DIR  = $(LOG_DIR)/vec_missed_log
ANALYZE_REPORT    = $(LOG_DIR)/analyze_gaussian.log

OUTPUT_DIR        = output
OUTPUT_DIR_BOARD  = $(OUTPUT_DIR)/board
OUTPUT_DIR_QEMU   = $(OUTPUT_DIR)/qemu

BUILD_DIR         = build
ASM_DIR           = $(BUILD_DIR)/asm
TEST_RVV_SHARED_SCRIPT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../../script)

BENCH_STD_OUTPUT_FILE     = $(OUTPUT_DIR_QEMU)/run_bench_std.log
BENCH_RVV_OUTPUT_FILE     = $(OUTPUT_DIR_QEMU)/run_bench_rvv.log
BENCH_COMPARE_OUTPUT_FILE = $(OUTPUT_DIR_QEMU)/bench_compare.log
TEST_STD_OUTPUT_FILE      = $(OUTPUT_DIR_QEMU)/run_test_std.log
TEST_RVV_OUTPUT_FILE      = $(OUTPUT_DIR_QEMU)/run_test_rvv.log
TEST_OUTPUT_FILE          = $(OUTPUT_DIR_QEMU)/run_test.log
TEST_COMPARE_OUTPUT_FILE  = $(OUTPUT_DIR_QEMU)/run_test_convolve_compare.log

ANALYZE_VEC_SCRIPT    = $(TEST_RVV_SHARED_SCRIPT_DIR)/analyze_vec_log.py
BENCH_COMPARE_SCRIPT  = $(TEST_RVV_SHARED_SCRIPT_DIR)/analyze_bench_compare.py
COMPARE_ASM_SCRIPT    = $(TEST_RVV_SHARED_SCRIPT_DIR)/compare_rvv_asm.py

VENV_ACTIVATE := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/.venv/bin/activate)
PYTHON ?= python3
PYTHON_RUN = bash -lc 'if [ -f "$(VENV_ACTIVATE)" ]; then . "$(VENV_ACTIVATE)"; fi; $(PYTHON) "$$@"' --

TARGET_BENCH       ?= bench_gaussian
TARGET_BENCH_STD   ?= bench_gaussian_std
TARGET_BENCH_RVV   ?= bench_gaussian_rvv
SRCS_BENCH          = bench_gaussian.cpp
TARGET_TEST        ?= test_gaussian
TARGET_TEST_STD    ?= test_gaussian_std
TARGET_TEST_RVV    ?= test_gaussian_rvv
# 同 TU 内 gaussian_convolve_float_local.hpp 标量与 RVV 对拍；可执行**固定** -D__RVV10__（见 CXXFLAGS_TEST_COMPARE），不链 libpcl_common
TARGET_TEST_COMPARE ?= test_gaussian_convolve_compare
SRCS_TEST           = test_gaussian.cpp
SRCS_TEST_COMPARE   = test_gaussian_convolve_compare.cpp
HDRS_TEST_COMPARE   = gaussian_convolve_float_local.hpp

GAUSSIAN_CPP        = $(PCL_SOURCE_ROOT)/common/src/gaussian.cpp
GAUSSIAN_OBJ        = $(BUILD_DIR)/$(ARCH)/gaussian_cpp.o

ARCH ?= riscv
TARGET_BENCH_BIN = $(BUILD_DIR)/$(ARCH)/$(TARGET_BENCH)
TARGET_TEST_BIN  = $(BUILD_DIR)/$(ARCH)/$(TARGET_TEST)
TARGET_TEST_COMPARE_BIN = $(BUILD_DIR)/$(ARCH)/$(TARGET_TEST_COMPARE)
TARGET_BENCH_ASM      = $(ASM_DIR)/$(ARCH)/$(TARGET_BENCH).asm
TARGET_BENCH_FULL_ASM = $(ASM_DIR)/$(ARCH)/$(TARGET_BENCH).full.asm

USE_PCL_RVV10  ?= 1

CC  = riscv64-unknown-linux-gnu-gcc
CXX = riscv64-unknown-linux-gnu-g++
STRIP = riscv64-unknown-linux-gnu-strip
OBJDUMP = riscv64-unknown-linux-gnu-objdump

WS               = /home/zoomin/codes/RISCV/workspace
PCL_INSTALL_ROOT = $(WS)/riscv/pcl-rvv
PCL_SOURCE_ROOT  = $(WS)/pcl
RISCV_DEPS       = $(WS)/riscv
RISCV_SYSROOT   := $(shell $(CC) -print-sysroot)

BOOST_ROOT  = $(RISCV_DEPS)/boost
EIGEN_ROOT  = $(RISCV_DEPS)/eigen-rvv
GTEST_ROOT  = $(RISCV_DEPS)/gtest
ZLIB_ROOT   = $(RISCV_DEPS)/zlib
LIBPNG_ROOT = $(RISCV_DEPS)/libpng

EIGEN_RVV_FLAGS ?= -DEIGEN_RISCV64_USE_RVV10 -march=rv64gcv_zvl256b -mrvv-vector-bits=zvl
LIB_DIRS_LIST = $(PCL_INSTALL_ROOT)/lib $(BOOST_ROOT)/lib $(GTEST_ROOT)/lib \
	$(ZLIB_ROOT)/lib $(LIBPNG_ROOT)/lib
empty :=
space := $(empty) $(empty)
LIB_PATH_VAL := $(subst $(space),:,$(LIB_DIRS_LIST))

CXXFLAGS_ARCH = -march=rv64gcv -mabi=lp64d \
	-fopt-info-vec-missed=$(LOG_FILE) \
	-DPCL_SILENCE_MALLOC_WARNING=1
CXXFLAGS_ARCH += $(EIGEN_RVV_FLAGS)
ifeq ($(USE_PCL_RVV10),1)
CXXFLAGS_ARCH += -D__RVV10__
endif

LDFLAGS = $(foreach dir,$(LIB_DIRS_LIST),-L$(dir) -Wl,-rpath-link=$(dir))
RUN_CMD = LD_LIBRARY_PATH=$(LIB_PATH_VAL):$$LD_LIBRARY_PATH qemu-riscv64 -L $(RISCV_SYSROOT) -cpu rv64,v=true,vlen=256,elen=64
VEC_TPYE_NAME = RVV (RISC-V Vector)
VEC_REGEX_STR := "[[:space:]]+v[a-z0-9]+(\.[a-z0-9]+)*[[:space:]]+"

INCLUDES = \
	-I$(PCL_SOURCE_ROOT)/common/include \
	-I$(PCL_SOURCE_ROOT)/test/include \
	-I$(PCL_SOURCE_ROOT) \
	-I$(PCL_INSTALL_ROOT)/include/pcl-1.15 \
	-I$(EIGEN_ROOT)/include/eigen3 \
	-I$(BOOST_ROOT)/include \
	-I$(GTEST_ROOT)/include \
	-I$(ZLIB_ROOT)/include \
	-I$(LIBPNG_ROOT)/include

EXTRA_CXXFLAGS ?=
EXTRA_LDFLAGS  ?=
CXXFLAGS = -std=c++17 -O3 -g $(CXXFLAGS_ARCH) -DPCL_NO_PRECOMPILE $(INCLUDES) $(EXTRA_CXXFLAGS)
LIBS_BENCH = -lpcl_common -lm
LIBS_TEST  = -lpcl_common -lgtest -lpthread -lm
# 对拍可执行：仅 gtest + pthread + m；**始终** -D__RVV10__（不跟随全局 USE_PCL_RVV10，单写 make run_test_convolve_compare 即可）
INCLUDES_TEST_COMPARE  = -I$(GTEST_ROOT)/include
CXXFLAGS_TEST_COMPARE  = -std=c++17 -O3 -g -march=rv64gcv -mabi=lp64d \
	-D__RVV10__ \
	$(INCLUDES_TEST_COMPARE) $(EXTRA_CXXFLAGS)
LDFLAGS_TEST_COMPARE   = -L$(GTEST_ROOT)/lib -Wl,-rpath-link=$(GTEST_ROOT)/lib
LIBS_TEST_COMPARE     = -lgtest -lpthread -lm
RUN_CMD_TEST_COMPARE  = LD_LIBRARY_PATH=$(GTEST_ROOT)/lib:$$LD_LIBRARY_PATH qemu-riscv64 -L $(RISCV_SYSROOT) -cpu rv64,v=true,vlen=256,elen=64

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)
$(LOG_DIR):
	mkdir -p $(LOG_DIR)
$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)
$(VEC_LOGS_DIR): $(LOG_DIR)
	mkdir -p $(VEC_LOGS_DIR)
$(LOG_VEC_MISS_DIR): $(LOG_DIR)
	mkdir -p $(LOG_VEC_MISS_DIR)
$(BUILD_DIR)/$(ARCH): $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/$(ARCH)
$(ASM_DIR): $(BUILD_DIR)
	mkdir -p $(ASM_DIR)
$(ASM_DIR)/$(ARCH): $(ASM_DIR)
	mkdir -p $(ASM_DIR)/$(ARCH)
$(OUTPUT_DIR_QEMU): $(OUTPUT_DIR)
	mkdir -p $(OUTPUT_DIR_QEMU)
$(OUTPUT_DIR_BOARD): $(OUTPUT_DIR)
	mkdir -p $(OUTPUT_DIR_BOARD)

# 仅编译 gaussian.cpp → 向量化诊断针对本 TU（不产生可执行文件）
$(GAUSSIAN_OBJ): $(GAUSSIAN_CPP) | $(LOG_DIR) $(LOG_VEC_MISS_DIR) $(BUILD_DIR)/$(ARCH)
	@echo "[BUILD] compile_gaussian_cpp (object only, for vec-missed on gaussian.cpp)..."
	$(CXX) $(CXXFLAGS) -c -o $(GAUSSIAN_OBJ) $(GAUSSIAN_CPP)

compile_gaussian_cpp: $(GAUSSIAN_OBJ)
	@echo "[DONE] $(GAUSSIAN_OBJ) ; see vec-missed under $(LOG_VEC_MISS_DIR)/"

$(TARGET_BENCH_BIN): $(SRCS_BENCH) | $(LOG_DIR) $(LOG_VEC_MISS_DIR) $(BUILD_DIR)/$(ARCH)
	@echo "[BUILD] Compiling Benchmark (riscv)..."
	$(CXX) $(CXXFLAGS) $(SRCS_BENCH) $(LDFLAGS) $(EXTRA_LDFLAGS) $(LIBS_BENCH) -o $(TARGET_BENCH_BIN)

$(TARGET_TEST_BIN): $(SRCS_TEST) | $(LOG_DIR) $(LOG_VEC_MISS_DIR) $(BUILD_DIR)/$(ARCH)
	@echo "[BUILD] Compiling Unit Test (riscv)..."
	$(CXX) $(CXXFLAGS) $(SRCS_TEST) $(LDFLAGS) $(EXTRA_LDFLAGS) $(LIBS_TEST) -o $(TARGET_TEST_BIN)

$(TARGET_TEST_COMPARE_BIN): $(SRCS_TEST_COMPARE) $(HDRS_TEST_COMPARE) | $(BUILD_DIR)/$(ARCH)
	@echo "[BUILD] test_gaussian_convolve_compare: local hpp + gtest only (no -lpcl_common)..."
	$(CXX) $(CXXFLAGS_TEST_COMPARE) $(SRCS_TEST_COMPARE) $(LDFLAGS_TEST_COMPARE) $(EXTRA_LDFLAGS) $(LIBS_TEST_COMPARE) -o $(TARGET_TEST_COMPARE_BIN)

run_bench: clean_bench $(TARGET_BENCH_BIN)
	@echo "[RUN] Benchmark on riscv (USE_PCL_RVV10=$(USE_PCL_RVV10))..."
	$(RUN_CMD) ./$(TARGET_BENCH_BIN)

run_test: clean_test $(TARGET_TEST_BIN) | $(BUILD_DIR)/$(ARCH) $(OUTPUT_DIR_QEMU)
	@echo "[RUN] Unit Test on riscv..."
	$(RUN_CMD) ./$(TARGET_TEST_BIN) 2>&1 | tee $(TEST_OUTPUT_FILE)

run_test_std: clean_test_std | $(BUILD_DIR)/$(ARCH) $(OUTPUT_DIR_QEMU)
	@$(MAKE) run_test USE_PCL_RVV10=0 TARGET_TEST=$(TARGET_TEST_STD) TEST_OUTPUT_FILE=$(TEST_STD_OUTPUT_FILE)
run_test_rvv: clean_test_rvv | $(BUILD_DIR)/$(ARCH) $(OUTPUT_DIR_QEMU)
	@$(MAKE) run_test USE_PCL_RVV10=1 TARGET_TEST=$(TARGET_TEST_RVV) TEST_OUTPUT_FILE=$(TEST_RVV_OUTPUT_FILE)
run_test_compare: run_test_std run_test_rvv

# 单进程对拍本目录 hpp 中标量与 RVV（不依赖安装版 libpcl；与双库对拍是不同证明路径）
run_test_convolve_compare: clean_test_convolve_compare $(TARGET_TEST_COMPARE_BIN) | $(OUTPUT_DIR_QEMU)
	@echo "[RUN] test_gaussian_convolve_compare (no libpcl_common, always -D__RVV10__)..."
	$(RUN_CMD_TEST_COMPARE) ./$(TARGET_TEST_COMPARE_BIN) 2>&1 | tee $(TEST_COMPARE_OUTPUT_FILE)

run_bench_rvv: clean_bench_rvv | $(BUILD_DIR)/$(ARCH) $(OUTPUT_DIR_QEMU)
	@$(MAKE) run_bench USE_PCL_RVV10=1 TARGET_BENCH=$(TARGET_BENCH_RVV) 2>&1 | tee $(BENCH_RVV_OUTPUT_FILE)
run_bench_std: clean_bench_std | $(BUILD_DIR)/$(ARCH) $(OUTPUT_DIR_QEMU)
	@$(MAKE) run_bench USE_PCL_RVV10=0 TARGET_BENCH=$(TARGET_BENCH_STD) 2>&1 | tee $(BENCH_STD_OUTPUT_FILE)

analyze_bench_compare: $(BENCH_STD_OUTPUT_FILE) $(BENCH_RVV_OUTPUT_FILE) | $(OUTPUT_DIR_QEMU)
	@$(PYTHON_RUN) $(BENCH_COMPARE_SCRIPT) \
		--std-log $(BENCH_STD_OUTPUT_FILE) --rvv-log $(BENCH_RVV_OUTPUT_FILE) 2>&1 | tee $(BENCH_COMPARE_OUTPUT_FILE)
run_bench_compare: run_bench_std run_bench_rvv analyze_bench_compare

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
	$(PYTHON_RUN) "$(ANALYZE_VEC_SCRIPT)" "$(FILTER_REPORT)" -o "$(ANALYZE_REPORT)" -s "$(VEC_LOGS_DIR)"; \
	echo "[DONE] Analysis report saved to: $(ANALYZE_REPORT)"

generate_vec_report_gaussian_cpp: $(GAUSSIAN_OBJ) | $(LOG_DIR) $(LOG_VEC_MISS_DIR)
	@latest_file=""; \
	if ls "$(LOG_VEC_MISS_DIR)"/vec_missed_*.log >/dev/null 2>&1; then \
		latest_file=$$(realpath "$$(ls -1t "$(LOG_VEC_MISS_DIR)"/vec_missed_*.log | head -n 1)"); \
	else \
		: > "$(LATEST_VEC_LOG)"; \
	fi; \
	if [ -n "$$latest_file" ]; then \
		ln -sf "$$latest_file" "$(LATEST_VEC_LOG)"; \
	fi; \
	echo "[INFO] vec-missed for gaussian.cpp TU -> filter gaussian.cpp"; \
	grep 'gaussian.cpp' '$(LATEST_VEC_LOG)' -A 1 > '$(FILTER_REPORT)' || true; \
	$(PYTHON_RUN) "$(ANALYZE_VEC_SCRIPT)" "$(FILTER_REPORT)" -o "$(ANALYZE_REPORT)" -s "$(VEC_LOGS_DIR)"; \
	echo "[DONE] $(ANALYZE_REPORT)"

dump_bench_inner: $(TARGET_BENCH_BIN) | $(ASM_DIR)/$(ARCH)
	@echo "[DUMP] Generating disassembly for riscv (USE_PCL_RVV10=$(USE_PCL_RVV10), TARGET=$(TARGET_BENCH))..."
	@$(OBJDUMP) -d ./$(TARGET_BENCH_BIN) > $(TARGET_BENCH_ASM)
	@$(OBJDUMP) -d -C -S -l --no-show-raw-insn --wide ./$(TARGET_BENCH_BIN) > $(TARGET_BENCH_FULL_ASM)
	@echo "[DONE] Full disassembly saved to $(TARGET_BENCH_FULL_ASM)"
	@echo "[CHECK] Vector instructions/registers ($(VEC_TPYE_NAME)) in $(TARGET_BENCH_ASM):"
	@echo "  Total matches: $$(grep -E $(VEC_REGEX_STR) $(TARGET_BENCH_ASM) | wc -l)"

dump_bench: clean dump_bench_inner
dump_bench_rvv:
	@$(MAKE) clean
	@$(MAKE) dump_bench_inner USE_PCL_RVV10=1 TARGET_BENCH=$(TARGET_BENCH_RVV)
dump_bench_std:
	@$(MAKE) clean
	@$(MAKE) dump_bench_inner USE_PCL_RVV10=0 TARGET_BENCH=$(TARGET_BENCH_STD)

ASM_FULL_STD ?= $(ASM_DIR)/$(ARCH)/$(TARGET_BENCH_STD).full.asm
ASM_FULL_RVV ?= $(ASM_DIR)/$(ARCH)/$(TARGET_BENCH_RVV).full.asm
compare_asm: dump_bench_rvv dump_bench_std $(COMPARE_ASM_SCRIPT) | $(ASM_DIR)
	@$(PYTHON_RUN) $(COMPARE_ASM_SCRIPT) "$(ASM_FULL_STD)" "$(ASM_FULL_RVV)" $(EXTRA_ARGS)

clean:
	rm -f $(LATEST_VEC_LOG)
	rm -f $(TARGET_BENCH_BIN) $(TARGET_TEST_BIN) $(TARGET_TEST_COMPARE_BIN)
	rm -f $(GAUSSIAN_OBJ)
	rm -rf $(ASM_DIR)/$(ARCH)
	rm -rf $(FILTER_REPORT) $(ANALYZE_REPORT) $(VEC_LOGS_DIR) $(LOG_VEC_MISS_DIR)
clean_bench_std:
	rm -rf $(BUILD_DIR)/$(ARCH)/$(TARGET_BENCH_STD)
clean_bench_rvv:
	rm -rf $(BUILD_DIR)/$(ARCH)/$(TARGET_BENCH_RVV)
clean_bench:
	rm -rf $(BUILD_DIR)/$(ARCH)/bench_gaussian
clean_test:
	rm -rf $(BUILD_DIR)/$(ARCH)/test_gaussian
clean_test_convolve_compare:
	rm -f $(TARGET_TEST_COMPARE_BIN)
clean_test_std:
	rm -rf $(BUILD_DIR)/$(ARCH)/$(TARGET_TEST_STD)
clean_test_rvv:
	rm -rf $(BUILD_DIR)/$(ARCH)/$(TARGET_TEST_RVV)

info:
	@echo "ARCH             : $(ARCH)"
	@echo "CXX              : $(CXX)"
	@echo "PCL_SOURCE_ROOT  : $(PCL_SOURCE_ROOT)"
	@echo "GAUSSIAN_CPP     : $(GAUSSIAN_CPP)"
	@echo "PCL_INSTALL_ROOT : $(PCL_INSTALL_ROOT)"
	@echo "RUN_CMD          : $(RUN_CMD)"

REMOTE_USER = root
REMOTE_IP   = 172.16.89.126
REMOTE_DIR  = ~/pcl-test/common/gaussian
REMOTE_SCRIPT_DIR = $(REMOTE_DIR)/script

REMOTE_LIB_DIR = ~/pcl-test/lib
TEMP_LIB_DIR = ./slim_lib
CC = riscv64-unknown-linux-gnu-gcc
CXX = riscv64-unknown-linux-gnu-g++
WORKSPACE = /home/zoomin/codes/RISCV/workspace
PCL_INSTALL_ROOT = $(WORKSPACE)/riscv/pcl-rvv
RISCV_DEPS = $(WORKSPACE)/riscv
SYSROOT := $(shell $(CC) -print-sysroot)

deploy_lib:
	@echo "[PREPARE] Creating local slim_lib directory..."
	@mkdir -p $(TEMP_LIB_DIR)

	@echo "[COPY] Collecting libraries..."
# 1. 收集 PCL 核心库
	@cp $(PCL_INSTALL_ROOT)/lib/lib*.so* $(TEMP_LIB_DIR)/ 2>/dev/null || true
# 2. 收集 关键依赖库 (boost, lz4, hdf5, flann, png, zlib)
	@cp $(RISCV_DEPS)/boost/lib/lib*.so* $(TEMP_LIB_DIR)/ 2>/dev/null || true
	@cp $(RISCV_DEPS)/lz4/lib/lib*.so* $(TEMP_LIB_DIR)/ 2>/dev/null || true
	@cp $(RISCV_DEPS)/hdf5/lib/lib*.so* $(TEMP_LIB_DIR)/ 2>/dev/null || true
	@cp $(RISCV_DEPS)/flann/lib/lib*.so* $(TEMP_LIB_DIR)/ 2>/dev/null || true
	@cp $(RISCV_DEPS)/libpng/lib/lib*.so* $(TEMP_LIB_DIR)/ 2>/dev/null || true
	@cp $(RISCV_DEPS)/zlib/lib/lib*.so* $(TEMP_LIB_DIR)/ 2>/dev/null || true
# 3. 收集 GTest
	@cp $(RISCV_DEPS)/gtest/lib/lib*.so* $(TEMP_LIB_DIR)/ 2>/dev/null || true
# 4. 收集 系统基础库
	@cp $(SYSROOT)/lib/libstdc++.so* $(SYSROOT)/lib/libc.so* $(SYSROOT)/lib/libm.so* $(SYSROOT)/lib/libgcc_s.so* $(TEMP_LIB_DIR)/ 2>/dev/null || true

	@echo "[STRIP] Slimming down libraries..."
	@riscv64-unknown-linux-gnu-strip -s $(TEMP_LIB_DIR)/*.so* 2>/dev/null || true

	@echo "[DEPLOY] Syncing slim lib to Jupiter..."
	@ssh $(REMOTE_USER)@$(REMOTE_IP) "mkdir -p $(REMOTE_LIB_DIR)"
	@rsync -avzP $(TEMP_LIB_DIR)/ $(REMOTE_USER)@$(REMOTE_IP):$(REMOTE_LIB_DIR)/

	@echo "[CLEANUP] Removing local temporary slim_lib..."
	@rm -rf $(TEMP_LIB_DIR)
	@echo "[DONE] All essential lib are deployed."

deploy_files:
	@ssh $(REMOTE_USER)@$(REMOTE_IP) "mkdir -p $(REMOTE_SCRIPT_DIR)"
	@rsync -avzP $(BENCH_COMPARE_SCRIPT) $(REMOTE_USER)@$(REMOTE_IP):$(REMOTE_SCRIPT_DIR)/

deploy_bench_rvv: deploy_files clean_bench_rvv
	@$(MAKE) USE_PCL_RVV10=1 TARGET_BENCH=$(TARGET_BENCH_RVV) $(BUILD_DIR)/$(ARCH)/$(TARGET_BENCH_RVV)
	@$(STRIP) -s ./$(BUILD_DIR)/$(ARCH)/$(TARGET_BENCH_RVV) -o ./$(TARGET_BENCH_RVV)_stripped
	@rsync -avzP ./$(TARGET_BENCH_RVV)_stripped $(REMOTE_USER)@$(REMOTE_IP):$(REMOTE_DIR)/$(TARGET_BENCH_RVV)
	@rm -f ./$(TARGET_BENCH_RVV)_stripped

deploy_bench_std: deploy_files clean_bench_std
	@$(MAKE) USE_PCL_RVV10=0 TARGET_BENCH=$(TARGET_BENCH_STD) $(BUILD_DIR)/$(ARCH)/$(TARGET_BENCH_STD)
	@$(STRIP) -s ./$(BUILD_DIR)/$(ARCH)/$(TARGET_BENCH_STD) -o ./$(TARGET_BENCH_STD)_stripped
	@rsync -avzP ./$(TARGET_BENCH_STD)_stripped $(REMOTE_USER)@$(REMOTE_IP):$(REMOTE_DIR)/$(TARGET_BENCH_STD)
	@rm -f ./$(TARGET_BENCH_STD)_stripped

deploy_test: deploy_files clean_test
	@$(MAKE) clean_test $(TARGET_TEST_BIN)
	@$(STRIP) -s ./$(TARGET_TEST_BIN) -o ./$(TARGET_TEST)_stripped
	@rsync -avzP ./$(TARGET_TEST)_stripped $(REMOTE_USER)@$(REMOTE_IP):$(REMOTE_DIR)/$(TARGET_TEST)
	@rm -f ./$(TARGET_TEST)_stripped

.PHONY: run_bench run_bench_std run_bench_rvv run_bench_compare analyze_bench_compare run_test run_test_std run_test_rvv run_test_compare run_test_convolve_compare \
	generate_vec_report generate_vec_report_gaussian_cpp compile_gaussian_cpp dump_bench dump_bench_inner dump_bench_rvv dump_bench_std compare_asm clean info \
	deploy_files deploy_bench_rvv deploy_bench_std deploy_test
