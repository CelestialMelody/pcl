# =============================================================================
# 板卡侧运行 Makefile（配套 test-rvv/common/centroid/Makefile 的 deploy_* 目标）
# =============================================================================

REMOTE_BENCH_STD = bench_centroid_std
REMOTE_BENCH_RVV = bench_centroid_rvv
REMOTE_TEST      = test_centroid

REMOTE_DIR        = /root/pcl-test/common/centroid
REMOTE_LIB_DIR    = /root/pcl-test/lib
REMOTE_OUTPUT_DIR = $(REMOTE_DIR)/output

SCRIPT_DIR           = $(REMOTE_DIR)/script
BENCH_COMPARE_SCRIPT = $(SCRIPT_DIR)/analyze_bench_compare.py

PYTHON ?= python3

BOARD_LABEL ?= Milkv-Jupiter
BENCH_COMPARE_SAVE ?=

REMOTE_BENCH_STD_OUTPUT_FILE = $(REMOTE_OUTPUT_DIR)/run_bench_std.log
REMOTE_BENCH_RVV_OUTPUT_FILE = $(REMOTE_OUTPUT_DIR)/run_bench_rvv.log
REMOTE_TEST_OUTPUT_FILE      = $(REMOTE_OUTPUT_DIR)/run_test.log
# 与开发机 Makefile 一致：单测 main 要求 argv[1] 为 bun0.pcd（deploy_test 会同步到 $(REMOTE_DIR)/pcd/）
REMOTE_TEST_BUN0_PCD         = $(REMOTE_DIR)/pcd/bun0.pcd

run_bench_std: | $(REMOTE_OUTPUT_DIR)
	@echo "[BOARD] Benchmark Std -> $(REMOTE_BENCH_STD_OUTPUT_FILE)"
	LD_LIBRARY_PATH=$(REMOTE_LIB_DIR):$$LD_LIBRARY_PATH \
	$(REMOTE_DIR)/$(REMOTE_BENCH_STD) 2>&1 | tee $(REMOTE_BENCH_STD_OUTPUT_FILE)

run_bench_rvv: | $(REMOTE_OUTPUT_DIR)
	@echo "[BOARD] Benchmark RVV -> $(REMOTE_BENCH_RVV_OUTPUT_FILE)"
	LD_LIBRARY_PATH=$(REMOTE_LIB_DIR):$$LD_LIBRARY_PATH \
	$(REMOTE_DIR)/$(REMOTE_BENCH_RVV) 2>&1 | tee $(REMOTE_BENCH_RVV_OUTPUT_FILE)

analyze_bench_compare:
	@test -f '$(BENCH_COMPARE_SCRIPT)' || (echo "缺少 $(BENCH_COMPARE_SCRIPT)，请将开发机 test-rvv/script/analyze_bench_compare.py 同步到板卡 $(SCRIPT_DIR)/" >&2; exit 1)
	$(PYTHON) '$(BENCH_COMPARE_SCRIPT)' \
		--std-log $(REMOTE_BENCH_STD_OUTPUT_FILE) \
		--rvv-log $(REMOTE_BENCH_RVV_OUTPUT_FILE) \
		--device "$(BOARD_LABEL)" \
		--vlen-desc "see SoC / ELF (board)" \
		$(if $(BENCH_COMPARE_SAVE),| tee $(BENCH_COMPARE_SAVE),)

run_bench_compare: run_bench_std run_bench_rvv analyze_bench_compare

run_test: | $(REMOTE_OUTPUT_DIR)
	@echo "[BOARD] Unit Test -> $(REMOTE_TEST_OUTPUT_FILE)"
	LD_LIBRARY_PATH=$(REMOTE_LIB_DIR):$$LD_LIBRARY_PATH \
	$(REMOTE_DIR)/$(REMOTE_TEST) "$(REMOTE_TEST_BUN0_PCD)" 2>&1 | tee $(REMOTE_TEST_OUTPUT_FILE)

$(REMOTE_OUTPUT_DIR):
	mkdir -p $(REMOTE_OUTPUT_DIR)

.PHONY: run_bench_std run_bench_rvv run_bench_compare analyze_bench_compare run_test

