# =============================================================================
# 板卡侧运行 Makefile（配套 test-rvv/common/common/Makefile 的 deploy_* 目标）
#
# 说明：
# - 本 Makefile 仅用于“在板卡上运行”已部署的二进制，不负责交叉编译/部署。
# - 请先在开发机执行（路径按你的实际情况）：
#     cd test-rvv/common/common
#     make deploy_libs
#     make deploy_bench_std
#     make deploy_bench_rvv
#     make deploy_test            # 可选
#     make deploy_atan2_test       # 可选
#     make deploy_expf_test        # 可选
#     make deploy_expf_remez_vs_taylor  # 可选
#
# 用法（在板卡上，于含本 Makefile 的目录执行）：
#   make run_bench_std && make run_bench_rvv
#   make run_bench_compare
# =============================================================================

REMOTE_BENCH_STD             = bench_common_std
REMOTE_BENCH_RVV             = bench_common_rvv
REMOTE_TEST                  = test_common_app
REMOTE_ATAN2_TEST            = atan2_test
REMOTE_EXPF_TEST             = expf_test
REMOTE_EXPF_REMEZ_VS_TAYLOR  = expf_remez_vs_taylor

# 与 test-rvv/common/common/Makefile 的 REMOTE_DIR 对齐（root 默认 ~ 即 /root）
REMOTE_DIR         = /root/pcl-test/common/common
REMOTE_LIB_DIR    = /root/pcl-test/lib
REMOTE_OUTPUT_DIR  = $(REMOTE_DIR)/output

# 与开发机 script 路径对应：部署时请 rsync/scp 整个 script/ 目录
SCRIPT_DIR         = $(REMOTE_DIR)/script
BENCH_COMPARE_SCRIPT = $(SCRIPT_DIR)/analyze_bench_compare.py

# 板卡端默认用 python3 执行脚本，避免 $(PYTHON) 未定义导致把 .py 当成可执行文件
PYTHON ?= python3

# 可选：对比表同时保存到文件，例如 BENCH_COMPARE_SAVE=$(REMOTE_OUTPUT_DIR)/bench_compare.log
BENCH_COMPARE_SAVE ?=

# 传给分析脚本的 --device（可覆盖，例如 BOARD_LABEL="Milkv-Jupiter"）
BOARD_LABEL ?= Milkv-Jupiter

REMOTE_BENCH_STD_OUTPUT_FILE  = $(REMOTE_OUTPUT_DIR)/run_bench_std.log
REMOTE_BENCH_RVV_OUTPUT_FILE  = $(REMOTE_OUTPUT_DIR)/run_bench_rvv.log
REMOTE_TEST_OUTPUT_FILE       = $(REMOTE_OUTPUT_DIR)/run_test.log
REMOTE_ATAN2_OUTPUT_FILE      = $(REMOTE_OUTPUT_DIR)/run_atan2_test.log
REMOTE_EXPF_OUTPUT_FILE       = $(REMOTE_OUTPUT_DIR)/run_expf_test.log
REMOTE_EXPF_REMEZ_OUTPUT_FILE = $(REMOTE_OUTPUT_DIR)/run_expf_remez_vs_taylor.log

run_bench: run_bench_rvv

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

# 一键：先 std 再 rvv 再出对比表（日志已分别 tee）
run_bench_compare: run_bench_std run_bench_rvv analyze_bench_compare

run_test: | $(REMOTE_OUTPUT_DIR)
	@echo "[BOARD] Unit Test -> $(REMOTE_TEST_OUTPUT_FILE)"
	LD_LIBRARY_PATH=$(REMOTE_LIB_DIR):$$LD_LIBRARY_PATH \
	$(REMOTE_DIR)/$(REMOTE_TEST) 2>&1 | tee $(REMOTE_TEST_OUTPUT_FILE)

run_atan2_test: | $(REMOTE_OUTPUT_DIR)
	@echo "[BOARD] atan2_test -> $(REMOTE_ATAN2_OUTPUT_FILE)"
	LD_LIBRARY_PATH=$(REMOTE_LIB_DIR):$$LD_LIBRARY_PATH \
	$(REMOTE_DIR)/$(REMOTE_ATAN2_TEST) 2>&1 | tee $(REMOTE_ATAN2_OUTPUT_FILE)

run_expf_test: | $(REMOTE_OUTPUT_DIR)
	@echo "[BOARD] expf_test -> $(REMOTE_EXPF_OUTPUT_FILE)"
	LD_LIBRARY_PATH=$(REMOTE_LIB_DIR):$$LD_LIBRARY_PATH \
	$(REMOTE_DIR)/$(REMOTE_EXPF_TEST) 2>&1 | tee $(REMOTE_EXPF_OUTPUT_FILE)

run_expf_remez_vs_taylor: | $(REMOTE_OUTPUT_DIR)
	@echo "[BOARD] expf_remez_vs_taylor -> $(REMOTE_EXPF_REMEZ_OUTPUT_FILE)"
	LD_LIBRARY_PATH=$(REMOTE_LIB_DIR):$$LD_LIBRARY_PATH \
	$(REMOTE_DIR)/$(REMOTE_EXPF_REMEZ_VS_TAYLOR) 2>&1 | tee $(REMOTE_EXPF_REMEZ_OUTPUT_FILE)

$(REMOTE_OUTPUT_DIR):
	mkdir -p $(REMOTE_OUTPUT_DIR)

.PHONY: run_bench run_bench_std run_bench_rvv run_bench_compare run_test \
	run_atan2_test run_expf_test run_expf_remez_vs_taylor

