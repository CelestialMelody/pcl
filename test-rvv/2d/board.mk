# =============================================================================
# 板卡侧运行脚本（部署路径与 test-rvv/2d/Makefile 中 REMOTE_* 一致）
#
# 使用前请保证：
#   1) 开发机已 make deploy_bench_std / deploy_bench_rvv 等，把二进制同步到 $(REMOTE_DIR)
#   2) 将本仓库 test-rvv/script/analyze_bench_compare.py 同步到板卡：$(REMOTE_DIR)/script/
#      （可选建 .venv 仅用 python3 亦可）
#   3) mkdir -p $(REMOTE_OUTPUT_DIR) 由下方 run_bench_* 自动创建
#
# 用法（在板卡上，于含本 Makefile 的目录执行，或复制为 /root/pcl-test/2d/Makefile.run）:
#   make run_bench_std && make run_bench_rvv && make analyze_bench_compare
#   make run_bench_compare          # 依次：std -> rvv -> 对比表
# =============================================================================

REMOTE_TEST                  = test_2d_app
REMOTE_BENCH_STD             = bench_2d_app_std
REMOTE_BENCH_RVV             = bench_2d_app_rvv
REMOTE_ATAN2_TEST            = atan2_test
REMOTE_EXPF_TEST             = expf_test
REMOTE_EXPF_REMEZ_VS_TAYLOR  = expf_remez_vs_taylor
REMOTE_EDGE_STORE_TEST       = test_edge_store_bench

REMOTE_DIR         = /root/pcl-test/2d
REMOTE_PCD_DIR    = $(REMOTE_DIR)/pcd
REMOTE_LIB_DIR    = /root/pcl-test/lib
REMOTE_OUTPUT_DIR  = $(REMOTE_DIR)/output
# 与开发机 script 路径对应：部署时请 rsync/scp 整个 script/ 目录
SCRIPT_DIR         = $(REMOTE_DIR)/script
BENCH_COMPARE_SCRIPT = $(SCRIPT_DIR)/analyze_bench_compare.py

REMOTE_BENCH_STD_OUTPUT_FILE  = $(REMOTE_OUTPUT_DIR)/run_bench_std.log
REMOTE_BENCH_RVV_OUTPUT_FILE  = $(REMOTE_OUTPUT_DIR)/run_bench_rvv.log
# 可选：对比表同时保存到文件，例如 BENCH_COMPARE_SAVE=$(REMOTE_OUTPUT_DIR)/bench_compare.log
BENCH_COMPARE_SAVE ?=

# 优先使用板卡上 $(REMOTE_DIR)/.venv/bin/python，否则 python3
VENV_PYTHON := $(abspath $(REMOTE_DIR)/.venv/bin/python)
PYTHON      := $(shell test -x '$(VENV_PYTHON)' && echo '$(VENV_PYTHON)' || echo python3)

# 传给分析脚本的 --device（可覆盖，例如 BOARD_LABEL="Milkv-Jupiter"）
BOARD_LABEL ?= Milkv-Jupiter

# 命令行参数按位置固定，顺序不能变
REMOTE_PCD_FILES = \
	$(REMOTE_PCD_DIR)/lena.pcd \
	$(REMOTE_PCD_DIR)/gauss_smooth.pcd \
	$(REMOTE_PCD_DIR)/erosion.pcd \
	$(REMOTE_PCD_DIR)/dilation.pcd \
	$(REMOTE_PCD_DIR)/opening.pcd \
	$(REMOTE_PCD_DIR)/closing.pcd \
	$(REMOTE_PCD_DIR)/erosion_binary.pcd \
	$(REMOTE_PCD_DIR)/dilation_binary.pcd \
	$(REMOTE_PCD_DIR)/opening_binary.pcd \
	$(REMOTE_PCD_DIR)/closing_binary.pcd \
	$(REMOTE_PCD_DIR)/canny.pcd

run_test:
	LD_LIBRARY_PATH=$(REMOTE_LIB_DIR):$$LD_LIBRARY_PATH \
	$(REMOTE_DIR)/$(REMOTE_TEST) $(REMOTE_PCD_FILES)

# 默认跑 RVV benchmark（无 tee）
run_bench:
	$(MAKE) run_bench_rvv

run_bench_std: | $(REMOTE_OUTPUT_DIR)
	@echo "[BOARD] Benchmark Std -> $(REMOTE_BENCH_STD_OUTPUT_FILE)"
	LD_LIBRARY_PATH=$(REMOTE_LIB_DIR):$$LD_LIBRARY_PATH \
	$(REMOTE_DIR)/$(REMOTE_BENCH_STD) 2>&1 | tee $(REMOTE_BENCH_STD_OUTPUT_FILE)

run_bench_rvv: | $(REMOTE_OUTPUT_DIR)
	@echo "[BOARD] Benchmark RVV -> $(REMOTE_BENCH_RVV_OUTPUT_FILE)"
	LD_LIBRARY_PATH=$(REMOTE_LIB_DIR):$$LD_LIBRARY_PATH \
	$(REMOTE_DIR)/$(REMOTE_BENCH_RVV) 2>&1 | tee $(REMOTE_BENCH_RVV_OUTPUT_FILE)

$(REMOTE_OUTPUT_DIR):
	mkdir -p $(REMOTE_OUTPUT_DIR)

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

run_bench_store_compare:
	LD_LIBRARY_PATH=$(REMOTE_LIB_DIR):$$LD_LIBRARY_PATH \
	$(REMOTE_DIR)/$(REMOTE_EDGE_STORE_TEST)

run_atan2_test:
	LD_LIBRARY_PATH=$(REMOTE_LIB_DIR):$$LD_LIBRARY_PATH \
	$(REMOTE_DIR)/$(REMOTE_ATAN2_TEST)

run_expf_test:
	LD_LIBRARY_PATH=$(REMOTE_LIB_DIR):$$LD_LIBRARY_PATH \
	$(REMOTE_DIR)/$(REMOTE_EXPF_TEST)

run_expf_remez_vs_taylor:
	LD_LIBRARY_PATH=$(REMOTE_LIB_DIR):$$LD_LIBRARY_PATH \
	$(REMOTE_DIR)/$(REMOTE_EXPF_REMEZ_VS_TAYLOR)

.PHONY: run_test run_bench run_bench_std run_bench_rvv \
	analyze_bench_compare run_bench_compare \
	run_atan2_test run_expf_test run_expf_remez_vs_taylor \
	run_bench_store_compare

