# =============================================================================
# Board-side runner for rvv/load_store micro-benchmarks
# =============================================================================

REMOTE_DIR     = /root/pcl-test/rvv/load_store
REMOTE_LIB_DIR = /root/pcl-test/lib
OUTPUT_DIR     = $(REMOTE_DIR)/output

RUN_LOAD_LOG  = $(OUTPUT_DIR)/run_load.log
RUN_STORE_LOG = $(OUTPUT_DIR)/run_store.log

BENCH_LOAD  = bench_rvv_load_compare
BENCH_STORE = bench_rvv_store_compare

# Default workload (tuned for board interactivity)
N_POINTS ?= 262144
ITERS    ?= 50
WARMUP   ?= 5

run_load: | $(OUTPUT_DIR)
	LD_LIBRARY_PATH=$(REMOTE_LIB_DIR):$$LD_LIBRARY_PATH \
	$(REMOTE_DIR)/$(BENCH_LOAD) $(N_POINTS) $(ITERS) $(WARMUP) 2>&1 | tee $(RUN_LOAD_LOG)

run_store: | $(OUTPUT_DIR)
	LD_LIBRARY_PATH=$(REMOTE_LIB_DIR):$$LD_LIBRARY_PATH \
	$(REMOTE_DIR)/$(BENCH_STORE) $(N_POINTS) $(ITERS) $(WARMUP) 2>&1 | tee $(RUN_STORE_LOG)

$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

.PHONY: run_load run_store

