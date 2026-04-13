ARCH ?= riscv

# =============================================================================
#  RVV load/store micro-benchmarks (isolated)
# =============================================================================
ifeq ($(ARCH), riscv)
CC  = riscv64-unknown-linux-gnu-gcc
CXX = riscv64-unknown-linux-gnu-g++
STRIP = riscv64-unknown-linux-gnu-strip

WS = /home/zoomin/codes/RISCV/workspace
PCL_SOURCE_ROOT  = $(WS)/pcl
PCL_INSTALL_ROOT = $(WS)/riscv/pcl-rvv
RISCV_SYSROOT   := $(shell $(CC) -print-sysroot)
EIGEN_ROOT       = $(WS)/riscv/eigen-rvv
BOOST_ROOT       = $(WS)/riscv/boost

EIGEN_RVV_FLAGS ?= -DEIGEN_RISCV64_USE_RVV10 -DEIGEN_RISCV64_DEFAULT_LMUL=2 -march=rv64gcv_zvl256b -mrvv-vector-bits=zvl

LIB_DIRS_LIST = $(PCL_INSTALL_ROOT)/lib $(BOOST_ROOT)/lib
empty :=
space := $(empty) $(empty)
LIB_PATH_VAL := $(subst $(space),:,$(LIB_DIRS_LIST))

INCLUDES = \
	-I$(PCL_SOURCE_ROOT)/common/include \
	-I$(PCL_SOURCE_ROOT) \
	-I$(PCL_INSTALL_ROOT)/include/pcl-1.15 \
	-I$(EIGEN_ROOT)/include/eigen3 \
	-I$(BOOST_ROOT)/include

CXXFLAGS = -std=c++17 -O3 -g \
	$(EIGEN_RVV_FLAGS) \
	-mabi=lp64d \
	-D__RVV10__ \
	-DPCL_NO_PRECOMPILE \
	$(INCLUDES)

LDFLAGS = $(foreach dir,$(LIB_DIRS_LIST),-L$(dir) -Wl,-rpath-link=$(dir))
LIBS = -lpcl_common -lm

RUN_CMD = LD_LIBRARY_PATH=$(LIB_PATH_VAL):$$LD_LIBRARY_PATH qemu-riscv64 -L $(RISCV_SYSROOT) -cpu rv64,v=true,vlen=256,elen=64
endif

BUILD_DIR = build/$(ARCH)
OUTPUT_DIR = output
OUTPUT_DIR_QEMU = $(OUTPUT_DIR)/qemu

TARGET_LOAD = bench_rvv_load_compare
TARGET_STORE = bench_rvv_store_compare

TARGET_LOAD_BIN  = $(BUILD_DIR)/$(TARGET_LOAD)
TARGET_STORE_BIN = $(BUILD_DIR)/$(TARGET_STORE)

# Default workload (tuned for board interactivity)
N_POINTS ?= 262144
ITERS    ?= 50
WARMUP   ?= 5

all: $(TARGET_LOAD_BIN) $(TARGET_STORE_BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(OUTPUT_DIR_QEMU):
	mkdir -p $(OUTPUT_DIR_QEMU)

$(TARGET_LOAD_BIN): bench_rvv_load_compare.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) $(LIBS) -o $@

$(TARGET_STORE_BIN): bench_rvv_store_compare.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) $(LIBS) -o $@

run_load: $(TARGET_LOAD_BIN) | $(OUTPUT_DIR_QEMU)
	$(RUN_CMD) ./$(TARGET_LOAD_BIN) $(N_POINTS) $(ITERS) $(WARMUP) 2>&1 | tee $(OUTPUT_DIR_QEMU)/run_load.log

run_store: $(TARGET_STORE_BIN) | $(OUTPUT_DIR_QEMU)
	$(RUN_CMD) ./$(TARGET_STORE_BIN) $(N_POINTS) $(ITERS) $(WARMUP) 2>&1 | tee $(OUTPUT_DIR_QEMU)/run_store.log

clean:
	rm -rf build

# =============================================================================
# Remote Deployment (board)
# =============================================================================
REMOTE_USER = root
REMOTE_IP   = 172.16.89.126
REMOTE_DIR  = ~/pcl-test/rvv/load_store

deploy_load: $(TARGET_LOAD_BIN)
	@echo "[DEPLOY] $(TARGET_LOAD) -> $(REMOTE_DIR)"
	@ssh $(REMOTE_USER)@$(REMOTE_IP) "mkdir -p $(REMOTE_DIR)"
	@$(STRIP) -s ./$(TARGET_LOAD_BIN) -o ./$(TARGET_LOAD)_stripped
	@rsync -avzP ./$(TARGET_LOAD)_stripped $(REMOTE_USER)@$(REMOTE_IP):$(REMOTE_DIR)/$(TARGET_LOAD)
	@rm -f ./$(TARGET_LOAD)_stripped

deploy_store: $(TARGET_STORE_BIN)
	@echo "[DEPLOY] $(TARGET_STORE) -> $(REMOTE_DIR)"
	@ssh $(REMOTE_USER)@$(REMOTE_IP) "mkdir -p $(REMOTE_DIR)"
	@$(STRIP) -s ./$(TARGET_STORE_BIN) -o ./$(TARGET_STORE)_stripped
	@rsync -avzP ./$(TARGET_STORE)_stripped $(REMOTE_USER)@$(REMOTE_IP):$(REMOTE_DIR)/$(TARGET_STORE)
	@rm -f ./$(TARGET_STORE)_stripped

.PHONY: all run_load run_store clean deploy_load deploy_store

