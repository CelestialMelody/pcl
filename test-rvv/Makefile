REMOTE_USER = root
REMOTE_IP   = 172.16.89.126
REMOTE_LIB_DIR = ~/pcl-test/lib
TEMP_LIB_DIR = ./slim_lib
CC = riscv64-unknown-linux-gnu-gcc
CXX = riscv64-unknown-linux-gnu-g++
WORKSPACE = /home/zoomin/codes/RISCV/workspace
PCL_INSTALL_ROOT = $(WORKSPACE)/riscv/pcl-rvv
RISCV_DEPS = $(WORKSPACE)/riscv

# 交叉编译工具链的 Sysroot 路径
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
