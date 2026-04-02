REMOTE_TEST = rvv_sac_plane_test
REMOTE_BENCH = bench_sac_normal_plane
REMOTE_PCD_FILE = /root/pcl-test/sample_consensus/plane_models/sac_plane_test.pcd
REMOTE_LIBS_DIR = /root/pcl-test/libs

run_test:
	LD_LIBRARY_PATH=$(REMOTE_LIBS_DIR) ./$(REMOTE_TEST) $(REMOTE_PCD_FILE)

run_bench:
	LD_LIBRARY_PATH=$(REMOTE_LIBS_DIR) ./$(REMOTE_BENCH) $(REMOTE_PCD_FILE)
