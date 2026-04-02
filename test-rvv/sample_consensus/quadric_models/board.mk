REMOTE_TEST = rvv_sac_quadric_test
REMOTE_LIBS_DIR = ~/pcl-test/libs

run_test:
	LD_LIBRARY_PATH=$(REMOTE_LIBS_DIR) ./$(REMOTE_TEST)
