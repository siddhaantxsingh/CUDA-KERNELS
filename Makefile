CXX ?= g++
CXXFLAGS ?= -O2 -std=c++20 -pthread -Wall -Wextra -fno-strict-aliasing -Iinclude
BUILD := build

.PHONY: test analyze ptx clean gpu-bench
$(BUILD)/test_kernels: tests/test_kernels.cpp include/kc/*.h include/kc/*.cuh
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@
$(BUILD)/test_tracer: tests/test_tracer.cpp include/kc/*.h
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@
test: $(BUILD)/test_kernels $(BUILD)/test_tracer
	./$(BUILD)/test_tracer
	./$(BUILD)/test_kernels
clean:
	rm -rf $(BUILD)
$(BUILD)/analyze: tools/analyze.cpp include/kc/*.h include/kc/*.cuh
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@
analyze: $(BUILD)/analyze
	./$(BUILD)/analyze

# ---- device builds (clang + a CUDA toolkit root; tools/fetch_cuda.sh makes one from PyPI wheels)
CUDA_ROOT ?= build/cuda-root
CUDA_ARCH ?= sm_80
GCC_DIR ?= /usr/lib/gcc/x86_64-linux-gnu/13
HOSTFLAGS := $(if $(wildcard $(GCC_DIR)),--gcc-install-dir=$(GCC_DIR))
ptx:
	CUDA_ROOT=$(CUDA_ROOT) tools/device_build.sh build/device sm_80 sm_89 sm_90
	python3 tools/resources.py build/device > build/resources.json
	python3 tools/sass_check.py build/device sm_80 sm_89 sm_90 > build/sass.json
gpu-bench: bench/gpu_bench.cu include/kc/*.h include/kc/*.cuh
	@mkdir -p $(BUILD)
	clang++ -x cuda --cuda-path=$(CUDA_ROOT) --cuda-gpu-arch=$(CUDA_ARCH) -O3 -std=c++17 -Wno-unknown-cuda-version $(HOSTFLAGS) -Iinclude \
	  $< -o $(BUILD)/gpu_bench -L$(CUDA_ROOT)/lib64 -l:libcudart.so.12 -Wl,-rpath,$(abspath $(CUDA_ROOT))/lib64
