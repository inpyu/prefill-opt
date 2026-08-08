CXX = g++
CXXFLAGS = -std=c++11 -Werror -Wformat -Werror=format-security 
# 헤더 의존성 자동 추적.
#
# 원래 규칙이 `app.o: src/app.cpp` 뿐이라 app.hpp 를 고쳐도 재컴파일되지 않았다.
# MAX_CONTROL_BATCH_POS(app.hpp)를 64 -> 512 로 올리고 make 했는데 rc=0 이면서
# 실제로는 아무것도 다시 컴파일되지 않아, 옛 상한 그대로인 바이너리로 스윕을 돌려
# 측정 한 시간을 날렸다. -MMD 는 규칙의 $^ 를 건드리지 않으므로 안전하다.
CXXFLAGS += -MMD -MP

ifndef TERMUX_VERSION
	# aarch64에서는 -mtune=native 를 함께 주면 -march=native 가 확장한 CPU 기능셋이
	# 기본 armv8-a 로 되돌아가 __ARM_FEATURE_DOTPROD 가 사라진다.
	# 그러면 llamafile_sgemm 의 Q40xQ80 배치 경로가 통째로 비활성화되고
	# prefill 이 토큰별 matvec 폴백으로 떨어진다(배치 이득 0).
	# aarch64 에서는 arch+tune 을 동시에 세팅하는 -mcpu=native 를 쓴다.
	UNAME_M := $(shell uname -m)
	ifneq (,$(filter aarch64 arm64,$(UNAME_M)))
		CXXFLAGS += -mcpu=native
	else
		CXXFLAGS += -march=native -mtune=native
	endif
endif

# Performance-first default:
# - Always build with -O3 unless ASAN is explicitly requested.
# - This avoids accidental slow builds when DEBUG is exported in the shell.
ifdef ASAN
	CXXFLAGS += -g -fsanitize=address
else
	CXXFLAGS += -O3
endif

ifdef WVLA
	CXXFLAGS += -Wvla-extension
endif

ifdef DLLAMA_VULKAN
	CGLSLC = glslc

ifeq ($(OS),Windows_NT)
	LIBS += -L$(VK_SDK_PATH)\lib -lvulkan-1
	CXXFLAGS += -DDLLAMA_VULKAN -I$(VK_SDK_PATH)\include
else
	LIBS += -lvulkan
	CXXFLAGS += -DDLLAMA_VULKAN
endif

	DEPS += nn-vulkan.o
endif

ifeq ($(OS),Windows_NT)
    LIBS += -lws2_32
	DELETE_CMD = del /f
else
    LIBS += -lpthread
    DELETE_CMD = rm -fv
endif

.PHONY: clean dllama dllama_0 dllama_1

clean:
	$(DELETE_CMD) *.o dllama dllama-* socket-benchmark mmap-buffer-* *-test *.exe

# nn
nn-quants.o: src/nn/nn-quants.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
nn-core.o: src/nn/nn-core.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
nn-executor.o: src/nn/nn-executor.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
nn-network.o: src/nn/nn-network.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
llamafile-sgemm.o: src/nn/llamafile/sgemm.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
nn-repack.o: src/nn/nn-repack.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
nn-cpu-ops.o: src/nn/nn-cpu-ops.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
nn-cpu.o: src/nn/nn-cpu.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
nn-pipeline.o: src/nn/nn-pipeline.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
nn-cpu-test: src/nn/nn-cpu-test.cpp nn-quants.o nn-core.o nn-executor.o llamafile-sgemm.o nn-repack.o nn-cpu-ops.o nn-cpu.o
	$(CXX) $(CXXFLAGS) $(filter-out %.spv %.hpp %.h, $^) -o $@ $(LIBS)
nn-cpu-ops-test: src/nn/nn-cpu-ops-test.cpp nn-quants.o nn-core.o nn-executor.o llamafile-sgemm.o nn-repack.o nn-cpu.o
	$(CXX) $(CXXFLAGS) $(filter-out %.spv %.hpp %.h, $^) -o $@ $(LIBS)
nn-pipeline-test: src/nn/nn-pipeline-test.cpp nn-quants.o nn-core.o nn-executor.o nn-network.o nn-pipeline.o
	$(CXX) $(CXXFLAGS) $(filter-out %.spv %.hpp %.h, $^) -o $@ $(LIBS)
nn-topology-test: src/nn/nn-topology-test.cpp nn-quants.o nn-core.o
	$(CXX) $(CXXFLAGS) $(filter-out %.spv %.hpp %.h, $^) -o $@ $(LIBS)
test-pp2-graph: test-pp2-graph.cpp nn-quants.o nn-core.o nn-executor.o nn-network.o llamafile-sgemm.o nn-repack.o nn-cpu-ops.o nn-cpu.o llm.o
	$(CXX) $(CXXFLAGS) $(filter-out %.spv %.hpp %.h, $^) -o $@ $(LIBS)
nn-vulkan.o: src/nn/nn-vulkan.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

ifdef DLLAMA_VULKAN
VULKAN_SHADER_SRCS := $(wildcard src/nn/vulkan/*.comp)
VULKAN_SHADER_BINS := $(VULKAN_SHADER_SRCS:.comp=.spv)
DEPS += $(VULKAN_SHADER_BINS)

%.spv: %.comp
	$(CGLSLC) -c $< -o $@ --target-env=vulkan1.2
nn-vulkan-test: src/nn/nn-vulkan-test.cpp nn-quants.o nn-core.o nn-executor.o nn-vulkan.o ${DEPS}
	$(CXX) $(CXXFLAGS) $(filter-out %.spv %.hpp %.h, $^) -o $@ $(LIBS)
endif

# llm
tokenizer.o: src/tokenizer.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
llm.o: src/llm.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
app.o: src/app.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
tokenizer-test: src/tokenizer-test.cpp nn-quants.o nn-core.o llamafile-sgemm.o nn-repack.o nn-cpu-ops.o tokenizer.o
	$(CXX) $(CXXFLAGS) $(filter-out %.spv %.hpp %.h, $^) -o $@ $(LIBS)
dllama: src/dllama.cpp nn-quants.o nn-core.o nn-executor.o nn-network.o nn-pipeline.o llamafile-sgemm.o nn-repack.o nn-cpu-ops.o nn-cpu.o tokenizer.o llm.o app.o ${DEPS}
	$(CXX) $(CXXFLAGS) $(filter-out %.spv %.hpp %.h, $^) -o $@ $(LIBS)
dllama-api: src/dllama-api.cpp nn-quants.o nn-core.o nn-executor.o nn-network.o nn-pipeline.o llamafile-sgemm.o nn-repack.o nn-cpu-ops.o nn-cpu.o tokenizer.o llm.o app.o ${DEPS}
	$(CXX) $(CXXFLAGS) $(filter-out %.spv %.hpp %.h, $^) -o $@ $(LIBS)
dllama-gateway: src/dllama-gateway.cpp nn-quants.o nn-core.o nn-executor.o nn-network.o nn-pipeline.o llamafile-sgemm.o nn-repack.o nn-cpu-ops.o nn-cpu.o tokenizer.o llm.o app.o ${DEPS}
	$(CXX) $(CXXFLAGS) $(filter-out %.spv %.hpp %.h, $^) -o $@ $(LIBS)

# Baseline build from the pre-refactor commit for performance comparison.
# Produces ./dllama_0 in the current repo root without modifying the current checkout.
dllama_0:
	python3 scripts/build_baseline_dllama.py --out dllama_0

# Build dllama_1 from the requested comparison commit.
# Produces ./dllama_1 in the current repo root without modifying the current checkout.
dllama_1:
	python3 scripts/build_baseline_dllama.py --commit f77754d0adb40f62475cb0c8da9a4e62155377e7 --out dllama_1

# Performance build (forces O3 and disables address sanitizer/debug flags
# even if DEBUG is exported in the shell environment).
dllama-fast:
	$(MAKE) clean
	$(MAKE) dllama DEBUG= CXXFLAGS='-std=c++11 -Werror -Wformat -Werror=format-security -O3'

# 위 -MMD 가 만든 .d 파일을 읽어 헤더 변경 시 재컴파일되게 한다.
# .d 는 헤더를 '전제조건'으로 추가하므로, 규칙에서 $^ 를 그대로 쓰면 헤더까지
# 컴파일러 인자로 넘어간다. 컴파일 규칙은 $< 를, 링크 규칙은 filter-out 을 쓴다.
-include $(wildcard *.d)
