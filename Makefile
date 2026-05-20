# ============================================================
# BlackFlash Makefile
# ============================================================
# 用法:
#   make            → 编译 Release 版本
#   make clean      → 清理
#   make run        → 编译并运行 (B=1 H=1 N=256 d=64)

# ---- 编译器和架构 ----
NVCC      := nvcc
ARCH      := -arch=sm_120
STD       := -std=c++17

# ---- 目录 ----
SRC_DIR   := src
INC_DIR   := include
BUILD_DIR := build

# ---- 编译选项 ----
COMMON_FLAGS := $(STD) $(ARCH) -I$(INC_DIR) --expt-relaxed-constexpr
RELEASE_FLAGS := -O3 --use_fast_math -lineinfo
DEBUG_FLAGS   := -G -g -O0

FLAGS := $(COMMON_FLAGS) $(RELEASE_FLAGS)

# ---- 目标 ----
TARGET := $(BUILD_DIR)/flash_attn

# ---- 源文件 ----
SRCS := kernel/flash_attn_mma.cu $(SRC_DIR)/main.cu

# ---- 默认参数 ----
B ?= 1
H ?= 1
N ?= 256
D ?= 64

# ============================================================
.PHONY: all clean run debug

all: $(TARGET)

$(TARGET): $(SRCS) $(INC_DIR)/*.cuh
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(FLAGS) $(SRCS) -o $(TARGET) -lcuda
	@echo "Build complete: $(TARGET)"

debug:
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(COMMON_FLAGS) $(DEBUG_FLAGS) $(SRCS) -o $(TARGET)_debug
	@echo "Debug build complete: $(TARGET)_debug"

clean:
	rm -rf $(BUILD_DIR)

run: $(TARGET)
	./$(TARGET) $(B) $(H) $(N) $(D)
