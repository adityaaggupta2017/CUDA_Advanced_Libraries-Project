# Makefile for GPU-Accelerated Iris ML Pipeline
# Targets CUDA 12, NVIDIA RTX A-series (sm_86); adjust ARCH as needed.

CUDA_PATH  ?= /usr/local/cuda
NVCC        = $(CUDA_PATH)/bin/nvcc
CXX         = g++

# --------------------------------------------------------------------------
# Compilation flags
# --------------------------------------------------------------------------
# sm_86 = Ampere (RTX A5000 / A6000); fall back to sm_75 for Turing.
ARCH       ?= sm_86

NVCC_FLAGS  = -std=c++14 \
              -O3 \
              -arch=$(ARCH) \
              -Xcompiler -Wall \
              -I$(CUDA_PATH)/include \
              -Isrc

LDFLAGS     = -L$(CUDA_PATH)/lib64 \
              -lcublas \
              -lcusolver \
              -lcudart \
              -Xlinker -rpath,$(CUDA_PATH)/lib64

# --------------------------------------------------------------------------
# Sources and target
# --------------------------------------------------------------------------
SRCS = src/main.cu        \
       src/data_loader.cu \
       src/normalizer.cu  \
       src/kmeans.cu      \
       src/knn.cu         \
       src/pca.cu         \
       src/silhouette.cu  \
       src/gnb.cu         \
       src/metrics.cu

TARGET = iris_gpu

# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------
.PHONY: all clean run help

all: $(TARGET)

$(TARGET): $(SRCS)
	@echo ">>> Compiling $(TARGET) with nvcc (arch=$(ARCH))..."
	$(NVCC) $(NVCC_FLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)
	@echo ">>> Build successful: ./$(TARGET)"

clean:
	@echo ">>> Cleaning build artifacts..."
	rm -f $(TARGET)

run: $(TARGET)
	@echo ">>> Running full pipeline on iris/iris.data ..."
	./$(TARGET) --data iris/iris.data \
	            --algorithm all \
	            --k 3 \
	            --knn-k 5 \
	            --components 2 \
	            --iterations 300 \
	            --seed 42 \
	            --output output

help:
	@echo "Targets:"
	@echo "  all    - build $(TARGET) (default)"
	@echo "  clean  - remove binary"
	@echo "  run    - build and run with default arguments"
	@echo "  help   - show this message"
	@echo ""
	@echo "Variables:"
	@echo "  CUDA_PATH  (default: /usr/local/cuda)"
	@echo "  ARCH       (default: sm_86)"
