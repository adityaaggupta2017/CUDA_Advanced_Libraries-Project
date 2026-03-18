// Copyright 2024 Iris GPU ML Pipeline
//
// Common definitions, error-checking macros, and shared constants
// used across all modules in the GPU-accelerated Iris ML pipeline.

#ifndef SRC_COMMON_H_
#define SRC_COMMON_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

// ---------------------------------------------------------------------------
// Error-checking helpers
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error in %s:%d  %s\n",                           \
              __FILE__, __LINE__, cudaGetErrorString(err));                   \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

#define CUBLAS_CHECK(call)                                                    \
  do {                                                                        \
    cublasStatus_t status = (call);                                           \
    if (status != CUBLAS_STATUS_SUCCESS) {                                    \
      fprintf(stderr, "cuBLAS error %d in %s:%d\n",                          \
              static_cast<int>(status), __FILE__, __LINE__);                  \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

#define CUSOLVER_CHECK(call)                                                  \
  do {                                                                        \
    cusolverStatus_t status = (call);                                         \
    if (status != CUSOLVER_STATUS_SUCCESS) {                                  \
      fprintf(stderr, "cuSolver error %d in %s:%d\n",                        \
              static_cast<int>(status), __FILE__, __LINE__);                  \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

// ---------------------------------------------------------------------------
// Iris dataset constants
// ---------------------------------------------------------------------------

static const int kNumFeatures = 4;
static const int kNumClasses  = 3;
static const int kNumSamples  = 150;

// Human-readable class names (Iris-setosa=0, Iris-versicolor=1, Iris-virginica=2)
static const char* kClassNames[] = {
    "Iris-setosa", "Iris-versicolor", "Iris-virginica"
};

// Feature names
static const char* kFeatureNames[] = {
    "sepal_length_cm", "sepal_width_cm",
    "petal_length_cm", "petal_width_cm"
};

#endif  // SRC_COMMON_H_
