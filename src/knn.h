// Copyright 2024 Iris GPU ML Pipeline
//
// GPU-accelerated K-Nearest Neighbours classification using cuBLAS + custom kernels.
// Supports three distance metrics:
//   Euclidean : ||a-b||^2 via SGEMM + norm correction
//   Manhattan : Σ|a_j - b_j| via custom L1 kernel
//   Cosine    : 1 - (a·b)/(||a||·||b||) via row-normalisation + SGEMM

#ifndef SRC_KNN_H_
#define SRC_KNN_H_

#include <vector>
#include <string>
#include "common.h"
#include <cublas_v2.h>

// DistanceMetric – selects pairwise distance computation.
enum class DistanceMetric {
  kEuclidean = 0,
  kManhattan  = 1,
  kCosine     = 2
};

// DistanceMetricName – human-readable string for logging.
inline const char* DistanceMetricName(DistanceMetric m) {
  switch (m) {
    case DistanceMetric::kEuclidean: return "euclidean";
    case DistanceMetric::kManhattan: return "manhattan";
    case DistanceMetric::kCosine:    return "cosine";
  }
  return "unknown";
}

// KNNResult – output of a KNN classification run.
struct KNNResult {
  std::vector<int> predictions;   // predicted label for each test sample
  std::vector<int> true_labels;   // ground-truth labels (copy from input)
  float  accuracy;                // classification accuracy in [0, 1]
  double gpu_ms;                  // GPU wall-clock time (milliseconds)
  DistanceMetric metric;          // metric used
};

// RunKNN – GPU K-Nearest Neighbours classifier (leave-one-out on full dataset).
//
//   metric: DistanceMetric::kEuclidean (default), kManhattan, or kCosine
void RunKNN(const float* d_data, const std::vector<int>& h_labels,
            int n_samples, int n_features, int k,
            cublasHandle_t cublas_handle,
            KNNResult* result,
            DistanceMetric metric = DistanceMetric::kEuclidean);

// PrintKNNResult – display result summary to stdout.
void PrintKNNResult(const KNNResult& result, int k);

#endif  // SRC_KNN_H_
