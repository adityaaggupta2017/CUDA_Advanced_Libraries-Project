// Copyright 2024 Iris GPU ML Pipeline
//
// GPU-accelerated silhouette coefficient.
// For each sample i:
//   a(i) = mean distance to all other samples in the same cluster
//   b(i) = min over other clusters of mean distance to that cluster
//   s(i) = (b(i) - a(i)) / max(a(i), b(i))   in [-1, 1]
//
// The pairwise Euclidean distance matrix is computed via cuBLAS SGEMM;
// a(i) and b(i) are computed by custom CUDA kernels.

#ifndef SRC_SILHOUETTE_H_
#define SRC_SILHOUETTE_H_

#include <vector>
#include "common.h"
#include <cublas_v2.h>

struct SilhouetteResult {
  float              mean_score;     // scalar mean in [-1, 1]
  std::vector<float> per_sample;     // s(i) for each sample (host)
  std::vector<float> per_cluster;    // mean s(i) per cluster (host)
  double             gpu_ms;
};

// ComputeSilhouette – computes the silhouette coefficient on the GPU.
//
//   d_data     : device – row-major normalised feature matrix (n_samples x n_features)
//   d_labels   : device – integer cluster assignments [n_samples]
//   n_samples  : number of rows
//   n_features : number of feature dimensions
//   k          : number of clusters
//   cublas_handle : already-created cuBLAS handle
//   result     : populated on return
void ComputeSilhouette(const float* d_data,
                       const int*   d_labels,
                       int n_samples, int n_features, int k,
                       cublasHandle_t cublas_handle,
                       SilhouetteResult* result);

void PrintSilhouetteResult(const SilhouetteResult& result, int k);

#endif  // SRC_SILHOUETTE_H_
