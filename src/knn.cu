// Copyright 2024 Iris GPU ML Pipeline
//
// GPU-accelerated K-NN using cuBLAS (Euclidean/Cosine) and custom kernels
// (Manhattan).  Distance matrix is always produced row-major before sorting.

#include "knn.h"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>

// ---------------------------------------------------------------------------
// Device kernel: add squared norms to the distance matrix.
//
// dist[i,j] = -2 * X_test[i] . X_train[j]  (from SGEMM)
//           + ||X_test[i]||^2  + ||X_train[j]||^2
//
// The SGEMM output is stored column-major (cuBLAS convention);
// this kernel adds the row/column squared-norm corrections.
// ---------------------------------------------------------------------------

__global__ void AddNormsKernel(float* dist_col_major,
                                const float* __restrict__ sq_norm_test,
                                const float* __restrict__ sq_norm_train,
                                int n_test, int n_train) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;  // test sample index
  int j = blockIdx.x * blockDim.x + threadIdx.x;  // train sample index
  if (i >= n_test || j >= n_train) return;

  // cuBLAS SGEMM result is column-major: element (i, j) is at j*n_test + i
  dist_col_major[j * n_test + i] += sq_norm_test[i] + sq_norm_train[j];
}

// ---------------------------------------------------------------------------
// Device kernel: set diagonal to FLT_MAX (exclude self in leave-one-out)
// ---------------------------------------------------------------------------

__global__ void SetDiagInfKernel(float* dist_col_major, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dist_col_major[i * n + i] = FLT_MAX;
}

// ---------------------------------------------------------------------------
// Device kernel: squared L2 norms of each row
// ---------------------------------------------------------------------------

__global__ void SquaredNormKernel(const float* __restrict__ data,
                                   float* __restrict__ sq_norm,
                                   int n_samples, int n_features) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_samples) return;
  float s = 0.f;
  for (int f = 0; f < n_features; ++f) {
    float v = data[i * n_features + f];
    s += v * v;
  }
  sq_norm[i] = s;
}

// ---------------------------------------------------------------------------
// Device kernel: majority vote among k nearest neighbours
// ---------------------------------------------------------------------------

__global__ void MajorityVoteKernel(const int*   __restrict__ sorted_idx,
                                    const int*   __restrict__ train_labels,
                                    int* __restrict__ predictions,
                                    int n_test, int n_train, int k,
                                    int n_classes) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_test) return;

  // Small vote array in registers (assumes n_classes <= 32).
  int votes[32] = {};
  for (int nb = 0; nb < k; ++nb) {
    int train_idx = sorted_idx[i * n_train + nb];
    int lbl       = train_labels[train_idx];
    if (lbl >= 0 && lbl < n_classes) votes[lbl]++;
  }
  int best_lbl = 0, best_cnt = -1;
  for (int c = 0; c < n_classes; ++c) {
    if (votes[c] > best_cnt) { best_cnt = votes[c]; best_lbl = c; }
  }
  predictions[i] = best_lbl;
}

// ---------------------------------------------------------------------------
// Manhattan L1 distance kernel: dist_row[i,j] = Σ|x[i,f] - x[j,f]|
// Output is row-major (n x n).
// ---------------------------------------------------------------------------

__global__ void L1DistanceKernel(const float* __restrict__ data,
                                  float* __restrict__ dist_row,
                                  int n_samples, int n_features) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_samples || j >= n_samples) return;
  float d = 0.f;
  for (int f = 0; f < n_features; ++f)
    d += fabsf(data[i * n_features + f] - data[j * n_features + f]);
  dist_row[i * n_samples + j] = d;
}

// ---------------------------------------------------------------------------
// Row L2-normalisation kernel for Cosine distance.
// Copies src -> dst, dividing each row by its L2 norm.
// ---------------------------------------------------------------------------

__global__ void RowNormalizeKernel(const float* __restrict__ src,
                                    float* __restrict__ dst,
                                    int n_samples, int n_features) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_samples) return;
  float norm = 0.f;
  for (int f = 0; f < n_features; ++f) {
    float v = src[i * n_features + f];
    norm += v * v;
  }
  norm = sqrtf(fmaxf(norm, 1e-12f));
  for (int f = 0; f < n_features; ++f)
    dst[i * n_features + f] = src[i * n_features + f] / norm;
}

// ---------------------------------------------------------------------------
// Kernel: elementwise add scalar to a row-major matrix.
// Used to convert cosine similarity -> cosine distance via 1 - similarity.
// ---------------------------------------------------------------------------

__global__ void AddScalarKernel(float* mat, float scalar, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) mat[i] += scalar;
}

// ---------------------------------------------------------------------------
// RunKNN
// ---------------------------------------------------------------------------

void RunKNN(const float* d_data, const std::vector<int>& h_labels,
            int n_samples, int n_features, int k,
            cublasHandle_t cublas_handle,
            KNNResult* result,
            DistanceMetric metric) {
  assert(result != nullptr);
  const int n_test  = n_samples;
  const int n_train = n_samples;

  // --- Allocate common device buffers ---
  int*   d_labels = nullptr;
  int*   d_preds  = nullptr;
  CUDA_CHECK(cudaMalloc(&d_labels, n_train  * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_preds,  n_test   * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_labels, h_labels.data(),
                        n_train * sizeof(int), cudaMemcpyHostToDevice));

  // --- CUDA event timing ---
  cudaEvent_t t0, t1;
  CUDA_CHECK(cudaEventCreate(&t0));
  CUDA_CHECK(cudaEventCreate(&t1));
  CUDA_CHECK(cudaEventRecord(t0));

  const int kThreads = 256;
  int blocks_n = (n_samples + kThreads - 1) / kThreads;

  // --- Build row-major distance matrix (n x n) depending on metric ---
  // All three paths produce h_dist_row: row-major float[n x n].

  std::vector<float> h_dist_row(n_test * n_train);

  if (metric == DistanceMetric::kEuclidean) {
    // Euclidean: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b  (via SGEMM)
    float* d_dist_col = nullptr;  // col-major from cuBLAS
    float* d_sq_norm  = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dist_col,
                          (size_t)n_test * n_train * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sq_norm, n_samples * sizeof(float)));

    SquaredNormKernel<<<blocks_n, kThreads>>>(
        d_data, d_sq_norm, n_samples, n_features);
    CUDA_CHECK(cudaGetLastError());

    const float alpha = -2.f, beta = 0.f;
    CUBLAS_CHECK(cublasSgemm(cublas_handle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              n_test, n_train, n_features,
                              &alpha,
                              d_data, n_features,
                              d_data, n_features,
                              &beta,
                              d_dist_col, n_test));

    dim3 b2d(16, 16);
    dim3 g2d((n_train + 15) / 16, (n_test + 15) / 16);
    AddNormsKernel<<<g2d, b2d>>>(d_dist_col, d_sq_norm, d_sq_norm,
                                  n_test, n_train);
    CUDA_CHECK(cudaGetLastError());

    // Transpose col-major -> row-major.
    std::vector<float> h_col(n_test * n_train);
    CUDA_CHECK(cudaMemcpy(h_col.data(), d_dist_col,
                          n_test * n_train * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < n_test; ++i)
      for (int j = 0; j < n_train; ++j)
        h_dist_row[i * n_train + j] = h_col[j * n_test + i];

    CUDA_CHECK(cudaFree(d_dist_col));
    CUDA_CHECK(cudaFree(d_sq_norm));

  } else if (metric == DistanceMetric::kManhattan) {
    // Manhattan: Σ|a_j - b_j|  via L1DistanceKernel (row-major output)
    float* d_dist_rowdev = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dist_rowdev,
                          (size_t)n_test * n_train * sizeof(float)));

    dim3 b2d(16, 16);
    dim3 g2d((n_train + 15) / 16, (n_test + 15) / 16);
    L1DistanceKernel<<<g2d, b2d>>>(d_data, d_dist_rowdev,
                                    n_samples, n_features);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_dist_row.data(), d_dist_rowdev,
                          n_test * n_train * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_dist_rowdev));

  } else {  // kCosine
    // Cosine distance = 1 - cosine_similarity
    // Step a: row-normalise X -> X_norm
    float* d_X_norm = nullptr;
    CUDA_CHECK(cudaMalloc(&d_X_norm, n_samples * n_features * sizeof(float)));
    RowNormalizeKernel<<<blocks_n, kThreads>>>(
        d_data, d_X_norm, n_samples, n_features);
    CUDA_CHECK(cudaGetLastError());

    // Step b: dot product matrix S = X_norm * X_norm^T via SGEMM
    // S(col-major, n x n) = X_norm(col-maj)^T * X_norm(col-maj)
    float* d_sim_col = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sim_col,
                          (size_t)n_samples * n_samples * sizeof(float)));

    const float alpha = 1.f, beta = 0.f;
    CUBLAS_CHECK(cublasSgemm(cublas_handle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              n_test, n_train, n_features,
                              &alpha,
                              d_X_norm, n_features,
                              d_X_norm, n_features,
                              &beta,
                              d_sim_col, n_test));

    // Step c: cosine dist = 1 - sim  (apply after transpose)
    std::vector<float> h_sim_col(n_samples * n_samples);
    CUDA_CHECK(cudaMemcpy(h_sim_col.data(), d_sim_col,
                          n_samples * n_samples * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < n_test; ++i)
      for (int j = 0; j < n_train; ++j)
        h_dist_row[i * n_train + j] = 1.f - h_sim_col[j * n_test + i];

    CUDA_CHECK(cudaFree(d_X_norm));
    CUDA_CHECK(cudaFree(d_sim_col));
  }

  // --- Set diagonal to FLT_MAX (leave-one-out: exclude self) ---
  for (int i = 0; i < n_samples; ++i)
    h_dist_row[i * n_samples + i] = FLT_MAX;

  // --- Sort each row by distance using Thrust ---
  thrust::device_vector<float> d_dist_row_dev(h_dist_row.begin(),
                                               h_dist_row.end());
  std::vector<int> h_idx_row(n_test * n_train);
  for (int i = 0; i < n_test; ++i)
    std::iota(h_idx_row.begin() + i * n_train,
              h_idx_row.begin() + (i + 1) * n_train, 0);

  thrust::device_vector<int> d_idx_row(h_idx_row.begin(), h_idx_row.end());

  for (int i = 0; i < n_test; ++i) {
    thrust::sort_by_key(
        d_dist_row_dev.begin() + i * n_train,
        d_dist_row_dev.begin() + (i + 1) * n_train,
        d_idx_row.begin()  + i * n_train);
  }

  // --- Majority vote ---
  int* d_sorted_idx_raw = thrust::raw_pointer_cast(d_idx_row.data());
  int blocks_vote = (n_test + kThreads - 1) / kThreads;
  MajorityVoteKernel<<<blocks_vote, kThreads>>>(
      d_sorted_idx_raw, d_labels, d_preds,
      n_test, n_train, k, kNumClasses);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(t1));
  CUDA_CHECK(cudaEventSynchronize(t1));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

  // --- Collect results ---
  result->predictions.resize(n_test);
  result->true_labels = h_labels;
  result->metric      = metric;
  CUDA_CHECK(cudaMemcpy(result->predictions.data(), d_preds,
                        n_test * sizeof(int), cudaMemcpyDeviceToHost));

  int correct = 0;
  for (int i = 0; i < n_test; ++i)
    if (result->predictions[i] == result->true_labels[i]) ++correct;
  result->accuracy = static_cast<float>(correct) /
                     static_cast<float>(n_test);
  result->gpu_ms = static_cast<double>(ms);

  // --- Free ---
  CUDA_CHECK(cudaFree(d_labels));
  CUDA_CHECK(cudaFree(d_preds));
  CUDA_CHECK(cudaEventDestroy(t0));
  CUDA_CHECK(cudaEventDestroy(t1));
}

// ---------------------------------------------------------------------------
// PrintKNNResult
// ---------------------------------------------------------------------------

void PrintKNNResult(const KNNResult& result, int k) {
  printf("\n=== KNN Result (k=%d, metric=%s, leave-one-out) ===\n",
         k, DistanceMetricName(result.metric));
  printf("  Accuracy  : %.2f%%\n", result.accuracy * 100.f);
  printf("  GPU time  : %.3f ms\n", result.gpu_ms);

  // Per-class accuracy.
  std::vector<int> per_class_correct(kNumClasses, 0);
  std::vector<int> per_class_total(kNumClasses, 0);
  for (int i = 0; i < static_cast<int>(result.true_labels.size()); ++i) {
    int gt  = result.true_labels[i];
    int pred = result.predictions[i];
    per_class_total[gt]++;
    if (pred == gt) per_class_correct[gt]++;
  }
  printf("  Per-class accuracy:\n");
  for (int c = 0; c < kNumClasses; ++c) {
    float acc = (per_class_total[c] > 0)
                    ? static_cast<float>(per_class_correct[c]) /
                          per_class_total[c] * 100.f
                    : 0.f;
    printf("    %-20s  %d/%d  (%.1f%%)\n",
           kClassNames[c], per_class_correct[c], per_class_total[c], acc);
  }
  printf("=========================================\n\n");
}
