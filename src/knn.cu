// Copyright 2024 Iris GPU ML Pipeline
//
// GPU-accelerated K-NN using cuBLAS distance matrix + Thrust sort.

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
// RunKNN
// ---------------------------------------------------------------------------

void RunKNN(const float* d_data, const std::vector<int>& h_labels,
            int n_samples, int n_features, int k,
            cublasHandle_t cublas_handle,
            KNNResult* result) {
  assert(result != nullptr);
  // For leave-one-out: n_test == n_train == n_samples.
  const int n_test  = n_samples;
  const int n_train = n_samples;

  // --- Allocate device buffers ---
  float* d_dist       = nullptr;  // n_test x n_train (col-major for cuBLAS)
  float* d_sq_test    = nullptr;  // squared norms of test  rows
  float* d_sq_train   = nullptr;  // squared norms of train rows
  int*   d_labels     = nullptr;  // training labels on device
  int*   d_preds      = nullptr;  // predictions on device

  CUDA_CHECK(cudaMalloc(&d_dist,
                        (size_t)n_test * n_train * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_sq_test,  n_test  * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_sq_train, n_train * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_labels,   n_train * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_preds,    n_test  * sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_labels, h_labels.data(),
                        n_train * sizeof(int), cudaMemcpyHostToDevice));

  // --- CUDA event timing ---
  cudaEvent_t t0, t1;
  CUDA_CHECK(cudaEventCreate(&t0));
  CUDA_CHECK(cudaEventCreate(&t1));
  CUDA_CHECK(cudaEventRecord(t0));

  const int kThreads = 256;

  // --- Step 1: compute squared norms ---
  int blocks_test  = (n_test  + kThreads - 1) / kThreads;
  SquaredNormKernel<<<blocks_test, kThreads>>>(
      d_data, d_sq_test,  n_test,  n_features);
  SquaredNormKernel<<<blocks_test, kThreads>>>(
      d_data, d_sq_train, n_train, n_features);
  CUDA_CHECK(cudaGetLastError());

  // --- Step 2: compute -2 * X_test * X_train^T via SGEMM (col-major) ---
  // cuBLAS column-major SGEMM:
  //   C = alpha * A * B + beta * C
  // We want  D = -2 * X_test(n_test x n_features) * X_train^T(n_features x n_train)
  // In cuBLAS column-major: treat matrices as transposed from row-major perspective.
  // X_test  stored row-major (n_test x n_features)
  //   => as cuBLAS column-major it is (n_features x n_test), i.e. X_test^T
  // X_train stored row-major (n_train x n_features)
  //   => as cuBLAS column-major it is (n_features x n_train), i.e. X_train^T
  //
  // We compute C = X_train(col-maj, n_features x n_train) * X_test(col-maj, n_features x n_test)^T
  // => C(col-maj) = X_train * X_test^T  but we need X_test * X_train^T
  // => use Op(A)=N, Op(B)=T:
  //    C(col-maj, n_test x n_train) = -2 * X_test(col-maj, n_features x n_test)^T
  //                                      * X_train(col-maj, n_features x n_train)
  // i.e. cublasSgemm(N, T, m=n_test, n=n_train, k=n_features,
  //                  alpha=-2, X_test_colmaj, lda=n_features,
  //                  X_train_colmaj, ldb=n_features, beta=0, C, ldc=n_test)
  // But row-major X_test is the same as col-major X_test^T.
  // Row-major X_test  (n_test  x n_features) == col-major X_test^T  (n_features x n_test)
  // Row-major X_train (n_train x n_features) == col-major X_train^T (n_features x n_train)
  //
  // cublasSgemm with Op(A)=T, Op(B)=N computes A^T * B.
  // We want (using row-major stored pointers as col-major):
  //   C = (-2) * X_test_row^T)^T * X_train_row^T
  //   = (-2) * X_test_row * X_train_row^T   <- desired inner product
  // So: Op(A)=T, Op(B)=N; A=X_test (as col-major n_features x n_test), B=X_train (col-major n_features x n_train)
  // => cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
  //               m=n_test, n=n_train, k=n_features,
  //               alpha=-2, X_test_ptr, lda=n_features,
  //               X_train_ptr, ldb=n_features, beta=0, dist, ldc=n_test)
  const float alpha = -2.f, beta = 0.f;
  CUBLAS_CHECK(cublasSgemm(cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            n_test, n_train, n_features,
                            &alpha,
                            d_data, n_features,
                            d_data, n_features,
                            &beta,
                            d_dist, n_test));

  // --- Step 3: add squared norms ---
  dim3 block2d(16, 16);
  dim3 grid2d((n_train + 15) / 16, (n_test + 15) / 16);
  AddNormsKernel<<<grid2d, block2d>>>(d_dist, d_sq_test, d_sq_train,
                                       n_test, n_train);
  CUDA_CHECK(cudaGetLastError());

  // --- Step 4: set diagonal to FLT_MAX (leave-one-out: exclude self) ---
  SetDiagInfKernel<<<(n_samples + kThreads - 1) / kThreads, kThreads>>>(
      d_dist, n_samples);
  CUDA_CHECK(cudaGetLastError());

  // --- Step 5: sort each test row by distance using Thrust ---
  // Flatten dist matrix to Thrust device_vector, then sort indices per row.
  // dist is col-major (n_test x n_train): element (i,j) at j*n_test+i.
  // We want row-major access (i,j) at i*n_train+j for sorting.
  // Copy+transpose to a row-major buffer.

  thrust::device_vector<float> d_dist_row(n_test * n_train);
  thrust::device_vector<int>   d_idx_row(n_test * n_train);

  // Transpose col-major -> row-major on host for simplicity (small matrix).
  std::vector<float> h_dist_col(n_test * n_train);
  CUDA_CHECK(cudaMemcpy(h_dist_col.data(), d_dist,
                        n_test * n_train * sizeof(float),
                        cudaMemcpyDeviceToHost));

  std::vector<float> h_dist_row(n_test * n_train);
  std::vector<int>   h_idx_row(n_test * n_train);
  for (int i = 0; i < n_test; ++i)
    for (int j = 0; j < n_train; ++j)
      h_dist_row[i * n_train + j] = h_dist_col[j * n_test + i];

  // Sort indices per test sample.
  std::vector<int> h_preds(n_test);
  for (int i = 0; i < n_test; ++i) {
    std::iota(h_idx_row.begin() + i * n_train,
              h_idx_row.begin() + (i + 1) * n_train, 0);
  }

  thrust::copy(h_dist_row.begin(), h_dist_row.end(), d_dist_row.begin());
  thrust::copy(h_idx_row.begin(),  h_idx_row.end(),  d_idx_row.begin());

  for (int i = 0; i < n_test; ++i) {
    thrust::sort_by_key(
        d_dist_row.begin() + i * n_train,
        d_dist_row.begin() + (i + 1) * n_train,
        d_idx_row.begin()  + i * n_train);
  }

  // --- Step 6: majority vote ---
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
  CUDA_CHECK(cudaMemcpy(result->predictions.data(), d_preds,
                        n_test * sizeof(int), cudaMemcpyDeviceToHost));

  int correct = 0;
  for (int i = 0; i < n_test; ++i)
    if (result->predictions[i] == result->true_labels[i]) ++correct;
  result->accuracy = static_cast<float>(correct) /
                     static_cast<float>(n_test);
  result->gpu_ms = static_cast<double>(ms);

  // --- Free ---
  CUDA_CHECK(cudaFree(d_dist));
  CUDA_CHECK(cudaFree(d_sq_test));
  CUDA_CHECK(cudaFree(d_sq_train));
  CUDA_CHECK(cudaFree(d_labels));
  CUDA_CHECK(cudaFree(d_preds));
  CUDA_CHECK(cudaEventDestroy(t0));
  CUDA_CHECK(cudaEventDestroy(t1));
}

// ---------------------------------------------------------------------------
// PrintKNNResult
// ---------------------------------------------------------------------------

void PrintKNNResult(const KNNResult& result, int k) {
  printf("\n=== KNN Result (k=%d, leave-one-out) ===\n", k);
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
