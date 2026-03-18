// Copyright 2024 Iris GPU ML Pipeline
//
// GPU-accelerated PCA using cuBLAS (covariance) + cuSolver (eigendecomposition).

#include "pca.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#include <thrust/device_vector.h>

// ---------------------------------------------------------------------------
// Device kernel: subtract per-column mean to centre the data matrix.
// ---------------------------------------------------------------------------

__global__ void SubtractMeanKernel(float* data, const float* col_mean,
                                    int n_samples, int n_features) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n_samples && col < n_features)
    data[row * n_features + col] -= col_mean[col];
}

// ---------------------------------------------------------------------------
// RunPCA
// ---------------------------------------------------------------------------

void RunPCA(const float* d_data, int n_samples, int n_features,
            int n_components,
            cublasHandle_t cublas_handle,
            cusolverDnHandle_t cusolver_handle,
            PCAResult* result) {
  assert(result != nullptr);
  assert(n_components <= n_features);

  // --- CUDA event timing ---
  cudaEvent_t t0, t1;
  CUDA_CHECK(cudaEventCreate(&t0));
  CUDA_CHECK(cudaEventCreate(&t1));
  CUDA_CHECK(cudaEventRecord(t0));

  // --- Step 1: copy data and centre columns ---
  float* d_X = nullptr;
  CUDA_CHECK(cudaMalloc(&d_X, n_samples * n_features * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_X, d_data,
                        n_samples * n_features * sizeof(float),
                        cudaMemcpyDeviceToDevice));

  // Compute per-feature mean on host (dataset is small: 150 x 4).
  std::vector<float> h_data_copy(n_samples * n_features);
  CUDA_CHECK(cudaMemcpy(h_data_copy.data(), d_X,
                        n_samples * n_features * sizeof(float),
                        cudaMemcpyDeviceToHost));
  std::vector<float> h_mean(n_features, 0.f);
  for (int col = 0; col < n_features; ++col) {
    float sum = 0.f;
    for (int row = 0; row < n_samples; ++row)
      sum += h_data_copy[row * n_features + col];
    h_mean[col] = sum / static_cast<float>(n_samples);
  }
  float* d_mean = nullptr;
  CUDA_CHECK(cudaMalloc(&d_mean, n_features * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_mean, h_mean.data(),
                        n_features * sizeof(float), cudaMemcpyHostToDevice));

  dim3 block2d(16, 16);
  dim3 grid2d((n_features + 15) / 16, (n_samples + 15) / 16);
  SubtractMeanKernel<<<grid2d, block2d>>>(d_X, d_mean, n_samples, n_features);
  CUDA_CHECK(cudaGetLastError());

  // --- Step 2: covariance = (1/(n-1)) * X^T * X  (n_features x n_features) ---
  // Row-major X (n_samples x n_features) = col-major X^T (n_features x n_samples).
  // We want C = X^T * X  in row-major => C = X^T * X.
  // Using col-major convention:
  //   C_colmaj = X_rowmaj^T_as_colmaj * X_rowmaj_as_colmaj
  //   i.e. (n_features x n_features) = (n_features x n_samples) * (n_samples x n_features)
  // cublasSgemm Op_A=N, Op_B=T:
  //   C = alpha * A * B^T + beta * C
  // with A = X_colmaj (n_features x n_samples), B = X_colmaj (n_features x n_samples)
  // => C = X_colmaj * X_colmaj^T = X * X^T  (n_features x n_features)  <- correct!
  float* d_cov = nullptr;
  CUDA_CHECK(cudaMalloc(&d_cov, n_features * n_features * sizeof(float)));

  const float alpha = 1.0f / static_cast<float>(n_samples - 1);
  const float beta  = 0.0f;

  // cublasSgemm(handle, Op_A, Op_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
  // A: X_colmaj  (n_features x n_samples), leading dim = n_features
  // B: X_colmaj  (n_features x n_samples), leading dim = n_features
  // C: cov       (n_features x n_features), leading dim = n_features
  CUBLAS_CHECK(cublasSgemm(cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            n_features, n_features, n_samples,
                            &alpha,
                            d_X, n_features,
                            d_X, n_features,
                            &beta,
                            d_cov, n_features));

  // --- Step 3: eigendecompose covariance with cuSolver Ssyevd ---
  float* d_eigenvalues = nullptr;
  CUDA_CHECK(cudaMalloc(&d_eigenvalues, n_features * sizeof(float)));

  int* d_info = nullptr;
  CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

  // Query workspace size.
  int lwork = 0;
  CUSOLVER_CHECK(cusolverDnSsyevd_bufferSize(
      cusolver_handle,
      CUSOLVER_EIG_MODE_VECTOR,   // compute eigenvectors
      CUBLAS_FILL_MODE_LOWER,
      n_features,
      d_cov,
      n_features,
      d_eigenvalues,
      &lwork));

  float* d_work = nullptr;
  CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(float)));

  CUSOLVER_CHECK(cusolverDnSsyevd(
      cusolver_handle,
      CUSOLVER_EIG_MODE_VECTOR,
      CUBLAS_FILL_MODE_LOWER,
      n_features,
      d_cov,                 // overwritten with eigenvectors (col-major)
      n_features,
      d_eigenvalues,         // eigenvalues in ascending order
      d_work, lwork, d_info));

  CUDA_CHECK(cudaDeviceSynchronize());

  int h_info = 0;
  CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
  if (h_info != 0) {
    fprintf(stderr, "[pca] cuSolver Ssyevd failed with info=%d\n", h_info);
  }

  // Copy eigenvalues and eigenvectors to host.
  std::vector<float> h_eigenvalues(n_features);
  CUDA_CHECK(cudaMemcpy(h_eigenvalues.data(), d_eigenvalues,
                        n_features * sizeof(float), cudaMemcpyDeviceToHost));

  // d_cov now holds eigenvectors as columns (col-major, n_features x n_features)
  // Eigenvalues are in ascending order; we want descending.
  std::vector<float> h_eigvec_colmaj(n_features * n_features);
  CUDA_CHECK(cudaMemcpy(h_eigvec_colmaj.data(), d_cov,
                        n_features * n_features * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // Reverse to get descending order.
  std::reverse(h_eigenvalues.begin(), h_eigenvalues.end());
  // Reverse eigenvector columns correspondingly.
  std::vector<float> h_eigvec_desc(n_features * n_features);
  for (int j = 0; j < n_features; ++j) {
    int src_col = n_features - 1 - j;
    for (int i = 0; i < n_features; ++i)
      h_eigvec_desc[j * n_features + i] = h_eigvec_colmaj[src_col * n_features + i];
  }

  // Clamp tiny negative eigenvalues to zero (numerical noise).
  float total_var = 0.f;
  for (float ev : h_eigenvalues) total_var += (ev > 0 ? ev : 0.f);

  result->eigenvalues = h_eigenvalues;
  result->explained_variance_ratio.resize(n_components);
  for (int c = 0; c < n_components; ++c) {
    float ev = (h_eigenvalues[c] > 0) ? h_eigenvalues[c] : 0.f;
    result->explained_variance_ratio[c] =
        (total_var > 0) ? ev / total_var : 0.f;
  }

  // Top n_components eigenvectors as rows (row-major on host).
  // h_eigvec_desc is column-major; column j contains the j-th eigenvector.
  result->components.resize(n_components * n_features);
  for (int c = 0; c < n_components; ++c) {
    for (int f = 0; f < n_features; ++f)
      result->components[c * n_features + f] = h_eigvec_desc[c * n_features + f];
  }

  // --- Step 4: project X onto top components:  Z = X_centered * V_top^T ---
  // Z (n_samples x n_components) = X (n_samples x n_features)
  //                                  * V_top^T (n_features x n_components)
  // Col-major cuBLAS convention:
  //   Z_col (n_samples x n_components)
  //     = X_col (n_features x n_samples)^T
  //       * V_col (n_features x n_components)
  // cublasSgemm Op_A=T, Op_B=N:
  //   C = alpha * A^T * B
  //   A = X_col (n_features x n_samples), B = V_col (n_features x n_components)
  //   C = X_col^T * V_col = X_rowmaj * V_col

  // Upload top eigenvectors to device (col-major: each column is one eigenvector).
  // h_eigvec_desc stores each eigenvector as a column (already col-major).
  // We only need the first n_components columns.
  float* d_V = nullptr;
  CUDA_CHECK(cudaMalloc(&d_V, n_features * n_components * sizeof(float)));
  // h_eigvec_desc is col-major (n_features x n_features); take first n_components cols.
  CUDA_CHECK(cudaMemcpy(d_V, h_eigvec_desc.data(),
                        n_features * n_components * sizeof(float),
                        cudaMemcpyHostToDevice));

  float* d_Z = nullptr;
  CUDA_CHECK(cudaMalloc(&d_Z, n_samples * n_components * sizeof(float)));

  const float alpha2 = 1.0f, beta2 = 0.0f;
  CUBLAS_CHECK(cublasSgemm(cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            n_samples, n_components, n_features,
                            &alpha2,
                            d_X, n_features,
                            d_V, n_features,
                            &beta2,
                            d_Z, n_samples));

  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(t1));
  CUDA_CHECK(cudaEventSynchronize(t1));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

  // Copy projected data to host (col-major n_samples x n_components -> row-major).
  std::vector<float> h_Z_colmaj(n_samples * n_components);
  CUDA_CHECK(cudaMemcpy(h_Z_colmaj.data(), d_Z,
                        n_samples * n_components * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // Transpose col-major to row-major.
  result->projected.resize(n_samples * n_components);
  for (int i = 0; i < n_samples; ++i)
    for (int c = 0; c < n_components; ++c)
      result->projected[i * n_components + c] = h_Z_colmaj[c * n_samples + i];

  result->n_components = n_components;
  result->gpu_ms       = static_cast<double>(ms);

  // --- Free ---
  CUDA_CHECK(cudaFree(d_X));
  CUDA_CHECK(cudaFree(d_mean));
  CUDA_CHECK(cudaFree(d_cov));
  CUDA_CHECK(cudaFree(d_eigenvalues));
  CUDA_CHECK(cudaFree(d_work));
  CUDA_CHECK(cudaFree(d_info));
  CUDA_CHECK(cudaFree(d_V));
  CUDA_CHECK(cudaFree(d_Z));
  CUDA_CHECK(cudaEventDestroy(t0));
  CUDA_CHECK(cudaEventDestroy(t1));
}

// ---------------------------------------------------------------------------
// PrintPCAResult
// ---------------------------------------------------------------------------

void PrintPCAResult(const PCAResult& result, int n_features) {
  printf("\n=== PCA Result (%d components) ===\n", result.n_components);
  printf("  GPU time : %.3f ms\n", result.gpu_ms);

  float cum = 0.f;
  for (int c = 0; c < result.n_components; ++c) {
    cum += result.explained_variance_ratio[c];
    printf("  PC%d  eigenvalue=%.4f  var_explained=%.2f%%  cumulative=%.2f%%\n",
           c + 1,
           result.eigenvalues[c],
           result.explained_variance_ratio[c] * 100.f,
           cum * 100.f);
  }
  printf("  Component loadings (eigenvectors as rows):\n");
  for (int c = 0; c < result.n_components; ++c) {
    printf("  PC%d: [", c + 1);
    for (int f = 0; f < n_features; ++f)
      printf(" %+.4f", result.components[c * n_features + f]);
    printf(" ]\n");
  }
  printf("==================================\n\n");
}
