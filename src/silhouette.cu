// Copyright 2024 Iris GPU ML Pipeline
//
// GPU silhouette coefficient using cuBLAS SGEMM (distance matrix) +
// custom kernels for a(i), b(i), s(i), and Thrust for mean reduction.

#include "silhouette.h"

#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

// ---------------------------------------------------------------------------
// Kernel: add squared row norms to the distance matrix.
// dist[i,j] += sq_norm[i] + sq_norm[j]
// Matrix stored row-major (n x n).
// ---------------------------------------------------------------------------

__global__ void SilAddNormsKernel(float* dist,
                                   const float* __restrict__ sq_norm,
                                   int n) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || j >= n) return;
  dist[i * n + j] += sq_norm[i] + sq_norm[j];
}

// ---------------------------------------------------------------------------
// Kernel: squared L2 norm of each row.
// ---------------------------------------------------------------------------

__global__ void SilSquaredNormKernel(const float* __restrict__ data,
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
// Kernel: compute a(i) – mean intra-cluster distance.
// Each thread handles one sample.
// ---------------------------------------------------------------------------

__global__ void ComputeAKernel(const float* __restrict__ dist,  // n x n row-major
                                 const int*   __restrict__ labels,
                                 float* __restrict__ a_vals,
                                 int n_samples) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_samples) return;

  int   my_cluster = labels[i];
  float sum        = 0.f;
  int   count      = 0;

  for (int j = 0; j < n_samples; ++j) {
    if (j != i && labels[j] == my_cluster) {
      sum += sqrtf(fmaxf(dist[i * n_samples + j], 0.f));
      ++count;
    }
  }
  a_vals[i] = (count > 0) ? sum / static_cast<float>(count) : 0.f;
}

// ---------------------------------------------------------------------------
// Kernel: compute b(i) – min mean distance to any other cluster.
// ---------------------------------------------------------------------------

__global__ void ComputeBKernel(const float* __restrict__ dist,
                                 const int*   __restrict__ labels,
                                 float* __restrict__ b_vals,
                                 int n_samples, int k) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_samples) return;

  int   my_cluster = labels[i];
  float best_b     = FLT_MAX;

  for (int c = 0; c < k; ++c) {
    if (c == my_cluster) continue;
    float sum   = 0.f;
    int   count = 0;
    for (int j = 0; j < n_samples; ++j) {
      if (labels[j] == c) {
        sum += sqrtf(fmaxf(dist[i * n_samples + j], 0.f));
        ++count;
      }
    }
    if (count > 0) {
      float mean_c = sum / static_cast<float>(count);
      best_b = fminf(best_b, mean_c);
    }
  }
  b_vals[i] = (best_b < FLT_MAX) ? best_b : 0.f;
}

// ---------------------------------------------------------------------------
// Kernel: compute s(i) = (b(i) - a(i)) / max(a(i), b(i)).
// ---------------------------------------------------------------------------

__global__ void ComputeSKernel(const float* __restrict__ a,
                                 const float* __restrict__ b,
                                 float* __restrict__ s,
                                 int n_samples) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_samples) return;

  float ai = a[i], bi = b[i];
  float denom = fmaxf(ai, bi);
  s[i] = (denom > 1e-8f) ? (bi - ai) / denom : 0.f;
}

// ---------------------------------------------------------------------------
// ComputeSilhouette
// ---------------------------------------------------------------------------

void ComputeSilhouette(const float* d_data,
                       const int*   d_labels,
                       int n_samples, int n_features, int k,
                       cublasHandle_t cublas_handle,
                       SilhouetteResult* result) {
  assert(result != nullptr);

  cudaEvent_t t0, t1;
  CUDA_CHECK(cudaEventCreate(&t0));
  CUDA_CHECK(cudaEventCreate(&t1));
  CUDA_CHECK(cudaEventRecord(t0));

  // --- Step 1: compute n x n squared Euclidean distance matrix via SGEMM ---
  // D[i,j] = ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * x_i . x_j
  //
  // cuBLAS Op_A=T, Op_B=N:  D_colmaj(n x n) = -2 * X_colmaj^T * X_colmaj
  // X is stored row-major (n x n_features) = colmaj (n_features x n).

  float* d_dist  = nullptr;  // col-major n x n from SGEMM
  float* d_sq_norm = nullptr;

  CUDA_CHECK(cudaMalloc(&d_dist,   (size_t)n_samples * n_samples * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_sq_norm, n_samples * sizeof(float)));

  const int kThreads = 256;
  int blocks_n = (n_samples + kThreads - 1) / kThreads;

  SilSquaredNormKernel<<<blocks_n, kThreads>>>(d_data, d_sq_norm,
                                                n_samples, n_features);
  CUDA_CHECK(cudaGetLastError());

  const float alpha = -2.f, beta = 0.f;
  // Result: D_colmaj(n_samples x n_samples)
  // cublasSgemm: Op_T, Op_N -> A^T * B
  //   A = X row-major pointer (colmaj: n_features x n_samples)
  //   B = X row-major pointer (colmaj: n_features x n_samples)
  // => D = X_colmaj^T * X_colmaj = X_rowmaj * X_rowmaj^T (n_samples x n_samples)
  CUBLAS_CHECK(cublasSgemm(cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            n_samples, n_samples, n_features,
                            &alpha,
                            d_data, n_features,
                            d_data, n_features,
                            &beta,
                            d_dist, n_samples));

  // Add squared norms: D_colmaj[j*n+i] += sq_norm[i] + sq_norm[j]
  // which, viewed row-major, is D[i,j] += sq_norm[i] + sq_norm[j] (symmetric)
  dim3 block2d(16, 16);
  dim3 grid2d((n_samples + 15) / 16, (n_samples + 15) / 16);
  SilAddNormsKernel<<<grid2d, block2d>>>(d_dist, d_sq_norm, n_samples);
  CUDA_CHECK(cudaGetLastError());

  // Transpose col-major to row-major on host (n=150 so this is trivial).
  std::vector<float> h_dist_col(n_samples * n_samples);
  CUDA_CHECK(cudaMemcpy(h_dist_col.data(), d_dist,
                        n_samples * n_samples * sizeof(float),
                        cudaMemcpyDeviceToHost));

  float* d_dist_row = nullptr;
  CUDA_CHECK(cudaMalloc(&d_dist_row,
                        (size_t)n_samples * n_samples * sizeof(float)));

  std::vector<float> h_dist_row(n_samples * n_samples);
  for (int i = 0; i < n_samples; ++i)
    for (int j = 0; j < n_samples; ++j)
      h_dist_row[i * n_samples + j] = h_dist_col[j * n_samples + i];

  CUDA_CHECK(cudaMemcpy(d_dist_row, h_dist_row.data(),
                        n_samples * n_samples * sizeof(float),
                        cudaMemcpyHostToDevice));

  // --- Step 2: compute a(i) and b(i) ---
  float* d_a = nullptr, *d_b = nullptr, *d_s = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, n_samples * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, n_samples * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_s, n_samples * sizeof(float)));

  ComputeAKernel<<<blocks_n, kThreads>>>(d_dist_row, d_labels,
                                          d_a, n_samples);
  CUDA_CHECK(cudaGetLastError());

  ComputeBKernel<<<blocks_n, kThreads>>>(d_dist_row, d_labels,
                                          d_b, n_samples, k);
  CUDA_CHECK(cudaGetLastError());

  // --- Step 3: s(i) ---
  ComputeSKernel<<<blocks_n, kThreads>>>(d_a, d_b, d_s, n_samples);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(t1));
  CUDA_CHECK(cudaEventSynchronize(t1));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

  // --- Step 4: mean score via Thrust ---
  thrust::device_ptr<float> s_ptr(d_s);
  float total = thrust::reduce(s_ptr, s_ptr + n_samples, 0.f,
                                thrust::plus<float>());
  result->mean_score = total / static_cast<float>(n_samples);

  // Per-sample to host.
  result->per_sample.resize(n_samples);
  CUDA_CHECK(cudaMemcpy(result->per_sample.data(), d_s,
                        n_samples * sizeof(float), cudaMemcpyDeviceToHost));

  // Per-cluster means (host).
  std::vector<int> h_labels(n_samples);
  CUDA_CHECK(cudaMemcpy(h_labels.data(), d_labels,
                        n_samples * sizeof(int), cudaMemcpyDeviceToHost));

  result->per_cluster.assign(k, 0.f);
  std::vector<int> cluster_count(k, 0);
  for (int i = 0; i < n_samples; ++i) {
    result->per_cluster[h_labels[i]] += result->per_sample[i];
    ++cluster_count[h_labels[i]];
  }
  for (int c = 0; c < k; ++c)
    if (cluster_count[c] > 0)
      result->per_cluster[c] /= static_cast<float>(cluster_count[c]);

  result->gpu_ms = static_cast<double>(ms);

  // Free.
  CUDA_CHECK(cudaFree(d_dist));
  CUDA_CHECK(cudaFree(d_dist_row));
  CUDA_CHECK(cudaFree(d_sq_norm));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_s));
  CUDA_CHECK(cudaEventDestroy(t0));
  CUDA_CHECK(cudaEventDestroy(t1));
}

// ---------------------------------------------------------------------------
// PrintSilhouetteResult
// ---------------------------------------------------------------------------

void PrintSilhouetteResult(const SilhouetteResult& result, int k) {
  printf("\n=== Silhouette Coefficient (k=%d) ===\n", k);
  printf("  Mean silhouette score : %.4f\n", result.mean_score);
  printf("  Interpretation        : ");
  if      (result.mean_score > 0.7f)  printf("Strong structure\n");
  else if (result.mean_score > 0.5f)  printf("Reasonable structure\n");
  else if (result.mean_score > 0.25f) printf("Weak structure\n");
  else                                 printf("No substantial structure\n");
  printf("  Per-cluster mean silhouette:\n");
  for (int c = 0; c < k; ++c)
    printf("    Cluster %d : %.4f\n", c, result.per_cluster[c]);
  printf("  GPU time : %.3f ms\n", result.gpu_ms);
  printf("=====================================\n\n");
}
