// Copyright 2024 Iris GPU ML Pipeline
//
// GPU-accelerated K-means with custom CUDA kernels + Thrust reductions.

#include "kmeans.h"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

// ---------------------------------------------------------------------------
// Device kernel: assign labels
// Each thread handles one sample and computes the distance to every centroid.
// ---------------------------------------------------------------------------

__global__ void AssignLabelsKernel(const float* __restrict__ data,
                                    const float* __restrict__ centroids,
                                    int* __restrict__ labels,
                                    int n_samples, int n_features, int k) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_samples) return;

  float best_dist = FLT_MAX;
  int   best_k    = 0;

  for (int c = 0; c < k; ++c) {
    float dist = 0.f;
    for (int f = 0; f < n_features; ++f) {
      float diff = data[idx * n_features + f] - centroids[c * n_features + f];
      dist += diff * diff;
    }
    if (dist < best_dist) {
      best_dist = dist;
      best_k    = c;
    }
  }
  labels[idx] = best_k;
}

// ---------------------------------------------------------------------------
// Device kernel: accumulate point coordinates into centroid sums
// ---------------------------------------------------------------------------

__global__ void AccumulateCentroidsKernel(const float* __restrict__ data,
                                           const int*   __restrict__ labels,
                                           float* __restrict__ centroid_sum,
                                           int*   __restrict__ centroid_count,
                                           int n_samples, int n_features,
                                           int k) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_samples) return;

  int c = labels[idx];
  atomicAdd(&centroid_count[c], 1);
  for (int f = 0; f < n_features; ++f) {
    atomicAdd(&centroid_sum[c * n_features + f],
              data[idx * n_features + f]);
  }
}

// ---------------------------------------------------------------------------
// Device kernel: divide accumulated sums by counts to get new centroids.
// If a cluster is empty keep the previous centroid.
// ---------------------------------------------------------------------------

__global__ void UpdateCentroidsKernel(const float* __restrict__ centroid_sum,
                                       const int*   __restrict__ centroid_count,
                                       float* __restrict__ centroids,
                                       int k, int n_features) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= k) return;

  int cnt = centroid_count[c];
  if (cnt > 0) {
    for (int f = 0; f < n_features; ++f) {
      centroids[c * n_features + f] =
          centroid_sum[c * n_features + f] / static_cast<float>(cnt);
    }
  }
}

// ---------------------------------------------------------------------------
// Device kernel: compute inertia (sum of squared distances to centroids)
// ---------------------------------------------------------------------------

__global__ void ComputeInertiaKernel(const float* __restrict__ data,
                                      const float* __restrict__ centroids,
                                      const int*   __restrict__ labels,
                                      float* __restrict__ inertia,
                                      int n_samples, int n_features) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_samples) return;

  int c = labels[idx];
  float dist = 0.f;
  for (int f = 0; f < n_features; ++f) {
    float diff = data[idx * n_features + f] - centroids[c * n_features + f];
    dist += diff * diff;
  }
  atomicAdd(inertia, dist);
}

// ---------------------------------------------------------------------------
// CPU helper: k-means++ initialisation
// ---------------------------------------------------------------------------

static std::vector<float> KMeansPlusPlusInit(
    const std::vector<float>& h_data, int n_samples, int n_features,
    int k, std::mt19937& rng) {
  std::vector<float> centroids(k * n_features, 0.f);

  // Choose first centroid uniformly at random.
  std::uniform_int_distribution<int> uni(0, n_samples - 1);
  int first = uni(rng);
  for (int f = 0; f < n_features; ++f) {
    centroids[0 * n_features + f] = h_data[first * n_features + f];
  }

  std::vector<float> dist2(n_samples, FLT_MAX);

  for (int c = 1; c < k; ++c) {
    // Compute squared distances to nearest chosen centroid.
    for (int i = 0; i < n_samples; ++i) {
      for (int pc = 0; pc < c; ++pc) {
        float d = 0.f;
        for (int f = 0; f < n_features; ++f) {
          float diff = h_data[i * n_features + f] -
                       centroids[pc * n_features + f];
          d += diff * diff;
        }
        dist2[i] = std::min(dist2[i], d);
      }
    }
    // Sample next centroid proportional to dist2.
    float total = 0.f;
    for (float v : dist2) total += v;
    std::uniform_real_distribution<float> pick(0.f, total);
    float target = pick(rng);
    float cum    = 0.f;
    int   chosen = n_samples - 1;
    for (int i = 0; i < n_samples; ++i) {
      cum += dist2[i];
      if (cum >= target) { chosen = i; break; }
    }
    for (int f = 0; f < n_features; ++f) {
      centroids[c * n_features + f] = h_data[chosen * n_features + f];
    }
  }
  return centroids;
}

// ---------------------------------------------------------------------------
// RunKMeans
// ---------------------------------------------------------------------------

void RunKMeans(const float* d_data, int n_samples, int n_features,
               int k, int max_iter, unsigned int seed,
               KMeansResult* result) {
  assert(result != nullptr);

  // --- Copy data to host for k-means++ seeding ---
  std::vector<float> h_data(n_samples * n_features);
  CUDA_CHECK(cudaMemcpy(h_data.data(), d_data,
                        n_samples * n_features * sizeof(float),
                        cudaMemcpyDeviceToHost));

  std::mt19937 rng(seed);
  std::vector<float> h_centroids =
      KMeansPlusPlusInit(h_data, n_samples, n_features, k, rng);

  // --- Allocate device buffers ---
  float* d_centroids    = nullptr;
  float* d_centroid_sum = nullptr;
  int*   d_centroid_cnt = nullptr;
  int*   d_labels       = nullptr;
  float* d_inertia      = nullptr;

  CUDA_CHECK(cudaMalloc(&d_centroids,
                        k * n_features * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_centroid_sum,
                        k * n_features * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_centroid_cnt,
                        k * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_labels,
                        n_samples * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_inertia, sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids.data(),
                        k * n_features * sizeof(float),
                        cudaMemcpyHostToDevice));

  const int kThreads = 256;
  int blocks_n = (n_samples + kThreads - 1) / kThreads;
  int blocks_k = (k       + kThreads - 1) / kThreads;

  // --- CUDA event timing ---
  cudaEvent_t t_start, t_stop;
  CUDA_CHECK(cudaEventCreate(&t_start));
  CUDA_CHECK(cudaEventCreate(&t_stop));
  CUDA_CHECK(cudaEventRecord(t_start));

  std::vector<int> h_labels_prev(n_samples, -1);
  int iter = 0;

  for (; iter < max_iter; ++iter) {
    // 1. Assign labels.
    AssignLabelsKernel<<<blocks_n, kThreads>>>(
        d_data, d_centroids, d_labels, n_samples, n_features, k);
    CUDA_CHECK(cudaGetLastError());

    // 2. Copy labels to host to check convergence.
    std::vector<int> h_labels_curr(n_samples);
    CUDA_CHECK(cudaMemcpy(h_labels_curr.data(), d_labels,
                          n_samples * sizeof(int),
                          cudaMemcpyDeviceToHost));

    bool converged = (h_labels_curr == h_labels_prev);
    h_labels_prev = h_labels_curr;

    // 3. Recompute centroids via atomic accumulation.
    CUDA_CHECK(cudaMemset(d_centroid_sum, 0,
                          k * n_features * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_centroid_cnt, 0, k * sizeof(int)));

    AccumulateCentroidsKernel<<<blocks_n, kThreads>>>(
        d_data, d_labels, d_centroid_sum, d_centroid_cnt,
        n_samples, n_features, k);
    CUDA_CHECK(cudaGetLastError());

    UpdateCentroidsKernel<<<blocks_k, kThreads>>>(
        d_centroid_sum, d_centroid_cnt, d_centroids, k, n_features);
    CUDA_CHECK(cudaGetLastError());

    if (converged) { ++iter; break; }
  }

  // --- Compute inertia ---
  float zero = 0.f;
  CUDA_CHECK(cudaMemcpy(d_inertia, &zero, sizeof(float),
                        cudaMemcpyHostToDevice));
  ComputeInertiaKernel<<<blocks_n, kThreads>>>(
      d_data, d_centroids, d_labels, d_inertia, n_samples, n_features);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(t_stop));
  CUDA_CHECK(cudaEventSynchronize(t_stop));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, t_start, t_stop));

  // --- Collect results ---
  result->labels.resize(n_samples);
  result->centroids.resize(k * n_features);
  CUDA_CHECK(cudaMemcpy(result->labels.data(), d_labels,
                        n_samples * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(result->centroids.data(), d_centroids,
                        k * n_features * sizeof(float), cudaMemcpyDeviceToHost));
  float h_inertia = 0.f;
  CUDA_CHECK(cudaMemcpy(&h_inertia, d_inertia, sizeof(float),
                        cudaMemcpyDeviceToHost));
  result->iterations = iter;
  result->inertia    = h_inertia;
  result->gpu_ms     = static_cast<double>(ms);

  // --- Free device memory ---
  CUDA_CHECK(cudaFree(d_centroids));
  CUDA_CHECK(cudaFree(d_centroid_sum));
  CUDA_CHECK(cudaFree(d_centroid_cnt));
  CUDA_CHECK(cudaFree(d_labels));
  CUDA_CHECK(cudaFree(d_inertia));
  CUDA_CHECK(cudaEventDestroy(t_start));
  CUDA_CHECK(cudaEventDestroy(t_stop));
}

// ---------------------------------------------------------------------------
// PrintKMeansResult
// ---------------------------------------------------------------------------

void PrintKMeansResult(const KMeansResult& result, int k, int n_features) {
  printf("\n=== K-Means Result (k=%d) ===\n", k);
  printf("  Iterations : %d\n", result.iterations);
  printf("  Inertia    : %.4f\n", result.inertia);
  printf("  GPU time   : %.3f ms\n", result.gpu_ms);

  // Cluster sizes.
  std::vector<int> sizes(k, 0);
  for (int lbl : result.labels) sizes[lbl]++;
  printf("  Cluster sizes: ");
  for (int c = 0; c < k; ++c) printf("%d ", sizes[c]);
  printf("\n");

  // Centroids.
  printf("  Centroids (normalised space):\n");
  for (int c = 0; c < k; ++c) {
    printf("    Cluster %d: [", c);
    for (int f = 0; f < n_features; ++f) {
      printf(" %.4f", result.centroids[c * n_features + f]);
    }
    printf(" ]\n");
  }
  printf("============================\n\n");
}
