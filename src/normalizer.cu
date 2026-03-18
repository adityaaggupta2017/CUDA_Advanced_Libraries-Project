// Copyright 2024 Iris GPU ML Pipeline
//
// GPU-accelerated Z-score normalization using Thrust.
// For each feature column j:
//   mu_j    = (1/N) * sum_i X[i,j]
//   sigma_j = sqrt( (1/N) * sum_i (X[i,j] - mu_j)^2 )
//   X[i,j]  = (X[i,j] - mu_j) / sigma_j

#include "normalizer.h"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

// ---------------------------------------------------------------------------
// Device functor: extract column j from row-major matrix
// ---------------------------------------------------------------------------

struct ExtractColumn {
  const float* data;
  int n_features;
  int col;

  __device__ float operator()(int row) const {
    return data[row * n_features + col];
  }
};

// ---------------------------------------------------------------------------
// Device functor: (x - mu)^2
// ---------------------------------------------------------------------------

struct SquaredDiff {
  float mu;
  __device__ float operator()(float x) const {
    return (x - mu) * (x - mu);
  }
};

// ---------------------------------------------------------------------------
// Kernel: apply Z-score to one feature column
// ---------------------------------------------------------------------------

__global__ void ApplyZscoreKernel(float* data, int n_samples, int n_features,
                                   int col, float mu, float sigma) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n_samples) {
    float val = data[row * n_features + col];
    data[row * n_features + col] = (val - mu) / sigma;
  }
}

// ---------------------------------------------------------------------------
// NormalizeGpu
// ---------------------------------------------------------------------------

void NormalizeGpu(float* d_data, int n_samples, int n_features,
                  NormStats* stats) {
  stats->mean.resize(n_features);
  stats->stddev.resize(n_features);

  thrust::device_ptr<float> ptr(d_data);
  const float inv_n = 1.0f / static_cast<float>(n_samples);

  for (int col = 0; col < n_features; ++col) {
    // --- Compute mean using a strided reduction via transform iterator ---
    auto idx_begin = thrust::make_counting_iterator<int>(0);
    auto idx_end   = thrust::make_counting_iterator<int>(n_samples);

    ExtractColumn extractor{d_data, n_features, col};
    auto val_begin = thrust::make_transform_iterator(idx_begin, extractor);
    auto val_end   = thrust::make_transform_iterator(idx_end,   extractor);

    float sum = thrust::reduce(val_begin, val_end, 0.0f, thrust::plus<float>());
    float mu  = sum * inv_n;
    stats->mean[col] = mu;

    // --- Compute variance ---
    SquaredDiff sq_diff{mu};
    auto sq_begin = thrust::make_transform_iterator(val_begin, sq_diff);
    auto sq_end   = thrust::make_transform_iterator(val_end,   sq_diff);

    float var_sum = thrust::reduce(sq_begin, sq_end, 0.0f, thrust::plus<float>());
    float sigma   = sqrtf(var_sum * inv_n);
    if (sigma < 1e-8f) sigma = 1.0f;   // guard against constant features
    stats->stddev[col] = sigma;

    // --- Apply normalization kernel ---
    int threads = 256;
    int blocks  = (n_samples + threads - 1) / threads;
    ApplyZscoreKernel<<<blocks, threads>>>(d_data, n_samples, n_features,
                                           col, mu, sigma);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// PrintNormStats
// ---------------------------------------------------------------------------

void PrintNormStats(const NormStats& stats, int n_features) {
  printf("\n=== Normalization Statistics (GPU Z-score) ===\n");
  for (int j = 0; j < n_features; ++j) {
    printf("  %-22s  mean=%7.4f  std=%7.4f\n",
           kFeatureNames[j], stats.mean[j], stats.stddev[j]);
  }
  printf("==============================================\n\n");
}
