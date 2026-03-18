// Copyright 2024 Iris GPU ML Pipeline
//
// GPU-accelerated Z-score feature normalization using NVIDIA Thrust.
// Normalizes each feature (column) independently to zero mean / unit variance.

#ifndef SRC_NORMALIZER_H_
#define SRC_NORMALIZER_H_

#include <vector>
#include <string>
#include "common.h"

// NormStats – per-feature statistics computed on the GPU.
struct NormStats {
  std::vector<float> mean;   // per-feature mean
  std::vector<float> stddev; // per-feature standard deviation
};

// NormalizeGpu – performs Z-score normalization on *d_data* in place.
//
//   d_data     : device pointer to row-major float matrix (n_samples × n_features)
//   n_samples  : number of data rows
//   n_features : number of feature columns
//   stats      : output – per-feature mean and stddev (computed on GPU)
//
// After the call, d_data[i][j] = (original - mean[j]) / stddev[j].
void NormalizeGpu(float* d_data, int n_samples, int n_features,
                  NormStats* stats);

// PrintNormStats – display per-feature mean and std to stdout.
void PrintNormStats(const NormStats& stats, int n_features);

#endif  // SRC_NORMALIZER_H_
