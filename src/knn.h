// Copyright 2024 Iris GPU ML Pipeline
//
// GPU-accelerated K-Nearest Neighbours classification using cuBLAS.
// Distance matrix is computed efficiently via the identity:
//   ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
// The inner-product matrix (-2*X_test * X_train^T) is computed with
// a single SGEMM call; squared norms are computed with Thrust.

#ifndef SRC_KNN_H_
#define SRC_KNN_H_

#include <vector>
#include <string>
#include "common.h"
#include <cublas_v2.h>

// KNNResult – output of a KNN classification run.
struct KNNResult {
  std::vector<int> predictions;   // predicted label for each test sample
  std::vector<int> true_labels;   // ground-truth labels (copy from input)
  float  accuracy;                // classification accuracy in [0, 1]
  double gpu_ms;                  // GPU wall-clock time (milliseconds)
};

// RunKNN – GPU K-Nearest Neighbours classifier (leave-one-out on full dataset).
//
//   d_data       : device – row-major normalised feature matrix (n x n_features)
//   h_labels     : host   – integer class labels [n]
//   n_samples    : total number of samples
//   n_features   : number of feature dimensions
//   k            : number of nearest neighbours to vote over
//   cublas_handle: an already-created cuBLAS handle
//   result       : populated on return
void RunKNN(const float* d_data, const std::vector<int>& h_labels,
            int n_samples, int n_features, int k,
            cublasHandle_t cublas_handle,
            KNNResult* result);

// PrintKNNResult – display result summary to stdout.
void PrintKNNResult(const KNNResult& result, int k);

#endif  // SRC_KNN_H_
