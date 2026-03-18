// Copyright 2024 Iris GPU ML Pipeline
//
// GPU-accelerated Principal Component Analysis (PCA) using cuBLAS and cuSolver.
// Algorithm:
//   1. Mean-centre the data matrix X  (using Thrust reductions).
//   2. Compute the covariance matrix  C = (1/(n-1)) * X^T * X  via cuBLAS SGEMM.
//   3. Eigendecompose C using cuSolver Dsyevd (symmetric eigenvalue solver).
//   4. Sort eigenvectors by descending eigenvalue magnitude.
//   5. Project X onto the top n_components eigenvectors:  Z = X * V_top^T.

#ifndef SRC_PCA_H_
#define SRC_PCA_H_

#include <vector>
#include <string>
#include "common.h"
#include <cublas_v2.h>
#include <cusolverDn.h>

// PCAResult – output of a PCA run.
struct PCAResult {
  // Projected data: n_samples x n_components (row-major, host)
  std::vector<float> projected;
  // Explained variance ratio for each retained component
  std::vector<float> explained_variance_ratio;
  // All eigenvalues (descending order)
  std::vector<float> eigenvalues;
  // Top eigenvectors as rows: n_components x n_features (row-major, host)
  std::vector<float> components;
  int    n_components;
  double gpu_ms;
};

// RunPCA – GPU PCA projection.
//
//   d_data        : device – row-major feature matrix (n_samples x n_features).
//                   NOTE: a working copy is made internally; d_data is unmodified.
//   n_samples     : number of rows
//   n_features    : number of columns / original dimensionality
//   n_components  : number of principal components to retain
//   cublas_handle : an already-created cuBLAS handle
//   cusolver_handle: an already-created cuSolver dense handle
//   result        : populated on return
void RunPCA(const float* d_data, int n_samples, int n_features,
            int n_components,
            cublasHandle_t cublas_handle,
            cusolverDnHandle_t cusolver_handle,
            PCAResult* result);

// PrintPCAResult – display variance explained and component loadings.
void PrintPCAResult(const PCAResult& result, int n_features);

#endif  // SRC_PCA_H_
