// Copyright 2024 Iris GPU ML Pipeline
//
// GPU-accelerated K-means clustering using custom CUDA kernels and Thrust.
// Algorithm:
//   1. Initialise k centroids via k-means++ seeding (on host).
//   2. GPU kernel: assign each point to its nearest centroid (Euclidean).
//   3. GPU reduction (Thrust): recompute centroids as cluster means.
//   4. Repeat until convergence or max_iter reached.

#ifndef SRC_KMEANS_H_
#define SRC_KMEANS_H_

#include <vector>
#include <string>
#include "common.h"

// KMeansResult – output of a K-means run.
struct KMeansResult {
  std::vector<int>   labels;       // cluster id for each sample [n_samples]
  std::vector<float> centroids;    // row-major centroid matrix [k x n_features]
  int    iterations;               // number of iterations until convergence
  float  inertia;                  // sum of squared distances to centroids
  double gpu_ms;                   // wall-clock GPU time (milliseconds)
};

// RunKMeans – performs GPU K-means clustering.
//
//   d_data     : device pointer – normalised row-major matrix (n_samples x n_features)
//   n_samples  : number of data rows
//   n_features : number of feature columns
//   k          : number of clusters
//   max_iter   : maximum number of iterations
//   seed       : RNG seed for centroid initialisation
//   result     : populated on return
void RunKMeans(const float* d_data, int n_samples, int n_features,
               int k, int max_iter, unsigned int seed,
               KMeansResult* result);

// PrintKMeansResult – display result summary to stdout.
void PrintKMeansResult(const KMeansResult& result, int k, int n_features);

#endif  // SRC_KMEANS_H_
