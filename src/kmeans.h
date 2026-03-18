// Copyright 2024 Iris GPU ML Pipeline
//
// GPU-accelerated K-means clustering using custom CUDA kernels and Thrust.
// Algorithm:
//   1. Initialise k centroids via k-means++ seeding (on host).
//   2. GPU kernel: assign each point to its nearest centroid (Euclidean).
//   3. GPU reduction (Thrust): recompute centroids as cluster means.
//   4. Repeat until convergence or max_iter reached.
//
// Extensions:
//   RunKMeansMulti – best-of-N restarts (lowest inertia wins).
//   RunElbowSweep  – sweeps k=1..k_max, records inertia and silhouette vs k.

#ifndef SRC_KMEANS_H_
#define SRC_KMEANS_H_

#include <vector>
#include <string>
#include "common.h"
#include <cublas_v2.h>

// KMeansResult – output of a K-means run.
struct KMeansResult {
  std::vector<int>   labels;       // cluster id for each sample [n_samples]
  std::vector<float> centroids;    // row-major centroid matrix [k x n_features]
  int    iterations;               // number of iterations until convergence
  float  inertia;                  // sum of squared distances to centroids
  double gpu_ms;                   // wall-clock GPU time (milliseconds)
};

// ElbowResult – output of a k-sweep elbow analysis.
struct ElbowResult {
  std::vector<int>   k_values;       // k = 1, 2, ..., k_max
  std::vector<float> inertias;       // inertia per k
  std::vector<float> silhouettes;    // silhouette score per k (0.f for k==1)
  int                optimal_k;      // k with highest silhouette score
  double             total_gpu_ms;
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// RunKMeans – single K-means run with given seed.
void RunKMeans(const float* d_data, int n_samples, int n_features,
               int k, int max_iter, unsigned int seed,
               KMeansResult* result);

// RunKMeansMulti – best-of-n_runs K-means (seeds = base_seed, base_seed+1, …).
// Returns the result with lowest inertia.
void RunKMeansMulti(const float* d_data, int n_samples, int n_features,
                    int k, int max_iter,
                    unsigned int base_seed, int n_runs,
                    KMeansResult* best_result);

// RunElbowSweep – sweeps k from 1 to k_max, calling RunKMeansMulti at each k.
// When cublas_handle != nullptr and k >= 2, computes the silhouette score.
void RunElbowSweep(const float* d_data, int n_samples, int n_features,
                   int k_max, int max_iter,
                   unsigned int base_seed, int n_runs_per_k,
                   cublasHandle_t cublas_handle,
                   ElbowResult* result);

// PrintKMeansResult – display result summary to stdout.
void PrintKMeansResult(const KMeansResult& result, int k, int n_features);

// PrintElbowResult – display elbow sweep table to stdout.
void PrintElbowResult(const ElbowResult& result);

#endif  // SRC_KMEANS_H_
