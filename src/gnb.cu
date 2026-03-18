// Copyright 2024 Iris GPU ML Pipeline
//
// GPU Gaussian Naive Bayes: Thrust training + custom log-likelihood kernel.

#include "gnb.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>

// ---------------------------------------------------------------------------
// File-scope Thrust functors for training (must be at namespace scope)
// ---------------------------------------------------------------------------

// Extract element data[row * n_features + col] from a flat device array.
struct ExtractFeatureFunctor {
  const float* data;
  int n_features;
  int col;
  __device__ float operator()(int row) const {
    return data[row * n_features + col];
  }
};

// Compute squared difference from mean: (x - mu)^2
struct SqDiffFunctor {
  float mu;
  __device__ float operator()(float x) const {
    return (x - mu) * (x - mu);
  }
};

// ---------------------------------------------------------------------------
// GPU predict kernel: log P(c|x) = log_prior[c] + log-likelihood(x | c)
// ---------------------------------------------------------------------------

__global__ void GNBPredictKernel(const float* __restrict__ data,
                                   const float* __restrict__ class_mean,
                                   const float* __restrict__ class_var,
                                   const float* __restrict__ log_prior,
                                   int* __restrict__ predictions,
                                   int n_samples, int n_features,
                                   int n_classes) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_samples) return;

  float best_log_prob = -1e30f;
  int   best_class    = 0;

  for (int c = 0; c < n_classes; ++c) {
    float log_prob = log_prior[c];
    for (int f = 0; f < n_features; ++f) {
      float x    = data[i * n_features + f];
      float mu   = class_mean[c * n_features + f];
      float var  = class_var[c  * n_features + f];
      // log N(x | mu, var) = -0.5 * [log(2π var) + (x-mu)²/var]
      log_prob -= 0.5f * (logf(2.f * 3.14159265f * var) +
                          (x - mu) * (x - mu) / var);
    }
    if (log_prob > best_log_prob) {
      best_log_prob = log_prob;
      best_class    = c;
    }
  }
  predictions[i] = best_class;
}

// ---------------------------------------------------------------------------
// TrainGNB
// ---------------------------------------------------------------------------

void TrainGNB(const float* d_data,
              const std::vector<int>& h_labels,
              int n_samples, int n_features, int n_classes,
              GNBModel* model) {
  assert(model != nullptr);
  model->n_classes  = n_classes;
  model->n_features = n_features;
  model->class_mean.assign(n_classes * n_features, 0.f);
  model->class_var.assign(n_classes  * n_features, 1.f);
  model->log_prior.assign(n_classes, 0.f);

  cudaEvent_t t0, t1;
  CUDA_CHECK(cudaEventCreate(&t0));
  CUDA_CHECK(cudaEventCreate(&t1));
  CUDA_CHECK(cudaEventRecord(t0));

  // Per-class: extract class-specific rows and compute mean + variance via Thrust.
  for (int c = 0; c < n_classes; ++c) {
    // Identify row indices belonging to class c.
    std::vector<int> class_rows;
    class_rows.reserve(n_samples / n_classes + 1);
    for (int i = 0; i < n_samples; ++i)
      if (h_labels[i] == c) class_rows.push_back(i);

    int nc = static_cast<int>(class_rows.size());
    if (nc == 0) continue;

    // Upload class row indices to device.
    thrust::device_vector<int> d_rows(class_rows);
    float inv_nc = 1.0f / static_cast<float>(nc);

    for (int f = 0; f < n_features; ++f) {
      // Build transform iterators: index -> data[index * n_features + f]
      ExtractFeatureFunctor ef{d_data, n_features, f};

      auto row_begin = d_rows.begin();
      auto row_end   = d_rows.end();
      auto val_begin = thrust::make_transform_iterator(row_begin, ef);
      auto val_end   = thrust::make_transform_iterator(row_end,   ef);

      // Mean.
      float sum = thrust::reduce(val_begin, val_end, 0.f,
                                  thrust::plus<float>());
      float mu  = sum * inv_nc;
      model->class_mean[c * n_features + f] = mu;

      // Variance: E[(x-mu)^2]
      SqDiffFunctor sqd{mu};
      auto sq_begin = thrust::make_transform_iterator(val_begin, sqd);
      auto sq_end   = thrust::make_transform_iterator(val_end,   sqd);
      float var_sum = thrust::reduce(sq_begin, sq_end, 0.f,
                                     thrust::plus<float>());
      float var = var_sum * inv_nc;
      // Apply variance floor to avoid log(0).
      model->class_var[c * n_features + f] = fmaxf(var, 1e-6f);
    }

    // Log prior: log P(class=c) = log(nc / n_samples)
    model->log_prior[c] = logf(static_cast<float>(nc) /
                                static_cast<float>(n_samples));
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(t1));
  CUDA_CHECK(cudaEventSynchronize(t1));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
  model->train_gpu_ms = static_cast<double>(ms);

  CUDA_CHECK(cudaEventDestroy(t0));
  CUDA_CHECK(cudaEventDestroy(t1));
}

// ---------------------------------------------------------------------------
// PredictGNB
// ---------------------------------------------------------------------------

void PredictGNB(const float* d_data,
                const GNBModel& model,
                const std::vector<int>& h_true_labels,
                int n_samples,
                GNBResult* result) {
  assert(result != nullptr);

  const int nc = model.n_classes;
  const int nf = model.n_features;

  // Upload model parameters to device.
  float* d_mean = nullptr, *d_var = nullptr, *d_prior = nullptr;
  int*   d_pred = nullptr;
  CUDA_CHECK(cudaMalloc(&d_mean,  nc * nf * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_var,   nc * nf * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_prior, nc      * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_pred,  n_samples * sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_mean,  model.class_mean.data(),
                        nc * nf * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_var,   model.class_var.data(),
                        nc * nf * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_prior, model.log_prior.data(),
                        nc      * sizeof(float), cudaMemcpyHostToDevice));

  cudaEvent_t t0, t1;
  CUDA_CHECK(cudaEventCreate(&t0));
  CUDA_CHECK(cudaEventCreate(&t1));
  CUDA_CHECK(cudaEventRecord(t0));

  int threads = 256;
  int blocks  = (n_samples + threads - 1) / threads;
  GNBPredictKernel<<<blocks, threads>>>(
      d_data, d_mean, d_var, d_prior, d_pred,
      n_samples, nf, nc);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(t1));
  CUDA_CHECK(cudaEventSynchronize(t1));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

  result->predictions.resize(n_samples);
  CUDA_CHECK(cudaMemcpy(result->predictions.data(), d_pred,
                        n_samples * sizeof(int), cudaMemcpyDeviceToHost));

  result->true_labels = h_true_labels;
  int correct = 0;
  for (int i = 0; i < n_samples; ++i)
    if (result->predictions[i] == result->true_labels[i]) ++correct;
  result->accuracy = static_cast<float>(correct) /
                     static_cast<float>(n_samples);
  result->predict_gpu_ms = static_cast<double>(ms);

  CUDA_CHECK(cudaFree(d_mean));
  CUDA_CHECK(cudaFree(d_var));
  CUDA_CHECK(cudaFree(d_prior));
  CUDA_CHECK(cudaFree(d_pred));
  CUDA_CHECK(cudaEventDestroy(t0));
  CUDA_CHECK(cudaEventDestroy(t1));
}

// ---------------------------------------------------------------------------
// PrintGNBResult
// ---------------------------------------------------------------------------

void PrintGNBResult(const GNBResult& result) {
  printf("\n=== Gaussian Naive Bayes ===\n");
  printf("  Accuracy         : %.2f%%\n", result.accuracy * 100.f);
  printf("  Train GPU time   : (included in model struct)\n");
  printf("  Predict GPU time : %.4f ms\n", result.predict_gpu_ms);

  // Per-class accuracy.
  std::vector<int> per_class_correct(kNumClasses, 0);
  std::vector<int> per_class_total(kNumClasses, 0);
  for (int i = 0; i < static_cast<int>(result.true_labels.size()); ++i) {
    int gt = result.true_labels[i];
    per_class_total[gt]++;
    if (result.predictions[i] == gt) per_class_correct[gt]++;
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
  printf("============================\n\n");
}
