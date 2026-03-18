// Copyright 2024 Iris GPU ML Pipeline
//
// GPU-accelerated Gaussian Naive Bayes (GNB) classifier.
// Training  : Thrust reductions compute per-class mean and variance.
// Prediction: custom CUDA kernel computes log-likelihood for each sample.
//
// Model: P(class=c | x) ∝ P(class=c) * ∏_j N(x_j | μ_{c,j}, σ²_{c,j})
// Log-likelihood: log P(x|c) = -0.5 * Σ_j [log(2π σ²_{c,j}) + (x_j−μ_{c,j})²/σ²_{c,j}]

#ifndef SRC_GNB_H_
#define SRC_GNB_H_

#include <vector>
#include "common.h"

// GNBModel – trained Gaussian Naive Bayes parameters (host storage).
struct GNBModel {
  // Row-major [n_classes x n_features]
  std::vector<float> class_mean;    // per-class feature means
  std::vector<float> class_var;     // per-class feature variances (with floor)
  std::vector<float> log_prior;     // log P(class=c)
  int   n_classes;
  int   n_features;
  double train_gpu_ms;
};

// GNBResult – output from PredictGNB.
struct GNBResult {
  std::vector<int>   predictions;  // predicted class per sample
  std::vector<int>   true_labels;  // copy of ground-truth labels
  float              accuracy;
  double             predict_gpu_ms;
};

// TrainGNB – uses Thrust reductions to compute per-class mean and variance
// on the GPU.  d_data must already be normalised.
void TrainGNB(const float* d_data,
              const std::vector<int>& h_labels,
              int n_samples, int n_features, int n_classes,
              GNBModel* model);

// PredictGNB – kernel computes log-likelihood for each (sample, class) pair.
void PredictGNB(const float* d_data,
                const GNBModel& model,
                const std::vector<int>& h_true_labels,
                int n_samples,
                GNBResult* result);

// PrintGNBResult – stdout summary.
void PrintGNBResult(const GNBResult& result);

#endif  // SRC_GNB_H_
