// Copyright 2024 Iris GPU ML Pipeline
//
// GPU-accelerated confusion matrix (atomic kernel) + host-side
// precision / recall / F1 computation.

#ifndef SRC_METRICS_H_
#define SRC_METRICS_H_

#include <string>
#include <vector>
#include "common.h"

// ---------------------------------------------------------------------------
// ConfusionMatrix
// ---------------------------------------------------------------------------

struct ConfusionMatrix {
  // Flattened row-major: data[true_label * n_classes + pred_label]
  std::vector<int> data;
  int n_classes;
};

// ---------------------------------------------------------------------------
// ClassificationMetrics
// ---------------------------------------------------------------------------

struct ClassificationMetrics {
  ConfusionMatrix confusion;
  std::vector<float> precision;   // per-class
  std::vector<float> recall;      // per-class
  std::vector<float> f1;          // per-class
  float macro_precision;
  float macro_recall;
  float macro_f1;
  float weighted_f1;              // support-weighted F1
  double gpu_ms;                  // time for atomic CM kernel
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// ComputeMetricsGpu – builds confusion matrix via atomic GPU kernel, then
// derives P/R/F1 on host.
//   d_predictions, d_true_labels: device int arrays [n_samples]
void ComputeMetricsGpu(const int* d_predictions,
                       const int* d_true_labels,
                       int n_samples, int n_classes,
                       ClassificationMetrics* metrics);

// Host-vector convenience overload (uploads to device internally).
void ComputeMetricsGpu(const std::vector<int>& predictions,
                       const std::vector<int>& true_labels,
                       int n_classes,
                       ClassificationMetrics* metrics);

// PrintClassificationMetrics – stdout summary.
void PrintClassificationMetrics(const ClassificationMetrics& m);

// SaveConfusionMatrixCsv – write confusion matrix to CSV.
bool SaveConfusionMatrixCsv(const std::string& path,
                             const ConfusionMatrix& cm);

// SaveMetricsCsv – write per-class P/R/F1 table to CSV.
bool SaveMetricsCsv(const std::string& path,
                    const ClassificationMetrics& m);

#endif  // SRC_METRICS_H_
