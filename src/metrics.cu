// Copyright 2024 Iris GPU ML Pipeline
//
// GPU confusion matrix via atomicAdd kernel; host-side P/R/F1 derivation.

#include "metrics.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

// ---------------------------------------------------------------------------
// Kernel: build confusion matrix with one atomicAdd per sample.
// ---------------------------------------------------------------------------

__global__ void ConfusionMatrixKernel(const int* __restrict__ predictions,
                                       const int* __restrict__ true_labels,
                                       int* __restrict__ cm,
                                       int n_samples, int n_classes) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_samples) return;
  int row = true_labels[i];
  int col = predictions[i];
  if (row >= 0 && row < n_classes && col >= 0 && col < n_classes)
    atomicAdd(&cm[row * n_classes + col], 1);
}

// ---------------------------------------------------------------------------
// Internal helper: derive P/R/F1 from a host confusion matrix.
// ---------------------------------------------------------------------------

static void DeriveMetrics(ClassificationMetrics* m) {
  const int nc = m->confusion.n_classes;
  m->precision.assign(nc, 0.f);
  m->recall.assign(nc,    0.f);
  m->f1.assign(nc,        0.f);

  const auto& cm = m->confusion.data;
  std::vector<int> support(nc, 0);

  for (int c = 0; c < nc; ++c) {
    int tp       = cm[c * nc + c];
    int col_sum  = 0;  // predicted as c
    int row_sum  = 0;  // actually c
    for (int r = 0; r < nc; ++r) col_sum += cm[r * nc + c];
    for (int p = 0; p < nc; ++p) row_sum += cm[c * nc + p];
    support[c] = row_sum;

    float prec = (col_sum > 0) ? static_cast<float>(tp) / col_sum : 0.f;
    float rec  = (row_sum > 0) ? static_cast<float>(tp) / row_sum : 0.f;
    float f1   = (prec + rec > 0.f) ? 2.f * prec * rec / (prec + rec) : 0.f;

    m->precision[c] = prec;
    m->recall[c]    = rec;
    m->f1[c]        = f1;
  }

  // Macro averages (unweighted mean over classes).
  m->macro_precision = std::accumulate(m->precision.begin(),
                                       m->precision.end(), 0.f) / nc;
  m->macro_recall    = std::accumulate(m->recall.begin(),
                                       m->recall.end(), 0.f) / nc;
  m->macro_f1        = std::accumulate(m->f1.begin(),
                                       m->f1.end(), 0.f) / nc;

  // Weighted F1 (weight = class support / total samples).
  int total = std::accumulate(support.begin(), support.end(), 0);
  float wf1 = 0.f;
  for (int c = 0; c < nc; ++c)
    wf1 += m->f1[c] * static_cast<float>(support[c]);
  m->weighted_f1 = (total > 0) ? wf1 / static_cast<float>(total) : 0.f;
}

// ---------------------------------------------------------------------------
// ComputeMetricsGpu (device pointers)
// ---------------------------------------------------------------------------

void ComputeMetricsGpu(const int* d_predictions,
                       const int* d_true_labels,
                       int n_samples, int n_classes,
                       ClassificationMetrics* metrics) {
  assert(metrics != nullptr);

  cudaEvent_t t0, t1;
  CUDA_CHECK(cudaEventCreate(&t0));
  CUDA_CHECK(cudaEventCreate(&t1));

  // Allocate device CM (zero-initialised).
  int* d_cm = nullptr;
  CUDA_CHECK(cudaMalloc(&d_cm, n_classes * n_classes * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_cm, 0, n_classes * n_classes * sizeof(int)));

  CUDA_CHECK(cudaEventRecord(t0));

  int threads = 256;
  int blocks  = (n_samples + threads - 1) / threads;
  ConfusionMatrixKernel<<<blocks, threads>>>(d_predictions, d_true_labels,
                                             d_cm, n_samples, n_classes);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(t1));
  CUDA_CHECK(cudaEventSynchronize(t1));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

  // Copy CM to host.
  metrics->confusion.n_classes = n_classes;
  metrics->confusion.data.resize(n_classes * n_classes);
  CUDA_CHECK(cudaMemcpy(metrics->confusion.data.data(), d_cm,
                        n_classes * n_classes * sizeof(int),
                        cudaMemcpyDeviceToHost));
  metrics->gpu_ms = static_cast<double>(ms);

  DeriveMetrics(metrics);

  CUDA_CHECK(cudaFree(d_cm));
  CUDA_CHECK(cudaEventDestroy(t0));
  CUDA_CHECK(cudaEventDestroy(t1));
}

// ---------------------------------------------------------------------------
// ComputeMetricsGpu (host vectors)
// ---------------------------------------------------------------------------

void ComputeMetricsGpu(const std::vector<int>& predictions,
                       const std::vector<int>& true_labels,
                       int n_classes,
                       ClassificationMetrics* metrics) {
  int n = static_cast<int>(predictions.size());
  assert(static_cast<int>(true_labels.size()) == n);

  int* d_pred  = nullptr;
  int* d_true  = nullptr;
  CUDA_CHECK(cudaMalloc(&d_pred, n * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_true, n * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_pred, predictions.data(),
                        n * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_true, true_labels.data(),
                        n * sizeof(int), cudaMemcpyHostToDevice));

  ComputeMetricsGpu(d_pred, d_true, n, n_classes, metrics);

  CUDA_CHECK(cudaFree(d_pred));
  CUDA_CHECK(cudaFree(d_true));
}

// ---------------------------------------------------------------------------
// PrintClassificationMetrics
// ---------------------------------------------------------------------------

void PrintClassificationMetrics(const ClassificationMetrics& m) {
  const int nc = m.confusion.n_classes;
  printf("\n=== Classification Metrics ===\n");

  // Confusion matrix.
  printf("  Confusion Matrix (rows=true, cols=predicted):\n");
  printf("  %20s", "");
  for (int c = 0; c < nc; ++c)
    printf("  %-18s", kClassNames[c]);
  printf("\n");
  for (int r = 0; r < nc; ++r) {
    printf("  %-20s", kClassNames[r]);
    for (int c = 0; c < nc; ++c)
      printf("  %-18d", m.confusion.data[r * nc + c]);
    printf("\n");
  }

  // Per-class P/R/F1.
  printf("\n  %-20s  %9s  %9s  %9s\n", "Class", "Precision", "Recall", "F1");
  for (int c = 0; c < nc; ++c) {
    printf("  %-20s  %9.4f  %9.4f  %9.4f\n",
           kClassNames[c], m.precision[c], m.recall[c], m.f1[c]);
  }
  printf("  ---\n");
  printf("  %-20s  %9.4f  %9.4f  %9.4f\n",
         "Macro avg", m.macro_precision, m.macro_recall, m.macro_f1);
  printf("  %-20s  %26.4f\n", "Weighted F1", m.weighted_f1);
  printf("  GPU time (CM kernel): %.4f ms\n", m.gpu_ms);
  printf("==============================\n\n");
}

// ---------------------------------------------------------------------------
// SaveConfusionMatrixCsv / SaveMetricsCsv
// ---------------------------------------------------------------------------

bool SaveConfusionMatrixCsv(const std::string& path,
                             const ConfusionMatrix& cm) {
  std::ofstream f(path);
  if (!f.is_open()) return false;
  const int nc = cm.n_classes;
  // Header.
  f << "true_label\\pred_label";
  for (int c = 0; c < nc; ++c) f << "," << kClassNames[c];
  f << "\n";
  for (int r = 0; r < nc; ++r) {
    f << kClassNames[r];
    for (int c = 0; c < nc; ++c) f << "," << cm.data[r * nc + c];
    f << "\n";
  }
  return true;
}

bool SaveMetricsCsv(const std::string& path,
                    const ClassificationMetrics& m) {
  std::ofstream f(path);
  if (!f.is_open()) return false;
  const int nc = m.confusion.n_classes;
  f << "class,precision,recall,f1,support\n";
  for (int c = 0; c < nc; ++c) {
    int support = 0;
    for (int p = 0; p < nc; ++p)
      support += m.confusion.data[c * nc + p];
    f << kClassNames[c] << "," << m.precision[c] << ","
      << m.recall[c] << "," << m.f1[c] << "," << support << "\n";
  }
  f << "macro_avg," << m.macro_precision << ","
    << m.macro_recall << "," << m.macro_f1 << ",\n";
  f << "weighted_avg,,,," << m.weighted_f1 << ",\n";
  return true;
}
