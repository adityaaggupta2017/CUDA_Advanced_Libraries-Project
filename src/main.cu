// Copyright 2024 Iris GPU ML Pipeline
//
// Main entry point for the GPU-accelerated Iris ML Pipeline.
// Supports K-Means (multi-run + elbow sweep), KNN (3 distance metrics),
// PCA, Gaussian Naive Bayes, Silhouette scoring, and Confusion Matrix / F1.
//
// Usage:
//   ./iris_gpu --data <iris.data> --algorithm <kmeans|knn|pca|gnb|all>
//              [--k N] [--knn-k N] [--knn-metric euclidean|manhattan|cosine]
//              [--components N] [--iterations N] [--runs N]
//              [--elbow] [--k-max N] [--seed N] [--output <dir>]

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "common.h"
#include "data_loader.h"
#include "normalizer.h"
#include "kmeans.h"
#include "knn.h"
#include "pca.h"
#include "silhouette.h"
#include "gnb.h"
#include "metrics.h"

// ---------------------------------------------------------------------------
// CLI Configuration
// ---------------------------------------------------------------------------

struct Config {
  std::string    data_path    = "iris/iris.data";
  std::string    algorithm    = "all";
  std::string    output_dir   = "output";
  int            k_clusters   = 3;
  int            k_neighbors  = 5;
  int            n_components = 2;
  int            max_iter     = 300;
  unsigned       seed         = 42;
  DistanceMetric knn_metric   = DistanceMetric::kEuclidean;
  int            n_runs       = 5;    // K-Means restarts
  bool           run_elbow    = false;
  int            k_max        = 8;    // elbow sweep upper bound
  int            n_runs_elbow = 3;    // restarts per k in elbow sweep
};

static void PrintUsage(const char* prog) {
  printf("Usage: %s [OPTIONS]\n\n", prog);
  printf("Options:\n");
  printf("  --data        <path>   Iris CSV data file  (default: iris/iris.data)\n");
  printf("  --algorithm   <name>   kmeans|knn|pca|gnb|all  (default: all)\n");
  printf("  --k           <int>    K-Means clusters       (default: 3)\n");
  printf("  --knn-k       <int>    KNN neighbours         (default: 5)\n");
  printf("  --knn-metric  <name>   euclidean|manhattan|cosine (default: euclidean)\n");
  printf("  --components  <int>    PCA components         (default: 2)\n");
  printf("  --iterations  <int>    Max K-Means iterations (default: 300)\n");
  printf("  --runs        <int>    K-Means restarts       (default: 5)\n");
  printf("  --elbow                Run k=1..k-max elbow sweep\n");
  printf("  --k-max       <int>    Elbow sweep max k      (default: 8)\n");
  printf("  --seed        <uint>   RNG seed               (default: 42)\n");
  printf("  --output      <dir>    Output directory       (default: output)\n");
  printf("  --help                 Show this message\n\n");
  printf("Example:\n");
  printf("  %s --data iris/iris.data --algorithm all --knn-metric cosine --elbow\n",
         prog);
}

static Config ParseArgs(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      exit(EXIT_SUCCESS);
    } else if (arg == "--data" && i + 1 < argc) {
      cfg.data_path = argv[++i];
    } else if (arg == "--algorithm" && i + 1 < argc) {
      cfg.algorithm = argv[++i];
    } else if (arg == "--k" && i + 1 < argc) {
      cfg.k_clusters = std::atoi(argv[++i]);
    } else if (arg == "--knn-k" && i + 1 < argc) {
      cfg.k_neighbors = std::atoi(argv[++i]);
    } else if (arg == "--knn-metric" && i + 1 < argc) {
      std::string m = argv[++i];
      if      (m == "euclidean") cfg.knn_metric = DistanceMetric::kEuclidean;
      else if (m == "manhattan") cfg.knn_metric = DistanceMetric::kManhattan;
      else if (m == "cosine")    cfg.knn_metric = DistanceMetric::kCosine;
      else {
        fprintf(stderr, "Unknown knn-metric '%s'\n", m.c_str());
        exit(EXIT_FAILURE);
      }
    } else if (arg == "--components" && i + 1 < argc) {
      cfg.n_components = std::atoi(argv[++i]);
    } else if (arg == "--iterations" && i + 1 < argc) {
      cfg.max_iter = std::atoi(argv[++i]);
    } else if (arg == "--runs" && i + 1 < argc) {
      cfg.n_runs = std::atoi(argv[++i]);
    } else if (arg == "--elbow") {
      cfg.run_elbow = true;
    } else if (arg == "--k-max" && i + 1 < argc) {
      cfg.k_max = std::atoi(argv[++i]);
    } else if (arg == "--seed" && i + 1 < argc) {
      cfg.seed = static_cast<unsigned>(std::atoi(argv[++i]));
    } else if (arg == "--output" && i + 1 < argc) {
      cfg.output_dir = argv[++i];
    } else {
      fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
      PrintUsage(argv[0]);
      exit(EXIT_FAILURE);
    }
  }
  return cfg;
}

// ---------------------------------------------------------------------------
// GPU device info
// ---------------------------------------------------------------------------

static void PrintGpuInfo() {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  printf("=== GPU Device ===\n");
  printf("  Name          : %s\n", prop.name);
  printf("  Compute cap.  : %d.%d\n", prop.major, prop.minor);
  printf("  Global mem    : %.1f GB\n",
         static_cast<double>(prop.totalGlobalMem) /
             (1024.0 * 1024.0 * 1024.0));
  printf("  SMs           : %d\n", prop.multiProcessorCount);
  printf("  Max threads/SM: %d\n", prop.maxThreadsPerMultiProcessor);
  printf("==================\n\n");
}

// ---------------------------------------------------------------------------
// Output path helper
// ---------------------------------------------------------------------------

static std::string OutPath(const std::string& dir, const std::string& name) {
  return dir + "/" + name;
}

// ---------------------------------------------------------------------------
// Timing summary
// ---------------------------------------------------------------------------

static bool SaveTimingCsv(const std::string& path,
                           const std::vector<std::string>& names,
                           const std::vector<double>& times_ms) {
  std::ofstream f(path);
  if (!f.is_open()) return false;
  f << "algorithm,gpu_time_ms\n";
  for (size_t i = 0; i < names.size(); ++i)
    f << names[i] << "," << times_ms[i] << "\n";
  return true;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  Config cfg = ParseArgs(argc, argv);

  printf("============================================================\n");
  printf("  GPU-Accelerated Iris ML Pipeline\n");
  printf("  CUDA Advanced Libraries Capstone Project\n");
  printf("============================================================\n\n");

  PrintGpuInfo();

  const std::string alg = cfg.algorithm;
  bool run_kmeans = (alg == "kmeans" || alg == "all");
  bool run_knn    = (alg == "knn"    || alg == "all");
  bool run_pca    = (alg == "pca"    || alg == "all");
  bool run_gnb    = (alg == "gnb"    || alg == "all");

  if (!run_kmeans && !run_knn && !run_pca && !run_gnb) {
    fprintf(stderr,
            "Unknown algorithm '%s'. Use kmeans, knn, pca, gnb, or all.\n",
            alg.c_str());
    return EXIT_FAILURE;
  }

  // ---- Load data ----
  IrisDataset dataset;
  if (!LoadIrisData(cfg.data_path, &dataset)) {
    fprintf(stderr, "Failed to load '%s'\n", cfg.data_path.c_str());
    return EXIT_FAILURE;
  }
  PrintDatasetSummary(dataset);

  const int n  = dataset.n_samples;
  const int nf = dataset.n_features;

  // ---- Upload raw features to device ----
  float* d_data = nullptr;
  CUDA_CHECK(cudaMalloc(&d_data, n * nf * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_data, dataset.features.data(),
                        n * nf * sizeof(float), cudaMemcpyHostToDevice));

  // Save raw CSV.
  std::string raw_hdr = "sepal_length_cm,sepal_width_cm,"
                         "petal_length_cm,petal_width_cm,label";
  SaveLabeledCsv(OutPath(cfg.output_dir, "raw_data.csv"),
                 dataset.features, n, nf, dataset.labels, raw_hdr);
  printf("[main] Saved raw data        -> %s\n",
         OutPath(cfg.output_dir, "raw_data.csv").c_str());

  // ---- GPU Z-score normalisation ----
  printf("\n[main] Running GPU Z-score normalisation (Thrust)...\n");
  NormStats norm_stats;
  NormalizeGpu(d_data, n, nf, &norm_stats);
  PrintNormStats(norm_stats, nf);

  {
    std::ofstream ns(OutPath(cfg.output_dir, "norm_stats.csv"));
    ns << "feature,mean,stddev\n";
    for (int j = 0; j < nf; ++j)
      ns << kFeatureNames[j] << "," << norm_stats.mean[j]
         << "," << norm_stats.stddev[j] << "\n";
  }
  std::vector<float> h_norm(n * nf);
  CUDA_CHECK(cudaMemcpy(h_norm.data(), d_data,
                        n * nf * sizeof(float), cudaMemcpyDeviceToHost));
  std::string norm_hdr = "norm_sepal_length,norm_sepal_width,"
                          "norm_petal_length,norm_petal_width,label";
  SaveLabeledCsv(OutPath(cfg.output_dir, "normalised_data.csv"),
                 h_norm, n, nf, dataset.labels, norm_hdr);
  printf("[main] Saved normalised data -> %s\n",
         OutPath(cfg.output_dir, "normalised_data.csv").c_str());

  // ---- Create library handles ----
  cublasHandle_t cublas_handle;
  CUBLAS_CHECK(cublasCreate(&cublas_handle));
  cusolverDnHandle_t cusolver_handle;
  CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

  std::vector<std::string> timing_names;
  std::vector<double>      timing_ms;

  // ==================================================================
  // K-MEANS (multi-run best-of-N)
  // ==================================================================
  if (run_kmeans) {
    printf("[main] Running GPU K-Means (k=%d, %d restarts, seed=%u)...\n",
           cfg.k_clusters, cfg.n_runs, cfg.seed);

    KMeansResult km;
    RunKMeansMulti(d_data, n, nf, cfg.k_clusters, cfg.max_iter,
                   cfg.seed, cfg.n_runs, &km);
    PrintKMeansResult(km, cfg.k_clusters, nf);

    timing_names.push_back("kmeans_multi");
    timing_ms.push_back(km.gpu_ms);

    // Cluster purity.
    int correct = 0;
    std::vector<std::vector<int>> cl(cfg.k_clusters,
                                     std::vector<int>(kNumClasses, 0));
    for (int i = 0; i < n; ++i) cl[km.labels[i]][dataset.labels[i]]++;
    for (int c = 0; c < cfg.k_clusters; ++c)
      correct += *std::max_element(cl[c].begin(), cl[c].end());
    float purity = static_cast<float>(correct) / static_cast<float>(n);
    printf("[main] K-Means cluster purity: %.2f%%\n\n", purity * 100.f);

    // Save outputs.
    SaveIntCsv(OutPath(cfg.output_dir, "kmeans_labels.csv"),
               km.labels, "cluster_id");
    std::string cent_hdr;
    for (int f = 0; f < nf; ++f) {
      if (f) cent_hdr += ",";
      cent_hdr += std::string("centroid_") + kFeatureNames[f];
    }
    SaveFloatCsv(OutPath(cfg.output_dir, "kmeans_centroids.csv"),
                 km.centroids, cfg.k_clusters, nf, cent_hdr);
    SaveLabeledCsv(OutPath(cfg.output_dir, "kmeans_result.csv"),
                   h_norm, n, nf, km.labels,
                   "norm_sepal_length,norm_sepal_width,"
                   "norm_petal_length,norm_petal_width,cluster_id");
    {
      std::ofstream sum(OutPath(cfg.output_dir, "kmeans_summary.csv"));
      sum << "metric,value\n";
      sum << "k,"            << cfg.k_clusters  << "\n";
      sum << "n_runs,"       << cfg.n_runs       << "\n";
      sum << "iterations,"   << km.iterations    << "\n";
      sum << "inertia,"      << km.inertia       << "\n";
      sum << "purity,"       << purity           << "\n";
      sum << "total_gpu_ms," << km.gpu_ms        << "\n";
    }
    printf("[main] Saved K-Means outputs -> output/kmeans_*\n");

    // ------------------------------------------------------------------
    // Silhouette coefficient for the K-Means clustering
    // ------------------------------------------------------------------
    if (cfg.k_clusters >= 2) {
      printf("[main] Computing silhouette coefficient (cuBLAS + custom kernels)...\n");
      int* d_km_labels = nullptr;
      CUDA_CHECK(cudaMalloc(&d_km_labels, n * sizeof(int)));
      CUDA_CHECK(cudaMemcpy(d_km_labels, km.labels.data(),
                            n * sizeof(int), cudaMemcpyHostToDevice));

      SilhouetteResult sil;
      ComputeSilhouette(d_data, d_km_labels, n, nf, cfg.k_clusters,
                        cublas_handle, &sil);
      CUDA_CHECK(cudaFree(d_km_labels));
      PrintSilhouetteResult(sil, cfg.k_clusters);

      timing_names.push_back("silhouette");
      timing_ms.push_back(sil.gpu_ms);

      // Save per-sample silhouette scores.
      {
        std::ofstream sf(OutPath(cfg.output_dir, "silhouette_scores.csv"));
        sf << "sample_id,cluster_id,silhouette_score\n";
        for (int i = 0; i < n; ++i)
          sf << i << "," << km.labels[i] << ","
             << sil.per_sample[i] << "\n";
      }
      {
        std::ofstream sf(OutPath(cfg.output_dir, "silhouette_summary.csv"));
        sf << "metric,value\n";
        sf << "mean_silhouette," << sil.mean_score << "\n";
        for (int c = 0; c < cfg.k_clusters; ++c)
          sf << "cluster_" << c << "_silhouette," << sil.per_cluster[c] << "\n";
        sf << "gpu_ms," << sil.gpu_ms << "\n";
      }
      printf("[main] Saved silhouette outputs -> output/silhouette_*\n");
    }
  }

  // ==================================================================
  // ELBOW SWEEP
  // ==================================================================
  if (cfg.run_elbow) {
    printf("[main] Running elbow sweep (k=1..%d, %d runs each)...\n",
           cfg.k_max, cfg.n_runs_elbow);
    ElbowResult elbow;
    RunElbowSweep(d_data, n, nf, cfg.k_max, cfg.max_iter,
                  cfg.seed, cfg.n_runs_elbow, cublas_handle, &elbow);
    PrintElbowResult(elbow);
    timing_names.push_back("elbow_sweep");
    timing_ms.push_back(elbow.total_gpu_ms);

    // Save elbow CSV.
    {
      std::ofstream ef(OutPath(cfg.output_dir, "elbow_method.csv"));
      ef << "k,inertia,silhouette_score\n";
      for (size_t i = 0; i < elbow.k_values.size(); ++i)
        ef << elbow.k_values[i] << "," << elbow.inertias[i]
           << "," << elbow.silhouettes[i] << "\n";
    }
    printf("[main] Saved elbow data          -> %s\n",
           OutPath(cfg.output_dir, "elbow_method.csv").c_str());
  }

  // ==================================================================
  // KNN CLASSIFICATION
  // ==================================================================
  if (run_knn) {
    // Run all three metrics when algorithm == "all".
    std::vector<DistanceMetric> metrics_to_run;
    if (alg == "all") {
      metrics_to_run = {DistanceMetric::kEuclidean,
                        DistanceMetric::kManhattan,
                        DistanceMetric::kCosine};
    } else {
      metrics_to_run = {cfg.knn_metric};
    }

    for (DistanceMetric m : metrics_to_run) {
      printf("[main] Running GPU KNN (k=%d, metric=%s, LOO)...\n",
             cfg.k_neighbors, DistanceMetricName(m));
      KNNResult knn_result;
      RunKNN(d_data, dataset.labels, n, nf, cfg.k_neighbors,
             cublas_handle, &knn_result, m);
      PrintKNNResult(knn_result, cfg.k_neighbors);

      timing_names.push_back(std::string("knn_") + DistanceMetricName(m));
      timing_ms.push_back(knn_result.gpu_ms);

      // Confusion matrix + F1.
      ClassificationMetrics knn_metrics;
      ComputeMetricsGpu(knn_result.predictions, knn_result.true_labels,
                        kNumClasses, &knn_metrics);
      PrintClassificationMetrics(knn_metrics);

      // Save outputs.
      std::string suffix = std::string("_") + DistanceMetricName(m);
      {
        std::ofstream f(OutPath(cfg.output_dir,
                                "knn_predictions" + suffix + ".csv"));
        f << "true_label,predicted_label,correct\n";
        for (int i = 0; i < n; ++i) {
          int gt = knn_result.true_labels[i], pr = knn_result.predictions[i];
          f << gt << "," << pr << "," << (gt == pr ? 1 : 0) << "\n";
        }
      }
      SaveConfusionMatrixCsv(
          OutPath(cfg.output_dir, "knn_confusion" + suffix + ".csv"),
          knn_metrics.confusion);
      SaveMetricsCsv(
          OutPath(cfg.output_dir, "knn_metrics" + suffix + ".csv"),
          knn_metrics);
      {
        std::ofstream sum(OutPath(cfg.output_dir,
                                  "knn_summary" + suffix + ".csv"));
        sum << "metric,value\n";
        sum << "k,"           << cfg.k_neighbors       << "\n";
        sum << "distance,"    << DistanceMetricName(m)  << "\n";
        sum << "accuracy,"    << knn_result.accuracy    << "\n";
        sum << "macro_f1,"    << knn_metrics.macro_f1   << "\n";
        sum << "weighted_f1," << knn_metrics.weighted_f1 << "\n";
        sum << "gpu_ms,"      << knn_result.gpu_ms       << "\n";
      }
      printf("[main] Saved KNN (%s) outputs -> output/knn_*%s.*\n",
             DistanceMetricName(m), suffix.c_str());
    }
  }

  // ==================================================================
  // PCA
  // ==================================================================
  if (run_pca) {
    int comp = std::min(cfg.n_components, nf);
    printf("[main] Running GPU PCA (%d components, cuBLAS+cuSolver)...\n",
           comp);
    PCAResult pca_result;
    RunPCA(d_data, n, nf, comp, cublas_handle, cusolver_handle, &pca_result);
    PrintPCAResult(pca_result, nf);
    timing_names.push_back("pca");
    timing_ms.push_back(pca_result.gpu_ms);

    // Save projection.
    {
      std::ofstream f(OutPath(cfg.output_dir, "pca_projection.csv"));
      f << "sample_id";
      for (int c = 0; c < comp; ++c) f << ",PC" << (c + 1);
      f << ",true_label\n";
      for (int i = 0; i < n; ++i) {
        f << i;
        for (int c = 0; c < comp; ++c)
          f << "," << pca_result.projected[i * comp + c];
        f << "," << dataset.labels[i] << "\n";
      }
    }
    // Save components / loadings.
    {
      std::ofstream f(OutPath(cfg.output_dir, "pca_components.csv"));
      f << "component";
      for (int j = 0; j < nf; ++j) f << "," << kFeatureNames[j];
      f << ",eigenvalue,explained_variance_ratio\n";
      for (int c = 0; c < comp; ++c) {
        f << "PC" << (c + 1);
        for (int j = 0; j < nf; ++j)
          f << "," << pca_result.components[c * nf + j];
        f << "," << pca_result.eigenvalues[c]
          << "," << pca_result.explained_variance_ratio[c] << "\n";
      }
    }
    {
      std::ofstream sum(OutPath(cfg.output_dir, "pca_summary.csv"));
      sum << "metric,value\n";
      sum << "n_components," << comp << "\n";
      float cum = 0.f;
      for (int c = 0; c < comp; ++c) {
        cum += pca_result.explained_variance_ratio[c];
        sum << "PC" << (c + 1) << "_eigenvalue,"
            << pca_result.eigenvalues[c] << "\n";
        sum << "PC" << (c + 1) << "_explained_var,"
            << pca_result.explained_variance_ratio[c] << "\n";
      }
      sum << "cumulative_variance," << cum << "\n";
      sum << "gpu_ms," << pca_result.gpu_ms << "\n";
    }
    printf("[main] Saved PCA outputs         -> output/pca_*\n");
  }

  // ==================================================================
  // GAUSSIAN NAIVE BAYES
  // ==================================================================
  if (run_gnb) {
    printf("[main] Training GPU Gaussian Naive Bayes (Thrust reductions)...\n");
    GNBModel gnb_model;
    TrainGNB(d_data, dataset.labels, n, nf, kNumClasses, &gnb_model);

    printf("[main] Predicting with GNB (log-likelihood kernel)...\n");
    GNBResult gnb_result;
    PredictGNB(d_data, gnb_model, dataset.labels, n, &gnb_result);
    PrintGNBResult(gnb_result);

    timing_names.push_back("gnb_train");
    timing_ms.push_back(gnb_model.train_gpu_ms);
    timing_names.push_back("gnb_predict");
    timing_ms.push_back(gnb_result.predict_gpu_ms);

    // Confusion matrix + F1.
    ClassificationMetrics gnb_metrics;
    ComputeMetricsGpu(gnb_result.predictions, gnb_result.true_labels,
                      kNumClasses, &gnb_metrics);
    PrintClassificationMetrics(gnb_metrics);

    // Save.
    {
      std::ofstream f(OutPath(cfg.output_dir, "gnb_predictions.csv"));
      f << "true_label,predicted_label,correct\n";
      for (int i = 0; i < n; ++i) {
        int gt = gnb_result.true_labels[i], pr = gnb_result.predictions[i];
        f << gt << "," << pr << "," << (gt == pr ? 1 : 0) << "\n";
      }
    }
    SaveConfusionMatrixCsv(OutPath(cfg.output_dir, "gnb_confusion.csv"),
                           gnb_metrics.confusion);
    SaveMetricsCsv(OutPath(cfg.output_dir, "gnb_metrics.csv"), gnb_metrics);

    // Save GNB model parameters for inspection.
    {
      std::ofstream mf(OutPath(cfg.output_dir, "gnb_model.csv"));
      mf << "class";
      for (int f = 0; f < nf; ++f)
        mf << ",mean_" << kFeatureNames[f];
      for (int f = 0; f < nf; ++f)
        mf << ",var_" << kFeatureNames[f];
      mf << ",log_prior\n";
      for (int c = 0; c < kNumClasses; ++c) {
        mf << kClassNames[c];
        for (int f = 0; f < nf; ++f)
          mf << "," << gnb_model.class_mean[c * nf + f];
        for (int f = 0; f < nf; ++f)
          mf << "," << gnb_model.class_var[c * nf + f];
        mf << "," << gnb_model.log_prior[c] << "\n";
      }
    }
    {
      std::ofstream sum(OutPath(cfg.output_dir, "gnb_summary.csv"));
      sum << "metric,value\n";
      sum << "accuracy,"      << gnb_result.accuracy        << "\n";
      sum << "macro_f1,"      << gnb_metrics.macro_f1       << "\n";
      sum << "weighted_f1,"   << gnb_metrics.weighted_f1    << "\n";
      sum << "train_gpu_ms,"  << gnb_model.train_gpu_ms     << "\n";
      sum << "predict_gpu_ms," << gnb_result.predict_gpu_ms << "\n";
    }
    printf("[main] Saved GNB outputs         -> output/gnb_*\n");
  }

  // ==================================================================
  // Save timing CSV and summary report
  // ==================================================================
  SaveTimingCsv(OutPath(cfg.output_dir, "timing_results.csv"),
                timing_names, timing_ms);
  printf("[main] Saved timing results      -> %s\n",
         OutPath(cfg.output_dir, "timing_results.csv").c_str());

  {
    std::ofstream rep(OutPath(cfg.output_dir, "summary_report.txt"));
    rep << "GPU-Accelerated Iris ML Pipeline - Summary Report\n";
    rep << "==================================================\n\n";
    rep << "Dataset   : " << cfg.data_path  << "\n";
    rep << "Samples   : " << n              << "\n";
    rep << "Features  : " << nf             << "\n\n";
    rep << "Algorithms: " << cfg.algorithm  << "\n\n";
    rep << "GPU timing (ms):\n";
    for (size_t i = 0; i < timing_names.size(); ++i)
      rep << "  " << timing_names[i] << " : " << timing_ms[i] << " ms\n";
    rep << "\nOutput directory: " << cfg.output_dir << "/\n";
  }
  printf("[main] Saved summary report      -> %s\n",
         OutPath(cfg.output_dir, "summary_report.txt").c_str());

  // ---- Cleanup ----
  CUDA_CHECK(cudaFree(d_data));
  CUBLAS_CHECK(cublasDestroy(cublas_handle));
  CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));

  printf("\n[main] Pipeline complete.\n");
  return EXIT_SUCCESS;
}
