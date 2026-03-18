// Copyright 2024 Iris GPU ML Pipeline
//
// Main entry point for the GPU-accelerated Iris ML Pipeline.
// Supports K-means clustering, KNN classification, and PCA via CLI flags.
//
// Usage:
//   ./iris_gpu --data <path/to/iris.data> --algorithm <kmeans|knn|pca|all>
//              [--k <clusters>] [--knn-k <neighbours>]
//              [--components <n>] [--iterations <max>]
//              [--seed <uint>] [--output <dir>]

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

// ---------------------------------------------------------------------------
// CLI argument parsing
// ---------------------------------------------------------------------------

struct Config {
  std::string data_path   = "iris/iris.data";
  std::string algorithm   = "all";   // kmeans | knn | pca | all
  std::string output_dir  = "output";
  int         k_clusters  = 3;
  int         k_neighbors = 5;
  int         n_components = 2;
  int         max_iter    = 300;
  unsigned    seed        = 42;
};

static void PrintUsage(const char* prog) {
  printf("Usage: %s [OPTIONS]\n\n", prog);
  printf("Options:\n");
  printf("  --data        <path>   Path to iris.data  (default: iris/iris.data)\n");
  printf("  --algorithm   <name>   kmeans | knn | pca | all  (default: all)\n");
  printf("  --k           <int>    Number of K-means clusters (default: 3)\n");
  printf("  --knn-k       <int>    Number of KNN neighbours  (default: 5)\n");
  printf("  --components  <int>    PCA components to retain  (default: 2)\n");
  printf("  --iterations  <int>    Max K-means iterations    (default: 300)\n");
  printf("  --seed        <uint>   RNG seed                  (default: 42)\n");
  printf("  --output      <dir>    Output directory          (default: output)\n");
  printf("  --help                 Show this message\n\n");
  printf("Example:\n");
  printf("  %s --data iris/iris.data --algorithm all --k 3 --knn-k 5\n", prog);
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
    } else if (arg == "--components" && i + 1 < argc) {
      cfg.n_components = std::atoi(argv[++i]);
    } else if (arg == "--iterations" && i + 1 < argc) {
      cfg.max_iter = std::atoi(argv[++i]);
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
         static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));
  printf("  SMs           : %d\n", prop.multiProcessorCount);
  printf("  Max threads/SM: %d\n", prop.maxThreadsPerMultiProcessor);
  printf("==================\n\n");
}

// ---------------------------------------------------------------------------
// Timing summary CSV writer
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
// Build an output path helper
// ---------------------------------------------------------------------------

static std::string OutPath(const std::string& dir, const std::string& name) {
  return dir + "/" + name;
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

  // ---- Validate algorithm argument ----
  const std::string alg = cfg.algorithm;
  bool run_kmeans = (alg == "kmeans" || alg == "all");
  bool run_knn    = (alg == "knn"    || alg == "all");
  bool run_pca    = (alg == "pca"    || alg == "all");
  if (!run_kmeans && !run_knn && !run_pca) {
    fprintf(stderr, "Unknown algorithm '%s'. Use kmeans, knn, pca, or all.\n",
            alg.c_str());
    return EXIT_FAILURE;
  }

  // ---- Load iris data ----
  IrisDataset dataset;
  if (!LoadIrisData(cfg.data_path, &dataset)) {
    fprintf(stderr, "Failed to load data from '%s'\n", cfg.data_path.c_str());
    return EXIT_FAILURE;
  }
  PrintDatasetSummary(dataset);

  const int n  = dataset.n_samples;
  const int nf = dataset.n_features;

  // ---- Copy raw features to device ----
  float* d_data = nullptr;
  CUDA_CHECK(cudaMalloc(&d_data, n * nf * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_data, dataset.features.data(),
                        n * nf * sizeof(float), cudaMemcpyHostToDevice));

  // ---- Save raw data CSV ----
  std::string raw_header = "sepal_length_cm,sepal_width_cm,"
                            "petal_length_cm,petal_width_cm,label";
  SaveLabeledCsv(OutPath(cfg.output_dir, "raw_data.csv"),
                 dataset.features, n, nf, dataset.labels, raw_header);
  printf("[main] Saved raw data  -> %s\n",
         OutPath(cfg.output_dir, "raw_data.csv").c_str());

  // ---- GPU normalisation (Z-score via Thrust) ----
  printf("\n[main] Running GPU Z-score normalisation...\n");
  NormStats norm_stats;
  NormalizeGpu(d_data, n, nf, &norm_stats);
  PrintNormStats(norm_stats, nf);

  // Save normalisation stats.
  {
    std::ofstream ns(OutPath(cfg.output_dir, "norm_stats.csv"));
    ns << "feature,mean,stddev\n";
    for (int j = 0; j < nf; ++j)
      ns << kFeatureNames[j] << ","
         << norm_stats.mean[j] << ","
         << norm_stats.stddev[j] << "\n";
  }

  // Save normalised data.
  std::vector<float> h_norm(n * nf);
  CUDA_CHECK(cudaMemcpy(h_norm.data(), d_data,
                        n * nf * sizeof(float), cudaMemcpyDeviceToHost));
  std::string norm_header = "norm_sepal_length,norm_sepal_width,"
                             "norm_petal_length,norm_petal_width,label";
  SaveLabeledCsv(OutPath(cfg.output_dir, "normalised_data.csv"),
                 h_norm, n, nf, dataset.labels, norm_header);
  printf("[main] Saved normalised data -> %s\n",
         OutPath(cfg.output_dir, "normalised_data.csv").c_str());

  // ---- Create cuBLAS and cuSolver handles ----
  cublasHandle_t cublas_handle;
  CUBLAS_CHECK(cublasCreate(&cublas_handle));

  cusolverDnHandle_t cusolver_handle;
  CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

  // ---- Timing vectors ----
  std::vector<std::string> timing_names;
  std::vector<double>      timing_ms;

  // ==================================================================
  // K-MEANS CLUSTERING
  // ==================================================================
  if (run_kmeans) {
    printf("[main] Running GPU K-Means (k=%d, max_iter=%d, seed=%u)...\n",
           cfg.k_clusters, cfg.max_iter, cfg.seed);

    KMeansResult km_result;
    RunKMeans(d_data, n, nf, cfg.k_clusters, cfg.max_iter, cfg.seed,
              &km_result);
    PrintKMeansResult(km_result, cfg.k_clusters, nf);

    timing_names.push_back("kmeans");
    timing_ms.push_back(km_result.gpu_ms);

    // --- Save cluster labels ---
    SaveIntCsv(OutPath(cfg.output_dir, "kmeans_labels.csv"),
               km_result.labels, "cluster_id");
    printf("[main] Saved K-Means labels    -> %s\n",
           OutPath(cfg.output_dir, "kmeans_labels.csv").c_str());

    // --- Save centroids ---
    std::string cent_hdr;
    for (int f = 0; f < nf; ++f) {
      if (f) cent_hdr += ",";
      cent_hdr += std::string("centroid_") + kFeatureNames[f];
    }
    SaveFloatCsv(OutPath(cfg.output_dir, "kmeans_centroids.csv"),
                 km_result.centroids, cfg.k_clusters, nf, cent_hdr);
    printf("[main] Saved K-Means centroids -> %s\n",
           OutPath(cfg.output_dir, "kmeans_centroids.csv").c_str());

    // --- Compute cluster purity ---
    int correct_purity = 0;
    std::vector<std::vector<int>> cluster_labels(
        cfg.k_clusters, std::vector<int>(kNumClasses, 0));
    for (int i = 0; i < n; ++i)
      cluster_labels[km_result.labels[i]][dataset.labels[i]]++;

    for (int c = 0; c < cfg.k_clusters; ++c) {
      int best = *std::max_element(cluster_labels[c].begin(),
                                   cluster_labels[c].end());
      correct_purity += best;
    }
    float purity = static_cast<float>(correct_purity) / static_cast<float>(n);
    printf("[main] K-Means cluster purity: %.2f%%\n\n", purity * 100.f);

    // Save clustering summary.
    {
      std::ofstream sum(OutPath(cfg.output_dir, "kmeans_summary.csv"));
      sum << "metric,value\n";
      sum << "k," << cfg.k_clusters << "\n";
      sum << "iterations," << km_result.iterations << "\n";
      sum << "inertia," << km_result.inertia << "\n";
      sum << "purity," << purity << "\n";
      sum << "gpu_time_ms," << km_result.gpu_ms << "\n";
    }

    // Save labeled result for visualisation.
    std::vector<float> h_labeled_data(n * (nf + 1));
    for (int i = 0; i < n; ++i) {
      for (int f = 0; f < nf; ++f)
        h_labeled_data[i * (nf + 1) + f] = h_norm[i * nf + f];
      h_labeled_data[i * (nf + 1) + nf] =
          static_cast<float>(km_result.labels[i]);
    }
    // Re-use SaveFloatCsv for the combined table.
    std::string km_vis_hdr = "norm_sepal_length,norm_sepal_width,"
                              "norm_petal_length,norm_petal_width,cluster_id";
    SaveLabeledCsv(OutPath(cfg.output_dir, "kmeans_result.csv"),
                   h_norm, n, nf, km_result.labels, km_vis_hdr);
  }

  // ==================================================================
  // KNN CLASSIFICATION (leave-one-out)
  // ==================================================================
  if (run_knn) {
    printf("[main] Running GPU KNN (k=%d, leave-one-out)...\n",
           cfg.k_neighbors);

    KNNResult knn_result;
    RunKNN(d_data, dataset.labels, n, nf, cfg.k_neighbors,
           cublas_handle, &knn_result);
    PrintKNNResult(knn_result, cfg.k_neighbors);

    timing_names.push_back("knn");
    timing_ms.push_back(knn_result.gpu_ms);

    // Save predictions.
    {
      std::ofstream f(OutPath(cfg.output_dir, "knn_predictions.csv"));
      f << "true_label,predicted_label,correct\n";
      for (int i = 0; i < n; ++i) {
        int gt   = knn_result.true_labels[i];
        int pred = knn_result.predictions[i];
        f << gt << "," << pred << "," << (gt == pred ? 1 : 0) << "\n";
      }
    }
    printf("[main] Saved KNN predictions   -> %s\n",
           OutPath(cfg.output_dir, "knn_predictions.csv").c_str());

    // Save KNN summary.
    {
      std::ofstream sum(OutPath(cfg.output_dir, "knn_summary.csv"));
      sum << "metric,value\n";
      sum << "k," << cfg.k_neighbors << "\n";
      sum << "accuracy," << knn_result.accuracy << "\n";
      sum << "gpu_time_ms," << knn_result.gpu_ms << "\n";
    }
  }

  // ==================================================================
  // PCA
  // ==================================================================
  if (run_pca) {
    int comp = std::min(cfg.n_components, nf);
    printf("[main] Running GPU PCA (%d components)...\n", comp);

    PCAResult pca_result;
    RunPCA(d_data, n, nf, comp, cublas_handle, cusolver_handle, &pca_result);
    PrintPCAResult(pca_result, nf);

    timing_names.push_back("pca");
    timing_ms.push_back(pca_result.gpu_ms);

    // Save projected data.
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
    printf("[main] Saved PCA projection    -> %s\n",
           OutPath(cfg.output_dir, "pca_projection.csv").c_str());

    // Save eigenvectors / loadings.
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
    printf("[main] Saved PCA components    -> %s\n",
           OutPath(cfg.output_dir, "pca_components.csv").c_str());

    // Save PCA summary.
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
      sum << "gpu_time_ms," << pca_result.gpu_ms << "\n";
    }
  }

  // ==================================================================
  // Save overall timing CSV
  // ==================================================================
  timing_names.insert(timing_names.begin(), "normalization");
  timing_ms.insert(timing_ms.begin(), 0.0);  // placeholder (included in GPU alloc)
  SaveTimingCsv(OutPath(cfg.output_dir, "timing_results.csv"),
                timing_names, timing_ms);
  printf("[main] Saved timing results    -> %s\n",
         OutPath(cfg.output_dir, "timing_results.csv").c_str());

  // ==================================================================
  // Write summary report
  // ==================================================================
  {
    std::ofstream rep(OutPath(cfg.output_dir, "summary_report.txt"));
    rep << "GPU-Accelerated Iris ML Pipeline - Summary Report\n";
    rep << "==================================================\n\n";
    rep << "Dataset : " << cfg.data_path << "\n";
    rep << "Samples : " << n << "\n";
    rep << "Features: " << nf << "\n\n";
    rep << "Algorithms executed: " << cfg.algorithm << "\n\n";
    rep << "GPU timing (ms):\n";
    for (size_t i = 0; i < timing_names.size(); ++i)
      rep << "  " << timing_names[i] << " : " << timing_ms[i] << " ms\n";
    rep << "\nOutput files written to: " << cfg.output_dir << "/\n";
  }
  printf("[main] Saved summary report    -> %s\n",
         OutPath(cfg.output_dir, "summary_report.txt").c_str());

  // ---- Cleanup ----
  CUDA_CHECK(cudaFree(d_data));
  CUBLAS_CHECK(cublasDestroy(cublas_handle));
  CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));

  printf("\n[main] Pipeline complete.\n");
  return EXIT_SUCCESS;
}
