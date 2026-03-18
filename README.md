# GPU-Accelerated Iris ML Pipeline

**CUDA Advanced Libraries – Capstone Project**

A high-performance machine learning pipeline implemented entirely with CUDA advanced libraries, applied to the classic UCI Iris dataset. Seven GPU-accelerated algorithms and analyses are implemented across nine source modules: **Z-Score Normalisation**, **K-Means Clustering** (multi-run + elbow sweep), **K-Nearest Neighbours** (three distance metrics), **Principal Component Analysis**, **Silhouette Coefficient**, **Gaussian Naive Bayes**, and **Confusion Matrix / F1 scoring**.

---

## Table of Contents

1. [Project Description](#project-description)
2. [GPU Libraries Used](#gpu-libraries-used)
3. [Algorithms & Implementation Details](#algorithms--implementation-details)
4. [Directory Structure](#directory-structure)
5. [Requirements](#requirements)
6. [Building](#building)
7. [Running](#running)
8. [CLI Reference](#cli-reference)
9. [Output Files](#output-files)
10. [Results](#results)
11. [Lessons Learned](#lessons-learned)

---

## Project Description

The [UCI Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris) contains 150 samples of iris flowers across three species (*Iris-setosa*, *Iris-versicolor*, *Iris-virginica*), each described by four continuous measurements (sepal length, sepal width, petal length, petal width in cm).

This project builds a complete GPU-accelerated ML analysis pipeline with the following stages:

1. **Data loading & parsing** – CSV parsing in C++; supports both `iris.data` and `bezdekIris.data`.
2. **Z-score normalisation** – Per-feature mean/stddev computed via Thrust reductions; in-place normalisation with a custom CUDA kernel.
3. **K-Means clustering** – Custom CUDA kernels (assign, accumulate, update); k-means++ seeding; best-of-N multi-run restarts; elbow sweep (k = 1..k_max) with silhouette scoring.
4. **Silhouette coefficient** – Full pairwise distance matrix via cuBLAS SGEMM; per-sample a(i) and b(i) computed by custom GPU kernels; mean score via Thrust reduction.
5. **KNN classification (leave-one-out)** – Three selectable distance metrics: Euclidean (cuBLAS SGEMM), Manhattan (L1 custom kernel), Cosine (row-normalise + SGEMM); distances sorted per row with Thrust; majority vote via custom kernel.
6. **PCA** – Covariance matrix with cuBLAS SGEMM; eigendecomposition with cuSolver `Ssyevd`; projection with SGEMM.
7. **Gaussian Naive Bayes** – Per-class mean and variance trained with Thrust reductions; prediction via log-likelihood CUDA kernel.
8. **Confusion matrix + Precision / Recall / F1** – GPU atomic confusion matrix kernel; macro and weighted F1 derived on host; applied automatically to KNN and GNB results.

All GPU timings are captured with CUDA events. Every result is saved as a CSV file.

---

## GPU Libraries Used

| Library | Used In | Purpose |
|---|---|---|
| **CUDA Runtime** | All modules | Device/memory management, custom kernels, event timing |
| **Thrust** | Normaliser, GNB, Silhouette, KNN sort | Parallel reductions (mean, variance, sum) and `sort_by_key` |
| **cuBLAS** | KNN (Euclidean & Cosine), PCA, Silhouette | SGEMM for distance matrices, covariance, and projections |
| **cuSolver** | PCA | `Ssyevd` – dense symmetric eigenvalue decomposition |

---

## Algorithms & Implementation Details

### Z-Score Normalisation — `src/normalizer.h/.cu`

For each feature column *j*:

```
μ_j    = (1/N) Σ X[i,j]
σ_j    = sqrt((1/N) Σ (X[i,j] − μ_j)²)
X[i,j] = (X[i,j] − μ_j) / σ_j
```

- `ExtractColumn` / `SquaredDiff` file-scope Thrust functors extract column slices from the row-major matrix without materialising an intermediate copy.
- `ApplyZscoreKernel` applies the transformation in-place with one thread per row.

---

### K-Means Clustering — `src/kmeans.h/.cu`

**Single run (`RunKMeans`)**

- Initialisation: **k-means++** seeding on the host (distance-proportional sampling).
- `AssignLabelsKernel` – one thread per sample, branchless Euclidean distance loop over centroids, writes nearest centroid index.
- `AccumulateCentroidsKernel` – `atomicAdd` accumulates coordinate sums and per-cluster counts.
- `UpdateCentroidsKernel` – parallel division produces new centroids; empty clusters keep their previous centroid.
- `ComputeInertiaKernel` – `atomicAdd` accumulates total squared distances.
- Convergence: host-side comparison of label arrays each iteration.

**Multi-run restarts (`RunKMeansMulti`)**

Runs `RunKMeans` `n_runs` times with seeds `base_seed, base_seed+1, …` and returns the result with the lowest inertia. Eliminates poor random initialisations.

**Elbow sweep (`RunElbowSweep`)**

Sweeps k = 1 … k_max, calling `RunKMeansMulti` at each k. For k ≥ 2, computes the silhouette score via `ComputeSilhouette`. Records inertia and silhouette vs k; reports the k with the highest silhouette as `optimal_k`.

---

### Silhouette Coefficient — `src/silhouette.h/.cu`

For each sample *i* with cluster assignment *c(i)*:

```
a(i) = mean distance to all other samples in cluster c(i)
b(i) = min over clusters c ≠ c(i) of (mean distance to samples in c)
s(i) = (b(i) − a(i)) / max(a(i), b(i))     ∈ [−1, 1]
```

**GPU pipeline:**
1. `SilSquaredNormKernel` — per-row squared L2 norms.
2. `cublasSgemm` (CUBLAS_OP_T, CUBLAS_OP_N) — computes −2·XᵀX in one call.
3. `SilAddNormsKernel` — adds ‖xᵢ‖² + ‖xⱼ‖² to produce the full n×n squared-distance matrix.
4. Transpose to row-major on host (n=150, negligible cost).
5. `ComputeAKernel` — each thread scans its cluster to compute a(i).
6. `ComputeBKernel` — each thread scans all other clusters to compute b(i).
7. `ComputeSKernel` — elementwise s(i) computation.
8. `thrust::reduce` — mean silhouette score.

---

### KNN Classification — `src/knn.h/.cu`

Three selectable distance metrics via `DistanceMetric` enum:

**Euclidean** (default)
```
‖xᵢ − xⱼ‖² = ‖xᵢ‖² + ‖xⱼ‖² − 2 xᵢ·xⱼ
```
Cross-term matrix from one `cublasSgemm`; norms from `SquaredNormKernel`; combined by `AddNormsKernel`.

**Manhattan** (`--knn-metric manhattan`)
```
d(xᵢ, xⱼ) = Σ_f |xᵢ_f − xⱼ_f|
```
`L1DistanceKernel` launches a 2D thread grid; each thread (i, j) computes the L1 distance in a feature loop and writes to the row-major output matrix.

**Cosine** (`--knn-metric cosine`)
```
d(xᵢ, xⱼ) = 1 − (xᵢ·xⱼ) / (‖xᵢ‖ · ‖xⱼ‖)
```
`RowNormalizeKernel` divides each row by its L2 norm; then one `cublasSgemm` computes the dot-product matrix; cosine distance = 1 − similarity.

**Common suffix (all metrics):**
- Diagonal set to `FLT_MAX` to exclude self in leave-one-out.
- Thrust `sort_by_key` sorts distances per query row.
- `MajorityVoteKernel` counts votes among k nearest neighbours.

---

### PCA — `src/pca.h/.cu`

1. Host-side column-mean computation (150×4 — trivial).
2. `SubtractMeanKernel` — parallel mean centering.
3. `cublasSgemm` — covariance C = (1/(n−1))·XᵀX.
4. `cusolverDnSsyevd` — symmetric eigenvalue decomposition (eigenvalues in ascending order).
5. Eigenvalues/vectors reversed to descending order on host.
6. `cublasSgemm` — project data onto top components: Z = X·V_top.

---

### Gaussian Naive Bayes — `src/gnb.h/.cu`

**Training (`TrainGNB`)**

For each class c, uses Thrust to compute:
- Per-feature mean: `thrust::reduce` over class-filtered row indices via `ExtractFeatureFunctor` (file-scope functor).
- Per-feature variance: second Thrust pass with `SqDiffFunctor`; variance floor of 1×10⁻⁶ applied.
- Log prior: `log(class_count / n_samples)`.

**Prediction (`GNBPredictKernel`)**

One thread per sample. For each class c, computes:
```
log P(c|x) = log_prior[c] − 0.5 · Σ_f [ log(2π σ²_{c,f}) + (x_f − μ_{c,f})² / σ²_{c,f} ]
```
Assigns `argmax_c log P(c|x)`.

---

### Confusion Matrix + F1 — `src/metrics.h/.cu`

**`ConfusionMatrixKernel`** — one thread per sample; single `atomicAdd` into `cm[true_label * n_classes + pred_label]`.

Host-side derivation from the confusion matrix:
```
precision[c] = TP[c] / (sum of predicted-as-c)
recall[c]    = TP[c] / (sum of actual-c)
F1[c]        = 2 · P[c] · R[c] / (P[c] + R[c])
macro_F1     = mean(F1[c])
weighted_F1  = Σ_c F1[c] · support[c] / n_samples
```

Applied automatically after every KNN and GNB run.

---

## Directory Structure

```
CUDA_Advanced_Libraries-Project/
├── iris/                     # UCI Iris dataset (input data)
│   ├── iris.data
│   ├── bezdekIris.data
│   └── iris.names
├── src/                      # CUDA/C++ source modules
│   ├── main.cu               # CLI entry point and pipeline orchestration
│   ├── common.h              # Shared error-check macros and dataset constants
│   ├── data_loader.h/.cu     # CSV parsing and CSV result file writing
│   ├── normalizer.h/.cu      # GPU Z-score normalisation (Thrust)
│   ├── kmeans.h/.cu          # GPU K-Means: single run, multi-run, elbow sweep
│   ├── knn.h/.cu             # GPU KNN: Euclidean / Manhattan / Cosine
│   ├── pca.h/.cu             # GPU PCA (cuBLAS + cuSolver)
│   ├── silhouette.h/.cu      # GPU silhouette coefficient (cuBLAS + custom kernels)
│   ├── gnb.h/.cu             # GPU Gaussian Naive Bayes (Thrust + custom kernel)
│   └── metrics.h/.cu         # GPU confusion matrix + precision/recall/F1
├── output/                   # All generated CSV artifacts and logs
│   ├── execution_log.txt
│   ├── raw_data.csv
│   ├── normalised_data.csv
│   ├── norm_stats.csv
│   ├── kmeans_labels.csv
│   ├── kmeans_centroids.csv
│   ├── kmeans_result.csv
│   ├── kmeans_summary.csv
│   ├── elbow_method.csv
│   ├── silhouette_scores.csv
│   ├── silhouette_summary.csv
│   ├── knn_predictions_euclidean.csv
│   ├── knn_predictions_manhattan.csv
│   ├── knn_predictions_cosine.csv
│   ├── knn_confusion_euclidean.csv
│   ├── knn_confusion_manhattan.csv
│   ├── knn_confusion_cosine.csv
│   ├── knn_metrics_euclidean.csv
│   ├── knn_metrics_manhattan.csv
│   ├── knn_metrics_cosine.csv
│   ├── knn_summary_euclidean.csv
│   ├── knn_summary_manhattan.csv
│   ├── knn_summary_cosine.csv
│   ├── pca_projection.csv
│   ├── pca_components.csv
│   ├── pca_summary.csv
│   ├── gnb_predictions.csv
│   ├── gnb_confusion.csv
│   ├── gnb_metrics.csv
│   ├── gnb_model.csv
│   ├── gnb_summary.csv
│   ├── timing_results.csv
│   └── summary_report.txt
├── Makefile                  # Build system
├── run.sh                    # One-shot build + run script
└── README.md                 # This file
```

---

## Requirements

| Requirement | Version tested |
|---|---|
| NVIDIA GPU | RTX A6000 (sm_86); any sm_60+ supported |
| CUDA Toolkit | 12.0 (`/usr/local/cuda`) |
| nvcc | 12.0.76 |
| g++ | ≥ 7 (C++14) |
| cuBLAS | Included with CUDA Toolkit |
| cuSolver | Included with CUDA Toolkit |
| Thrust | Included with CUDA Toolkit |

**Install CUDA Toolkit** (Ubuntu/Debian):
```bash
# Download installer from https://developer.nvidia.com/cuda-downloads
# or via package manager:
sudo apt-get install -y cuda-toolkit-12-0
```

No Python, pip, or any third-party libraries are required.

---

## Building

```bash
# Default build (sm_86 = Ampere / RTX A-series)
make

# Override GPU architecture (e.g. Turing sm_75, Volta sm_70)
make ARCH=sm_75

# Custom CUDA path
make CUDA_PATH=/usr/local/cuda-12.0

# Clean binary
make clean

# Build and run in one step
make run
```

---

## Running

### Quick start

```bash
./run.sh
```

Builds the binary if needed and runs the full pipeline with default arguments, including the elbow sweep and all three KNN distance metrics.

### Full pipeline (manual)

```bash
./iris_gpu \
  --data       iris/iris.data \
  --algorithm  all \
  --k          3 \
  --knn-k      5 \
  --components 2 \
  --iterations 300 \
  --runs       5 \
  --elbow \
  --k-max      8 \
  --seed       42 \
  --output     output
```

### Run individual algorithms

```bash
# K-Means with 10 restarts and elbow analysis
./iris_gpu --algorithm kmeans --k 3 --runs 10 --elbow --k-max 6

# KNN with Manhattan distance
./iris_gpu --algorithm knn --knn-k 5 --knn-metric manhattan

# KNN with Cosine distance
./iris_gpu --algorithm knn --knn-k 7 --knn-metric cosine

# PCA with 3 components
./iris_gpu --algorithm pca --components 3

# Gaussian Naive Bayes only
./iris_gpu --algorithm gnb

# Alternate dataset
./iris_gpu --data iris/bezdekIris.data --algorithm all --elbow
```

---

## CLI Reference

| Flag | Type | Default | Description |
|---|---|---|---|
| `--data` | path | `iris/iris.data` | Path to the Iris CSV data file |
| `--algorithm` | string | `all` | `kmeans`, `knn`, `pca`, `gnb`, or `all` |
| `--k` | int | `3` | Number of K-Means clusters |
| `--knn-k` | int | `5` | Number of nearest neighbours |
| `--knn-metric` | string | `euclidean` | `euclidean`, `manhattan`, or `cosine` |
| `--components` | int | `2` | PCA components to retain |
| `--iterations` | int | `300` | Maximum K-Means iterations per run |
| `--runs` | int | `5` | K-Means restarts (best inertia wins) |
| `--elbow` | flag | off | Enable k=1..k-max elbow sweep |
| `--k-max` | int | `8` | Upper bound of elbow sweep |
| `--seed` | uint | `42` | Base RNG seed for K-Means |
| `--output` | path | `output` | Directory for output CSV/log files |
| `--help` | — | — | Print usage and exit |

When `--algorithm all` is used, KNN is automatically executed with **all three distance metrics** (euclidean, manhattan, cosine) in a single run.

---

## Output Files

### Normalisation
| File | Contents |
|---|---|
| `raw_data.csv` | Original 150×4 features + true class labels |
| `normalised_data.csv` | Z-score normalised features + true labels |
| `norm_stats.csv` | Per-feature mean and standard deviation |

### K-Means
| File | Contents |
|---|---|
| `kmeans_labels.csv` | Cluster assignment per sample |
| `kmeans_centroids.csv` | Final centroid coordinates (normalised space) |
| `kmeans_result.csv` | Normalised features + cluster labels (for plotting) |
| `kmeans_summary.csv` | k, n_runs, iterations, inertia, purity, total GPU time |
| `elbow_method.csv` | k, inertia, silhouette score for k=1..k_max |

### Silhouette
| File | Contents |
|---|---|
| `silhouette_scores.csv` | Per-sample s(i) with sample ID and cluster ID |
| `silhouette_summary.csv` | Mean score, per-cluster scores, GPU time |

### KNN (one set of files per metric suffix `_euclidean`, `_manhattan`, `_cosine`)
| File | Contents |
|---|---|
| `knn_predictions_<metric>.csv` | True label, predicted label, correct flag |
| `knn_confusion_<metric>.csv` | 3×3 confusion matrix |
| `knn_metrics_<metric>.csv` | Per-class precision, recall, F1, support |
| `knn_summary_<metric>.csv` | k, distance metric, accuracy, macro F1, GPU time |

### PCA
| File | Contents |
|---|---|
| `pca_projection.csv` | Per-sample PC scores + true label |
| `pca_components.csv` | Eigenvectors (rows), eigenvalues, explained variance |
| `pca_summary.csv` | Per-component stats, cumulative variance, GPU time |

### Gaussian Naive Bayes
| File | Contents |
|---|---|
| `gnb_predictions.csv` | True label, predicted label, correct flag |
| `gnb_confusion.csv` | 3×3 confusion matrix |
| `gnb_metrics.csv` | Per-class precision, recall, F1, support |
| `gnb_model.csv` | Per-class mean, variance, log-prior for each feature |
| `gnb_summary.csv` | Accuracy, macro F1, weighted F1, GPU times |

### Timing & Reports
| File | Contents |
|---|---|
| `timing_results.csv` | GPU wall-clock time (ms) per algorithm stage |
| `summary_report.txt` | Human-readable overall summary |
| `execution_log.txt` | Full stdout from the canonical pipeline run |

---

## Results

Executed on **NVIDIA RTX A6000** (Ampere, sm_86, 47.5 GB VRAM, 84 SMs).

### Z-Score Normalisation

| Feature | Mean | Std Dev |
|---|---|---|
| sepal_length_cm | 5.8433 | 0.8253 |
| sepal_width_cm | 3.0540 | 0.4321 |
| petal_length_cm | 3.7587 | 1.7585 |
| petal_width_cm | 1.1987 | 0.7606 |

### K-Means Clustering (k=3, 5 restarts, seed=42)

| Metric | Value |
|---|---|
| Iterations to converge (best run) | 5 |
| Best inertia | 140.97 |
| Cluster purity | **83.33%** |
| Total GPU time (5 runs) | 0.82 ms |

Iris-setosa is perfectly isolated. Iris-versicolor and Iris-virginica share overlapping feature regions, causing the residual impurity — a well-known property of this dataset.

### Elbow Method (k=1..8, 3 runs each)

| k | Inertia | Silhouette Score |
|---|---|---|
| 1 | 600.00 | — |
| 2 | 223.73 | **0.5802** |
| 3 | 141.15 | 0.4621 |
| 4 | 114.68 | 0.3872 |
| 5 | 91.14 | 0.3460 |
| 6 | 82.14 | 0.3472 |
| 7 | 73.02 | 0.3400 |
| 8 | 69.45 | 0.3448 |

**Optimal k = 2** by silhouette score — reflecting the fact that Iris-setosa is the only linearly separable class. Choosing k=3 is motivated by the known ground truth, not by unsupervised geometry alone.

### Silhouette Coefficient (k=3)

| Metric | Value |
|---|---|
| Mean silhouette score | **0.459** (reasonable structure) |
| Cluster 0 (Versicolor+Virginica mix) | 0.348 |
| Cluster 1 (Setosa) | **0.634** |
| Cluster 2 (Versicolor+Virginica mix) | 0.393 |
| GPU time | 2.55 ms |

### KNN Classification — Distance Metric Comparison (leave-one-out, k=5)

| Metric | Accuracy | Macro F1 | GPU time |
|---|---|---|---|
| **Manhattan** | **95.33%** | **0.953** | 5.40 ms |
| Euclidean | 94.67% | 0.947 | 6.04 ms |
| Cosine | 86.00% | 0.860 | 5.38 ms |

Manhattan outperforms Euclidean on this normalised 4D dataset. Cosine distance, which ignores magnitude and only considers direction, performs worse because the absolute distances between classes carry discriminative information.

**KNN Euclidean confusion matrix:**

|  | Predicted setosa | Predicted versicolor | Predicted virginica |
|---|---|---|---|
| **Actual setosa** | 50 | 0 | 0 |
| **Actual versicolor** | 0 | 46 | 4 |
| **Actual virginica** | 0 | 4 | 46 |

### PCA (2 principal components)

| Component | Eigenvalue | Variance Explained | Cumulative |
|---|---|---|---|
| PC1 | 2.9304 | 72.77% | 72.77% |
| PC2 | 0.9274 | 23.03% | **95.80%** |

**PC1 loading:** `[+0.5224, −0.2634, +0.5813, +0.5656]` — dominated by petal length and petal width, the most discriminative features.
**PC2 loading:** `[−0.3723, −0.9256, −0.0211, −0.0654]` — almost entirely sepal width.

Just two components capture **95.8% of total variance**, confirming the low intrinsic dimensionality of the Iris dataset.

### Gaussian Naive Bayes

| Metric | Value |
|---|---|
| Accuracy | **96.00%** |
| Macro F1 | **0.960** |
| Weighted F1 | **0.960** |
| Training GPU time | < 1 ms (Thrust) |
| Prediction GPU time | 0.024 ms (kernel) |

GNB achieves the highest accuracy of all classifiers in this pipeline. Both Versicolor and Virginica are correctly classified at 94%, with only 3 confusion errors each.

### Algorithm Accuracy / Performance Summary

| Algorithm | Accuracy / Score | GPU Time |
|---|---|---|
| K-Means (k=3, purity) | 83.33% | 0.82 ms (5 runs) |
| Silhouette (k=3) | 0.459 | 2.55 ms |
| KNN Euclidean (k=5) | 94.67% | 6.04 ms |
| KNN Manhattan (k=5) | 95.33% | 5.40 ms |
| KNN Cosine (k=5) | 86.00% | 5.38 ms |
| PCA (2 components) | 95.80% var. | 9.77 ms |
| Gaussian Naive Bayes | **96.00%** | 0.024 ms |

---

## Lessons Learned

1. **cuBLAS column-major convention** requires careful reasoning when working with row-major C++ arrays. Treating a row-major matrix A (n×f) as a column-major Aᵀ (f×n) allows the standard GEMM to express XᵀX naturally without explicit transposition of large matrices.

2. **Thrust device functors must be at file scope.** CUB — the backend Thrust uses for reductions — instantiates `__global__` templates that require types visible at translation-unit scope. Defining a functor struct inside a `__host__` function triggers a compiler error (`"A type defined inside a __host__ function cannot be used in the template argument"`). The fix is always to place functors at namespace scope.

3. **cuSolver `Ssyevd` returns eigenvalues in ascending order.** The dominant principal components correspond to the *largest* eigenvalues, so both the eigenvalue vector and eigenvector matrix columns must be reversed after the solver returns.

4. **K-Means++ seeding** dramatically reduces the number of iterations required for convergence and avoids the degenerate-cluster problem, at negligible cost for 150-sample datasets.

5. **Multi-run restarts matter even for small datasets.** A single K-Means run can converge to a local minimum; across 5 restarts the best inertia was consistently 1–5% lower than the worst. For larger datasets, the difference is more pronounced.

6. **Distance metric choice is not trivial.** On the normalised Iris features, Manhattan distance outperforms Euclidean for k-NN. Cosine distance, which discards magnitude, performs significantly worse — showing that the absolute scale of petal dimensions is discriminative information that should not be normalised away at the similarity level.

7. **The silhouette score provides a quantitative way to choose k.** The elbow in the inertia curve at k=2 is ambiguous; the silhouette score makes the preference for k=2 explicit (0.58 vs 0.46 at k=3), and correctly identifies that Iris-setosa is the only well-separated cluster in unsupervised 4D Euclidean space.

8. **Gaussian Naive Bayes achieved the best accuracy (96%)** despite its strong independence assumption. On low-dimensional, well-normalised data with balanced classes, the probabilistic log-likelihood approach is hard to beat — and at 0.024 ms prediction time, it is by far the fastest classifier in the pipeline.

9. **GPU overhead dominates for tiny datasets.** The RTX A6000 spends more time on API initialisation and host-device transfers than on actual computation for 150 samples. These techniques become essential at scale: tens of thousands of features, millions of samples, or real-time stream processing.
