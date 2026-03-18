# GPU-Accelerated Iris ML Pipeline

**CUDA Advanced Libraries – Capstone Project**

A high-performance machine learning pipeline implemented entirely with CUDA advanced libraries, applied to the classic UCI Iris dataset. Three fundamental ML algorithms are accelerated on the GPU: **K-Means clustering**, **K-Nearest Neighbours (KNN) classification**, and **Principal Component Analysis (PCA)**.

---

## Table of Contents

1. [Project Description](#project-description)
2. [GPU Libraries Used](#gpu-libraries-used)
3. [Algorithms](#algorithms)
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

This project builds a complete GPU-accelerated ML analysis pipeline:

1. **Data loading & parsing** – CSV parsing in C++.
2. **Z-score normalisation** – Per-feature mean/stddev via Thrust reductions; in-place normalisation via a custom CUDA kernel.
3. **K-Means clustering** – Custom CUDA kernels for distance computation, cluster assignment, and centroid accumulation (via atomic operations); convergence detection on host.
4. **KNN classification (leave-one-out)** – Full pairwise distance matrix computed with a single cuBLAS SGEMM call using the identity ‖a−b‖² = ‖a‖² + ‖b‖² − 2aᵀb; distances sorted per query point with Thrust; majority vote via a custom kernel.
5. **PCA** – Covariance matrix computed with cuBLAS SGEMM; eigendecomposition with cuSolver `Ssyevd`; data projected onto top components with another SGEMM.

All GPU timings are measured with CUDA events. Results are saved as CSV files for reproducibility and downstream visualisation.

---

## GPU Libraries Used

| Library | Purpose |
|---|---|
| **CUDA Runtime** | Device/memory management, custom kernels, event timing |
| **Thrust** | Parallel reductions (mean, variance) and sort-by-key for KNN |
| **cuBLAS** | SGEMM for KNN distance matrix and PCA covariance/projection |
| **cuSolver** | `Ssyevd` – dense symmetric eigenvalue decomposition for PCA |

---

## Algorithms

### Z-Score Normalisation (Thrust + CUDA kernel)

For each feature column *j*:

```
μ_j    = (1/N) Σ X[i,j]
σ_j    = sqrt((1/N) Σ (X[i,j] − μ_j)²)
X[i,j] = (X[i,j] − μ_j) / σ_j
```

The mean and variance sums are computed with Thrust `transform_reduce`; a custom `ApplyZscoreKernel` applies the transformation in parallel.

### K-Means (custom CUDA kernels)

- Initialisation: **k-means++** seeding on the host for good starting centroids.
- **`AssignLabelsKernel`**: each GPU thread handles one sample, computes Euclidean distance to every centroid, writes the nearest centroid index.
- **`AccumulateCentroidsKernel`**: atomic addition to accumulate coordinate sums and counts per cluster.
- **`UpdateCentroidsKernel`**: parallel division to produce new centroids.
- Convergence: label arrays compared host-side each iteration.

### KNN Classification (cuBLAS + Thrust)

Uses the decomposition:

```
‖X_i − X_j‖² = ‖X_i‖² + ‖X_j‖² − 2 Xᵢᵀ Xⱼ
```

- Squared norms computed with `SquaredNormKernel`.
- Cross-term matrix (−2 · X · Xᵀ) from one `cublasSgemm` call.
- Norms added via `AddNormsKernel`.
- Self-distances set to `FLT_MAX` for leave-one-out (`SetDiagInfKernel`).
- Each test row sorted by Thrust `sort_by_key`.
- `MajorityVoteKernel` determines the predicted class.

### PCA (cuBLAS + cuSolver)

1. Column means subtracted (`SubtractMeanKernel`).
2. Covariance: `C = (1/(n−1)) · Xᵀ · X` via `cublasSgemm`.
3. Eigendecomposition: `cusolverDnSsyevd` on the symmetric covariance matrix.
4. Eigenvectors sorted by descending eigenvalue.
5. Projection: `Z = X · V_top` via `cublasSgemm`.

---

## Directory Structure

```
CUDA_Advanced_Libraries-Project/
├── iris/                   # UCI Iris dataset (input data)
│   ├── iris.data
│   ├── bezdekIris.data
│   └── iris.names
├── src/                    # All CUDA/C++ source files
│   ├── main.cu             # CLI entry point and pipeline orchestration
│   ├── common.h            # Shared error macros and dataset constants
│   ├── data_loader.h/.cu   # CSV parsing and result file writing
│   ├── normalizer.h/.cu    # GPU Z-score normalisation (Thrust)
│   ├── kmeans.h/.cu        # GPU K-Means (custom kernels)
│   ├── knn.h/.cu           # GPU KNN (cuBLAS + Thrust)
│   └── pca.h/.cu           # GPU PCA (cuBLAS + cuSolver)
├── output/                 # Generated CSV artifacts and logs
│   ├── execution_log.txt
│   ├── raw_data.csv
│   ├── normalised_data.csv
│   ├── norm_stats.csv
│   ├── kmeans_labels.csv
│   ├── kmeans_centroids.csv
│   ├── kmeans_result.csv
│   ├── kmeans_summary.csv
│   ├── knn_predictions.csv
│   ├── knn_summary.csv
│   ├── pca_projection.csv
│   ├── pca_components.csv
│   ├── pca_summary.csv
│   ├── timing_results.csv
│   └── summary_report.txt
├── Makefile                # Build system
├── run.sh                  # One-shot build+run script
└── README.md               # This file
```

---

## Requirements

| Requirement | Version tested |
|---|---|
| NVIDIA GPU | RTX A6000 (sm_86); any sm_60+ |
| CUDA Toolkit | 12.0 (`/usr/local/cuda`) |
| nvcc | 12.0.76 |
| g++ | ≥ 7 (C++14) |
| cuBLAS | included with CUDA Toolkit |
| cuSolver | included with CUDA Toolkit |
| Thrust | included with CUDA Toolkit |

**Installation of CUDA Toolkit** (Ubuntu/Debian):
```bash
# Download from https://developer.nvidia.com/cuda-downloads
# or via package manager:
sudo apt-get install -y cuda-toolkit-12-0
```

No additional Python, pip, or third-party libraries are required.

---

## Building

```bash
# Default build (sm_86 = Ampere)
make

# Override for a different GPU architecture (e.g. Turing sm_75)
make ARCH=sm_75

# Specify a custom CUDA installation path
make CUDA_PATH=/usr/local/cuda-12.0

# Clean build artifacts
make clean
```

---

## Running

### Quick start (one command)

```bash
./run.sh
```

This script builds the binary (if needed) and runs the full pipeline with default arguments.

### Manual execution

```bash
./iris_gpu --data iris/iris.data \
           --algorithm all \
           --k 3 \
           --knn-k 5 \
           --components 2 \
           --iterations 300 \
           --seed 42 \
           --output output
```

### Run individual algorithms

```bash
# K-Means only
./iris_gpu --data iris/iris.data --algorithm kmeans --k 3

# KNN only (k=7 neighbours)
./iris_gpu --data iris/iris.data --algorithm knn --knn-k 7

# PCA only (retain 3 components)
./iris_gpu --data iris/iris.data --algorithm pca --components 3

# Alternate dataset
./iris_gpu --data iris/bezdekIris.data --algorithm all
```

---

## CLI Reference

| Flag | Type | Default | Description |
|---|---|---|---|
| `--data` | path | `iris/iris.data` | Path to the Iris CSV data file |
| `--algorithm` | string | `all` | `kmeans`, `knn`, `pca`, or `all` |
| `--k` | int | `3` | Number of K-Means clusters |
| `--knn-k` | int | `5` | Number of nearest neighbours for KNN |
| `--components` | int | `2` | PCA components to retain |
| `--iterations` | int | `300` | Maximum K-Means iterations |
| `--seed` | uint | `42` | RNG seed for K-Means initialisation |
| `--output` | path | `output` | Directory for output CSV/log files |
| `--help` | — | — | Print usage |

---

## Output Files

| File | Contents |
|---|---|
| `raw_data.csv` | Original 150×4 features + true labels |
| `normalised_data.csv` | Z-score normalised features + true labels |
| `norm_stats.csv` | Per-feature mean and standard deviation |
| `kmeans_labels.csv` | Cluster assignment per sample |
| `kmeans_centroids.csv` | Final centroid coordinates (normalised space) |
| `kmeans_result.csv` | Normalised features with cluster labels (for plotting) |
| `kmeans_summary.csv` | k, iterations, inertia, purity, GPU time |
| `knn_predictions.csv` | True label, predicted label, correct flag per sample |
| `knn_summary.csv` | k, accuracy, GPU time |
| `pca_projection.csv` | 2D/nD projected coordinates + true labels |
| `pca_components.csv` | Eigenvectors, eigenvalues, explained variance |
| `pca_summary.csv` | Per-component statistics + GPU time |
| `timing_results.csv` | GPU wall-clock time per algorithm |
| `summary_report.txt` | Human-readable overall summary |
| `execution_log.txt` | Full stdout from the canonical run |

---

## Results

Executed on **NVIDIA RTX A6000** (Ampere, sm_86, 47.5 GB VRAM).

### Z-Score Normalisation

| Feature | Mean | Std Dev |
|---|---|---|
| sepal_length_cm | 5.8433 | 0.8253 |
| sepal_width_cm | 3.0540 | 0.4321 |
| petal_length_cm | 3.7587 | 1.7585 |
| petal_width_cm | 1.1987 | 0.7606 |

### K-Means Clustering (k=3, seed=42)

| Metric | Value |
|---|---|
| Iterations to converge | 9 |
| Inertia | 141.15 |
| Cluster purity | **81.33%** |
| GPU time | 0.375 ms |

Iris-setosa is linearly separable, so its cluster is recovered perfectly. Iris-versicolor and Iris-virginica overlap in feature space, which explains the ~19% impurity – consistent with the known structure of this dataset.

### KNN Classification (leave-one-out, k=5)

| Class | Correct / Total | Accuracy |
|---|---|---|
| Iris-setosa | 50 / 50 | 100.0% |
| Iris-versicolor | 46 / 50 | 92.0% |
| Iris-virginica | 46 / 50 | 92.0% |
| **Overall** | **142 / 150** | **94.67%** |
| GPU time | — | 7.77 ms |

### PCA (2 principal components)

| Component | Eigenvalue | Variance Explained | Cumulative |
|---|---|---|---|
| PC1 | 2.9304 | 72.77% | 72.77% |
| PC2 | 0.9274 | 23.03% | 95.80% |

PC1 loading: `[+0.5224, −0.2634, +0.5813, +0.5656]` – heavily weighted on petal dimensions, the most discriminative features.
PC2 loading: `[−0.3723, −0.9256, −0.0211, −0.0654]` – almost entirely sepal width.

Two components capture **95.8%** of total variance, demonstrating the low intrinsic dimensionality of this dataset.

---

## Lessons Learned

1. **cuBLAS column-major convention** requires careful thought when working with row-major C++ arrays. The "transpose trick" (treating a row-major A as col-major Aᵀ) allows expressing row-major GEMM naturally.

2. **Thrust device functors must be defined at file scope**, not inside host functions, because CUB (the backend) instantiates `__global__` templates on them. Placing a struct inside a function violates this constraint and produces a compiler error.

3. **cuSolver's `Ssyevd`** returns eigenvalues in *ascending* order, which must be reversed to obtain the dominant PCA components first.

4. **K-Means++ seeding** greatly reduces the number of iterations needed compared to random initialisation, and is cheap for a 150-sample dataset.

5. **GPU overhead dominates** for tiny datasets – the RTX A6000 spends more time in CUDA API initialisation than in computation. GPU acceleration becomes essential for larger datasets (ImageNet, genomic sequences, etc.).
