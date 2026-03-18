// Copyright 2024 Iris GPU ML Pipeline
//
// Data loading utilities for the UCI Iris dataset (CSV format).
// Provides structures and functions to parse iris.data and write
// result CSV files to an output directory.

#ifndef SRC_DATA_LOADER_H_
#define SRC_DATA_LOADER_H_

#include <string>
#include <vector>
#include "common.h"

// ---------------------------------------------------------------------------
// IrisDataset  – host-side storage for the loaded Iris data
// ---------------------------------------------------------------------------

struct IrisDataset {
  // Row-major feature matrix: features[i * n_features + j]
  std::vector<float> features;
  // Integer class labels: 0=setosa, 1=versicolor, 2=virginica
  std::vector<int>   labels;
  int n_samples;
  int n_features;
  int n_classes;

  IrisDataset() : n_samples(0), n_features(kNumFeatures),
                  n_classes(kNumClasses) {}
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// LoadIrisData - parses a UCI iris.data CSV file.
// Returns true on success; populates *dataset.
bool LoadIrisData(const std::string& path, IrisDataset* dataset);

// PrintDatasetSummary - prints basic statistics to stdout.
void PrintDatasetSummary(const IrisDataset& dataset);

// SaveFloatCsv - writes an (n_rows x n_cols) row-major float matrix to CSV.
// header: comma-separated column names (may be empty to skip header line).
bool SaveFloatCsv(const std::string& path,
                  const std::vector<float>& data,
                  int n_rows, int n_cols,
                  const std::string& header = "");

// SaveIntCsv - writes a single int vector column to CSV.
bool SaveIntCsv(const std::string& path,
                const std::vector<int>& data,
                const std::string& header = "");

// SaveLabeledCsv - writes float data + int label column to CSV.
bool SaveLabeledCsv(const std::string& path,
                    const std::vector<float>& data, int n_rows, int n_cols,
                    const std::vector<int>& labels,
                    const std::string& header = "");

#endif  // SRC_DATA_LOADER_H_
