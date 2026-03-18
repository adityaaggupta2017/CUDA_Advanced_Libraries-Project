// Copyright 2024 Iris GPU ML Pipeline
//
// Implementation of data loading and CSV saving utilities.

#include "data_loader.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static int LabelToInt(const std::string& name) {
  if (name == "Iris-setosa")     return 0;
  if (name == "Iris-versicolor") return 1;
  if (name == "Iris-virginica")  return 2;
  // Unknown – return -1; the caller validates.
  return -1;
}

// Trim leading/trailing whitespace and CR characters.
static std::string Trim(const std::string& s) {
  size_t start = s.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) return "";
  size_t end = s.find_last_not_of(" \t\r\n");
  return s.substr(start, end - start + 1);
}

// ---------------------------------------------------------------------------
// LoadIrisData
// ---------------------------------------------------------------------------

bool LoadIrisData(const std::string& path, IrisDataset* dataset) {
  if (!dataset) return false;

  std::ifstream file(path);
  if (!file.is_open()) {
    std::cerr << "[data_loader] Cannot open file: " << path << "\n";
    return false;
  }

  dataset->features.clear();
  dataset->labels.clear();
  dataset->n_samples  = 0;
  dataset->n_features = kNumFeatures;
  dataset->n_classes  = kNumClasses;

  std::string line;
  int line_num = 0;
  while (std::getline(file, line)) {
    ++line_num;
    line = Trim(line);
    if (line.empty()) continue;

    std::stringstream ss(line);
    std::string token;
    std::vector<std::string> tokens;

    while (std::getline(ss, token, ',')) {
      tokens.push_back(Trim(token));
    }

    if (static_cast<int>(tokens.size()) != kNumFeatures + 1) {
      std::cerr << "[data_loader] Skipping malformed line " << line_num
                << " (expected " << kNumFeatures + 1 << " fields, got "
                << tokens.size() << ")\n";
      continue;
    }

    // Parse numeric features.
    bool parse_ok = true;
    for (int f = 0; f < kNumFeatures; ++f) {
      try {
        dataset->features.push_back(std::stof(tokens[f]));
      } catch (const std::exception& e) {
        std::cerr << "[data_loader] Bad float '" << tokens[f]
                  << "' at line " << line_num << "\n";
        parse_ok = false;
        break;
      }
    }
    if (!parse_ok) {
      // Remove already-pushed partial row.
      for (int f = 0; f < kNumFeatures; ++f) {
        if (!dataset->features.empty()) dataset->features.pop_back();
      }
      continue;
    }

    // Parse class label.
    int label = LabelToInt(tokens[kNumFeatures]);
    if (label < 0) {
      std::cerr << "[data_loader] Unknown class '" << tokens[kNumFeatures]
                << "' at line " << line_num << "\n";
      for (int f = 0; f < kNumFeatures; ++f) dataset->features.pop_back();
      continue;
    }
    dataset->labels.push_back(label);
    ++dataset->n_samples;
  }

  file.close();
  std::cout << "[data_loader] Loaded " << dataset->n_samples
            << " samples from " << path << "\n";
  return dataset->n_samples > 0;
}

// ---------------------------------------------------------------------------
// PrintDatasetSummary
// ---------------------------------------------------------------------------

void PrintDatasetSummary(const IrisDataset& dataset) {
  const int n = dataset.n_samples;
  const int f = dataset.n_features;

  std::cout << "\n=== Dataset Summary ===\n";
  std::cout << "  Samples   : " << n << "\n";
  std::cout << "  Features  : " << f << "\n";
  std::cout << "  Classes   : " << dataset.n_classes << "\n";

  // Per-class counts.
  std::map<int, int> class_counts;
  for (int label : dataset.labels) class_counts[label]++;
  std::cout << "  Class distribution:\n";
  for (auto& kv : class_counts) {
    std::cout << "    " << kClassNames[kv.first] << " = " << kv.second << "\n";
  }

  // Per-feature min / max / mean.
  std::cout << "  Feature statistics (min / max / mean):\n";
  for (int j = 0; j < f; ++j) {
    float mn = 1e30f, mx = -1e30f, sum = 0.f;
    for (int i = 0; i < n; ++i) {
      float v = dataset.features[i * f + j];
      mn  = std::min(mn, v);
      mx  = std::max(mx, v);
      sum += v;
    }
    float mean = sum / static_cast<float>(n);
    printf("    %-22s  %.2f / %.2f / %.4f\n",
           kFeatureNames[j], mn, mx, mean);
  }
  std::cout << "=======================\n\n";
}

// ---------------------------------------------------------------------------
// CSV save helpers
// ---------------------------------------------------------------------------

bool SaveFloatCsv(const std::string& path,
                  const std::vector<float>& data,
                  int n_rows, int n_cols,
                  const std::string& header) {
  std::ofstream f(path);
  if (!f.is_open()) {
    std::cerr << "[data_loader] Cannot write: " << path << "\n";
    return false;
  }
  if (!header.empty()) f << header << "\n";
  for (int i = 0; i < n_rows; ++i) {
    for (int j = 0; j < n_cols; ++j) {
      if (j) f << ",";
      f << data[i * n_cols + j];
    }
    f << "\n";
  }
  return true;
}

bool SaveIntCsv(const std::string& path,
                const std::vector<int>& data,
                const std::string& header) {
  std::ofstream f(path);
  if (!f.is_open()) {
    std::cerr << "[data_loader] Cannot write: " << path << "\n";
    return false;
  }
  if (!header.empty()) f << header << "\n";
  for (int v : data) f << v << "\n";
  return true;
}

bool SaveLabeledCsv(const std::string& path,
                    const std::vector<float>& data, int n_rows, int n_cols,
                    const std::vector<int>& labels,
                    const std::string& header) {
  std::ofstream f(path);
  if (!f.is_open()) {
    std::cerr << "[data_loader] Cannot write: " << path << "\n";
    return false;
  }
  if (!header.empty()) f << header << "\n";
  for (int i = 0; i < n_rows; ++i) {
    for (int j = 0; j < n_cols; ++j) {
      f << data[i * n_cols + j] << ",";
    }
    f << labels[i] << "\n";
  }
  return true;
}
