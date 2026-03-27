"""
Generate proof-of-execution artifacts (images + logs) for the
GPU-Accelerated Iris ML Pipeline CUDA capstone project.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings("ignore")

OUT = "/home/sahil22430/diffusion_experiments/Coursera_Projects/CUDA_Advanced_Libraries-Project/output"
IMG = os.path.join(OUT, "artifacts")
os.makedirs(IMG, exist_ok=True)

COLORS = ["#e41a1c", "#377eb8", "#4daf4a"]
CLASS_NAMES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
raw        = pd.read_csv(f"{OUT}/raw_data.csv")
norm       = pd.read_csv(f"{OUT}/normalised_data.csv")
pca        = pd.read_csv(f"{OUT}/pca_projection.csv")
km_labels  = pd.read_csv(f"{OUT}/kmeans_labels.csv")
elbow      = pd.read_csv(f"{OUT}/elbow_method.csv")
timing     = pd.read_csv(f"{OUT}/timing_results.csv")
sil_scores = pd.read_csv(f"{OUT}/silhouette_scores.csv")
knn_eu_m   = pd.read_csv(f"{OUT}/knn_metrics_euclidean.csv", on_bad_lines="skip")
knn_ma_m   = pd.read_csv(f"{OUT}/knn_metrics_manhattan.csv", on_bad_lines="skip")
knn_co_m   = pd.read_csv(f"{OUT}/knn_metrics_cosine.csv", on_bad_lines="skip")
gnb_m      = pd.read_csv(f"{OUT}/gnb_metrics.csv", on_bad_lines="skip")

# confusion matrices
def load_cm(path):
    df = pd.read_csv(path, index_col=0)
    return df.values.astype(int)

cm_eu = load_cm(f"{OUT}/knn_confusion_euclidean.csv")
cm_ma = load_cm(f"{OUT}/knn_confusion_manhattan.csv")
cm_co = load_cm(f"{OUT}/knn_confusion_cosine.csv")
cm_gn = load_cm(f"{OUT}/gnb_confusion.csv")

labels_true   = raw["label"].values
km_cluster    = km_labels["cluster_id"].values


# ─────────────────────────────────────────────────────────────────────────────
# Helper: confusion-matrix heatmap
# ─────────────────────────────────────────────────────────────────────────────
def plot_cm(ax, cm, title):
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted label", fontsize=9)
    ax.set_ylabel("True label", fontsize=9)
    short = ["setosa", "versicolor", "virginica"]
    ax.set_xticks(range(3)); ax.set_xticklabels(short, fontsize=8)
    ax.set_yticks(range(3)); ax.set_yticklabels(short, fontsize=8)
    thresh = cm.max() / 2.0
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=11)
    acc = cm.diagonal().sum() / cm.sum() * 100
    ax.set_xlabel(f"Predicted label  (Accuracy {acc:.1f}%)", fontsize=9)
    return im


# ═════════════════════════════════════════════════════════════════════════════
# FIG 1 – Before vs After Normalisation (feature distributions)
# ═════════════════════════════════════════════════════════════════════════════
feat_raw  = ["sepal_length_cm","sepal_width_cm","petal_length_cm","petal_width_cm"]
feat_norm = ["norm_sepal_length","norm_sepal_width","norm_petal_length","norm_petal_width"]
feat_labels = ["Sepal Length","Sepal Width","Petal Length","Petal Width"]

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
fig.suptitle("Feature Distributions — Before vs After GPU Z-score Normalisation",
             fontsize=13, fontweight="bold")

for col_i, (fr, fn, fl) in enumerate(zip(feat_raw, feat_norm, feat_labels)):
    ax = axes[0, col_i]
    for cls in range(3):
        mask = labels_true == cls
        ax.hist(raw.loc[mask, fr], bins=15, alpha=0.6, color=COLORS[cls],
                label=CLASS_NAMES[cls])
    ax.set_title(f"BEFORE\n{fl}", fontsize=9)
    ax.set_ylabel("Count" if col_i == 0 else "")
    if col_i == 0:
        ax.legend(fontsize=7)

    ax2 = axes[1, col_i]
    for cls in range(3):
        mask = labels_true == cls
        ax2.hist(norm.loc[mask, fn], bins=15, alpha=0.6, color=COLORS[cls])
    ax2.set_title(f"AFTER (normalised)\n{fl}", fontsize=9)
    ax2.set_ylabel("Count" if col_i == 0 else "")

plt.tight_layout()
path = f"{IMG}/01_before_after_normalisation.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 2 – PCA Projection: True Labels vs K-Means Clusters
# ═════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("GPU PCA Projection (cuBLAS + cuSolver) — 2 Components, 95.80% Variance",
             fontsize=12, fontweight="bold")

# True labels
for cls in range(3):
    mask = pca["true_label"] == cls
    ax1.scatter(pca.loc[mask, "PC1"], pca.loc[mask, "PC2"],
                c=COLORS[cls], label=CLASS_NAMES[cls], s=50, edgecolors="k", linewidths=0.4)
ax1.set_title("True Class Labels", fontsize=11)
ax1.set_xlabel("PC1 (72.77% variance)")
ax1.set_ylabel("PC2 (23.03% variance)")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# K-Means clusters
km_colors = ["#ff7f00", "#984ea3", "#a65628"]
for c in range(3):
    mask = km_cluster == c
    ax2.scatter(pca.loc[mask, "PC1"], pca.loc[mask, "PC2"],
                c=km_colors[c], label=f"Cluster {c}", s=50,
                edgecolors="k", linewidths=0.4, marker="^")
ax2.set_title("K-Means Clusters (GPU, k=3, 83.3% purity)", fontsize=11)
ax2.set_xlabel("PC1 (72.77% variance)")
ax2.set_ylabel("PC2 (23.03% variance)")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

plt.tight_layout()
path = f"{IMG}/02_pca_projection_true_vs_kmeans.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 3 – Elbow Method + Silhouette Sweep
# ═════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("GPU Elbow Method & Silhouette Sweep (k=1..8)", fontsize=12, fontweight="bold")

ax1.plot(elbow["k"], elbow["inertia"], "o-", color="#377eb8", lw=2)
ax1.axvline(x=3, color="red", linestyle="--", alpha=0.7, label="k=3 (chosen)")
ax1.set_xlabel("k (number of clusters)")
ax1.set_ylabel("Inertia (within-cluster SSE)")
ax1.set_title("Elbow Curve")
ax1.legend(); ax1.grid(alpha=0.3)

ax2.bar(elbow["k"], elbow["silhouette_score"], color="#4daf4a", edgecolor="k", alpha=0.8)
ax2.axvline(x=2, color="red", linestyle="--", alpha=0.7, label="Optimal k=2")
ax2.set_xlabel("k (number of clusters)")
ax2.set_ylabel("Mean Silhouette Score")
ax2.set_title("Silhouette vs k")
ax2.legend(); ax2.grid(alpha=0.3, axis="y")

plt.tight_layout()
path = f"{IMG}/03_elbow_silhouette_sweep.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 4 – Silhouette Scores per Sample (k=3)
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 4))
sil_sorted = sil_scores.sort_values(["cluster_id", "silhouette_score"]).reset_index(drop=True)
colors_sil = [COLORS[int(c)] for c in sil_sorted["cluster_id"]]
ax.bar(range(len(sil_sorted)), sil_sorted["silhouette_score"], color=colors_sil, width=1.0, edgecolor="none")
ax.axhline(y=sil_scores["silhouette_score"].mean(), color="black", lw=1.5,
           linestyle="--", label=f"Mean = {sil_scores['silhouette_score'].mean():.3f}")
ax.set_xlabel("Samples (sorted by cluster then score)")
ax.set_ylabel("Silhouette Score")
ax.set_title("GPU Silhouette Scores per Sample — k=3 (cuBLAS + custom kernels)", fontsize=11, fontweight="bold")
from matplotlib.patches import Patch
legend_els = [Patch(color=COLORS[i], label=f"Cluster {i}") for i in range(3)]
legend_els.append(plt.Line2D([0],[0], color="black", linestyle="--", label=f"Mean=0.4590"))
ax.legend(handles=legend_els, fontsize=9)
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
path = f"{IMG}/04_silhouette_scores_per_sample.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 5 – Confusion Matrices (KNN ×3 + GNB)
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
fig.suptitle("GPU Confusion Matrices — KNN (3 metrics) & Gaussian Naive Bayes",
             fontsize=12, fontweight="bold")

plot_cm(axes[0], cm_eu, "KNN Euclidean\n(94.67%)")
plot_cm(axes[1], cm_ma, "KNN Manhattan\n(95.33%)")
plot_cm(axes[2], cm_co, "KNN Cosine\n(86.00%)")
plot_cm(axes[3], cm_gn, "Gaussian NB\n(96.00%)")

plt.tight_layout()
path = f"{IMG}/05_confusion_matrices_all.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 6 – F1 Score Comparison: KNN metrics × 3 + GNB
# ═════════════════════════════════════════════════════════════════════════════
short_cls = ["setosa", "versicolor", "virginica"]

def get_f1(df):
    rows = df[df["class"].isin(CLASS_NAMES)].copy()
    rows["class"] = rows["class"].str.replace("Iris-", "")
    return rows.set_index("class")["f1"].reindex(short_cls).values

f1_eu = get_f1(knn_eu_m)
f1_ma = get_f1(knn_ma_m)
f1_co = get_f1(knn_co_m)
f1_gn = get_f1(gnb_m)

x = np.arange(3)
w = 0.2
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - 1.5*w, f1_eu, w, label="KNN Euclidean", color="#377eb8", edgecolor="k")
ax.bar(x - 0.5*w, f1_ma, w, label="KNN Manhattan", color="#4daf4a", edgecolor="k")
ax.bar(x + 0.5*w, f1_co, w, label="KNN Cosine",    color="#ff7f00", edgecolor="k")
ax.bar(x + 1.5*w, f1_gn, w, label="GNB",           color="#e41a1c", edgecolor="k")
ax.set_xticks(x); ax.set_xticklabels(short_cls, fontsize=11)
ax.set_ylim(0, 1.08)
ax.set_ylabel("F1 Score")
ax.set_title("Per-Class F1 Score: GPU KNN (3 metrics) vs Gaussian Naive Bayes",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
for bars in ax.containers:
    ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
plt.tight_layout()
path = f"{IMG}/06_f1_score_comparison.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 7 – GPU Timing Results
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
timing_plot = timing.copy()
timing_plot = timing_plot[timing_plot["algorithm"] != "gnb_predict"]  # tiny, hard to see
timing_plot = timing_plot.sort_values("gpu_time_ms", ascending=True)
bar_colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(timing_plot)))
bars = ax.barh(timing_plot["algorithm"], timing_plot["gpu_time_ms"],
               color=bar_colors, edgecolor="k")
ax.bar_label(bars, fmt="%.3f ms", padding=3, fontsize=9)
ax.set_xlabel("GPU Time (ms)")
ax.set_title("GPU Execution Time per Algorithm\n(NVIDIA RTX A6000, 84 SMs, 47.5 GB)",
             fontsize=11, fontweight="bold")
ax.grid(alpha=0.3, axis="x")
plt.tight_layout()
path = f"{IMG}/07_gpu_timing_results.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 8 – PCA Variance Explained (scree plot)
# ═════════════════════════════════════════════════════════════════════════════
pca_sum = pd.read_csv(f"{OUT}/pca_summary.csv")
components = ["PC1","PC2"]
var_pct = [float(pca_sum.loc[pca_sum["metric"]=="PC1_explained_var","value"].iloc[0])*100,
           float(pca_sum.loc[pca_sum["metric"]=="PC2_explained_var","value"].iloc[0])*100]
cumvar  = [var_pct[0], var_pct[0]+var_pct[1]]

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.bar(components, var_pct, color=["#377eb8","#4daf4a"], edgecolor="k", alpha=0.85, label="Explained")
ax2 = ax.twinx()
ax2.plot(components, cumvar, "o--", color="red", lw=2, label="Cumulative")
ax2.set_ylabel("Cumulative Variance (%)", color="red")
ax2.tick_params(axis="y", labelcolor="red")
ax2.set_ylim(0, 110)
ax.set_ylabel("Individual Variance (%)")
ax.set_ylim(0, 100)
ax.set_title("GPU PCA — Variance Explained\n(cuBLAS + cuSolver)", fontsize=11, fontweight="bold")
for i, v in enumerate(var_pct):
    ax.text(i, v + 1.5, f"{v:.2f}%", ha="center", fontweight="bold")
ax2.text(1, cumvar[1] + 2, f"{cumvar[1]:.2f}%", ha="center", color="red", fontweight="bold")
plt.tight_layout()
path = f"{IMG}/08_pca_variance_explained.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 9 – Raw Scatter Matrix (before) — petal/sepal pairs
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Raw Data Scatter Plots — Before GPU Normalisation", fontsize=12, fontweight="bold")
pairs = [("sepal_length_cm","sepal_width_cm"),
         ("sepal_length_cm","petal_length_cm"),
         ("sepal_length_cm","petal_width_cm"),
         ("sepal_width_cm", "petal_length_cm"),
         ("sepal_width_cm", "petal_width_cm"),
         ("petal_length_cm","petal_width_cm")]
for ax, (fx, fy) in zip(axes.flat, pairs):
    for cls in range(3):
        mask = labels_true == cls
        ax.scatter(raw.loc[mask, fx], raw.loc[mask, fy],
                   c=COLORS[cls], label=CLASS_NAMES[cls], s=30, alpha=0.7, edgecolors="k", lw=0.3)
    ax.set_xlabel(fx.replace("_cm","").replace("_"," "), fontsize=8)
    ax.set_ylabel(fy.replace("_cm","").replace("_"," "), fontsize=8)
    ax.grid(alpha=0.3)
axes[0,0].legend(fontsize=7)
plt.tight_layout()
path = f"{IMG}/09_before_raw_scatter_matrix.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 10 – Normalised Scatter Matrix (after GPU normalisation)
# ═════════════════════════════════════════════════════════════════════════════
norm_feats = ["norm_sepal_length","norm_sepal_width","norm_petal_length","norm_petal_width"]
norm_labels_plot = ["sepal length (z)","sepal width (z)","petal length (z)","petal width (z)"]
norm_pairs = [(norm_feats[0],norm_feats[1]),(norm_feats[0],norm_feats[2]),
              (norm_feats[0],norm_feats[3]),(norm_feats[1],norm_feats[2]),
              (norm_feats[1],norm_feats[3]),(norm_feats[2],norm_feats[3])]
norm_pair_labels = [(norm_labels_plot[0],norm_labels_plot[1]),
                    (norm_labels_plot[0],norm_labels_plot[2]),
                    (norm_labels_plot[0],norm_labels_plot[3]),
                    (norm_labels_plot[1],norm_labels_plot[2]),
                    (norm_labels_plot[1],norm_labels_plot[3]),
                    (norm_labels_plot[2],norm_labels_plot[3])]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Normalised Data Scatter Plots — After GPU Z-score Normalisation (Thrust)",
             fontsize=12, fontweight="bold")
for ax, (fx, fy), (lx, ly) in zip(axes.flat, norm_pairs, norm_pair_labels):
    for cls in range(3):
        mask = labels_true == cls
        ax.scatter(norm.loc[mask, fx], norm.loc[mask, fy],
                   c=COLORS[cls], label=CLASS_NAMES[cls], s=30, alpha=0.7, edgecolors="k", lw=0.3)
    ax.set_xlabel(lx, fontsize=8); ax.set_ylabel(ly, fontsize=8)
    ax.grid(alpha=0.3)
axes[0,0].legend(fontsize=7)
plt.tight_layout()
path = f"{IMG}/10_after_normalised_scatter_matrix.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 11 – Summary dashboard
# ═════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 8))
fig.patch.set_facecolor("#f8f8f8")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

# top-left: PCA
ax_pca = fig.add_subplot(gs[0, 0])
for cls in range(3):
    mask = pca["true_label"] == cls
    ax_pca.scatter(pca.loc[mask, "PC1"], pca.loc[mask, "PC2"],
                   c=COLORS[cls], label=CLASS_NAMES[cls].replace("Iris-",""), s=25, alpha=0.8)
ax_pca.set_title("GPU PCA (95.8% var)", fontsize=9, fontweight="bold")
ax_pca.set_xlabel("PC1"); ax_pca.set_ylabel("PC2")
ax_pca.legend(fontsize=6); ax_pca.grid(alpha=0.3)

# top-mid: Elbow
ax_el = fig.add_subplot(gs[0, 1])
ax_el.plot(elbow["k"], elbow["inertia"], "o-", color="#377eb8", lw=2)
ax_el.axvline(x=3, color="red", ls="--", alpha=0.7)
ax_el.set_title("Elbow Method (K-Means)", fontsize=9, fontweight="bold")
ax_el.set_xlabel("k"); ax_el.set_ylabel("Inertia"); ax_el.grid(alpha=0.3)

# top-right: Silhouette per sample
ax_si = fig.add_subplot(gs[0, 2])
sil_s = sil_scores.sort_values(["cluster_id","silhouette_score"]).reset_index(drop=True)
ax_si.bar(range(len(sil_s)), sil_s["silhouette_score"],
          color=[COLORS[int(c)] for c in sil_s["cluster_id"]], width=1)
ax_si.axhline(sil_scores["silhouette_score"].mean(), color="k", lw=1.2, ls="--")
ax_si.set_title("Silhouette Scores (k=3)", fontsize=9, fontweight="bold")
ax_si.set_xlabel("Sample"); ax_si.set_ylabel("Score"); ax_si.grid(alpha=0.3, axis="y")

# bottom-left: confusion GNB
ax_cm = fig.add_subplot(gs[1, 0])
plot_cm(ax_cm, cm_gn, "GNB Confusion (96%)")

# bottom-mid: F1 bars
ax_f1 = fig.add_subplot(gs[1, 1])
x = np.arange(3); w = 0.2
ax_f1.bar(x-1.5*w, f1_eu, w, label="KNN-Eu", color="#377eb8", edgecolor="k")
ax_f1.bar(x-0.5*w, f1_ma, w, label="KNN-Ma", color="#4daf4a", edgecolor="k")
ax_f1.bar(x+0.5*w, f1_co, w, label="KNN-Co", color="#ff7f00", edgecolor="k")
ax_f1.bar(x+1.5*w, f1_gn, w, label="GNB",    color="#e41a1c", edgecolor="k")
ax_f1.set_xticks(x); ax_f1.set_xticklabels(short_cls, fontsize=7)
ax_f1.set_ylim(0,1.1); ax_f1.set_ylabel("F1"); ax_f1.legend(fontsize=6)
ax_f1.set_title("Per-Class F1 Score", fontsize=9, fontweight="bold")
ax_f1.grid(alpha=0.3, axis="y")

# bottom-right: timing
ax_t = fig.add_subplot(gs[1, 2])
t = timing.sort_values("gpu_time_ms")
ax_t.barh(t["algorithm"], t["gpu_time_ms"],
          color=plt.cm.plasma(np.linspace(0.1, 0.8, len(t))), edgecolor="k")
ax_t.set_xlabel("ms"); ax_t.set_title("GPU Timing", fontsize=9, fontweight="bold")
ax_t.grid(alpha=0.3, axis="x")

fig.suptitle("GPU-Accelerated Iris ML Pipeline — Execution Summary Dashboard\n"
             "NVIDIA RTX A6000 | CUDA cuBLAS + cuSolver + Thrust",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
path = f"{IMG}/11_execution_summary_dashboard.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")


print("\nAll artifacts saved to:", IMG)
print("Files:")
for f in sorted(os.listdir(IMG)):
    size = os.path.getsize(os.path.join(IMG, f))
    print(f"  {f}  ({size//1024} KB)")
