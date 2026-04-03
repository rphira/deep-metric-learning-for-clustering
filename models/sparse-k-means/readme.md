# Nanopore Protein Analysis Pipeline

## Overview
This notebook implements a comprehensive pipeline for analyzing nanopore protein blockade events using feature extraction, dimensionality reduction, and sparse clustering techniques. The primary goal is to distinguish between different protein types (histones, PET, MDR, Aplysia peptides, and peptide ladders) based on their electrical signal characteristics.

## Dataset
The analysis processes multiple protein datasets:
- **Histones**: Histone protein blockade events
- **PET**: PET-TE protein data
- **MDR & Aplysia**: MDR1 and Aplysia peptide data
- **Peptide Ladders**: Six different peptide ladder sequences (L1-L6)

## Pipeline Workflow

### 1. Data Loading & Preprocessing
- Loads metadata from CSV files containing blockade event information
- Extracts raw current traces from ABF files using `pyabf`
- Applies filtering based on:
  - Event ID (excludes baseline events)
  - Current blockade ratio (I/Io)
  - Event length (dwell time)
- Removes outliers using percentile-based filtering (5th-95th percentile)
- Handles outlier removal per protein label using Isolation Forest (30% contamination)

### 2. Feature Extraction
Two complementary feature extraction approaches:

#### Basic Features
- **Amplitude (ΔI)**: Current drop from baseline
- **Standard Deviation**: Signal variability during blockade
- **Dwell Time**: Length of blockade event

#### Advanced Time-Series Features
- **catch22**: 22 canonical time-series features
- **TSFEL**: Comprehensive time-series feature extraction library
  - Temporal features
  - Statistical features
  - Spectral features
  - ECDF-based features

### 3. Feature Selection & Preprocessing
- Removes features with >50% NaN values
- Applies variance threshold filtering (threshold=0.01)
- Fills remaining NaNs with median values
- Removes highly correlated features (correlation >0.65 with mean)
- Drops redundant statistical features to prevent multicollinearity

### 4. Clustering Analysis

#### Standard K-Means Clustering
- Uses StandardScaler for feature normalization
- Applies UMAP for 2D dimensionality reduction
- K-Means with 4 clusters (k=4, n_init=10)

#### Sparse K-Means Clustering
Custom implementation following Witten & Tibshirani (2010):
- **Objective**: Maximize between-cluster sum of squares (BCSS) with L1 constraint on feature weights
- **Sparsity Parameter (s)**: Controls number of features selected (1 ≤ s ≤ √p)
- **Algorithm**:
  1. Initialize weights uniformly
  2. Perform weighted K-means clustering
  3. Compute BCSS for each feature
  4. Update weights using soft-thresholding with L1 constraint
  5. Iterate until convergence
- **Advantages over regular k-means**:
  - Automatic feature selection
  - Improved scores

### 5. Evaluation Metrics
- **Adjusted Rand Index (ARI)**: Measures clustering quality vs. true labels
- **Normalized Mutual Information (NMI)**: Information-theoretic clustering similarity
- **Confusion Matrix**: With Hungarian algorithm alignment for optimal label matching
- **Precision, Recall, F1-Score**: Per-class performance metrics
- **Silhouette Score**: Internal cluster quality measure
- **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster dispersion
- **Davies-Bouldin Index**: Average similarity between clusters

### 6. Feature Importance Analysis
- Visualizes feature weights from sparse K-Means
- Identifies top discriminative features per cluster
- Computes cluster-specific feature means and differences
- Statistical significance testing (t-tests) for feature differences
- Heatmap visualization of feature patterns across clusters

## Key Functions

### `sparse_kmeans(X, K, s, ...)`
Implements sparse K-means clustering:
- `X`: Feature matrix (n_samples × n_features)
- `K`: Number of clusters
- `s`: Sparsity constraint (L1 bound on weights)
- Returns: Cluster labels, feature weights, objective history

### `find_optimal_sparsity(X, K, ...)` 
Grid search for optimal sparsity parameter:
- Tests range of s values
- Evaluates using ARI, silhouette score, or objective function
- Returns: Best s value, corresponding results

### `plot_clusters(pca_features, labels, title)`
Creates clean 2D scatter plots of clustering results with optional label mapping.

## Dependencies
```python
pandas, numpy, scipy
sklearn (preprocessing, clustering, metrics, decomposition)
pyabf  # ABF file reading
tsfel, catch22  # Time-series feature extraction
umap-learn  # Dimensionality reduction
hdbscan  # Density-based clustering
plotly, seaborn, matplotlib  # Visualization
```

## Usage Example
```python
# Load and preprocess data
features_df = pd.read_csv("tsfel.csv")
X = features_df.drop(columns=['label']).values
y = features_df['label'].values

# Apply sparse K-means
labels, weights, objectives = sparse_kmeans(
    X=X, K=4, s=8.6, 
    true_labels=y, 
    max_iter=1000
)

# Evaluate
ari = adjusted_rand_score(y, labels)
print(f"ARI: {ari:.3f}")
```

## Results Interpretation
- **High ARI (>0.7)**: Strong agreement with true labels
- **Feature weights**: Higher values indicate more discriminative features
- **Cluster separation**: Visualized in UMAP 2D projections
- **Feature differences**: Positive values indicate higher in target cluster

## Notes
- Peptide ladders are combined into a single "ladder" label for clustering
- MFCC features are excluded due to computational constraints
- The notebook includes extensive visualization for cluster quality assessment
- Hungarian algorithm ensures optimal alignment between predicted and true labels

## Future Improvements
- Automated sparsity parameter selection
- Deep learning-based feature extraction
- Real-time classification pipeline
- Cross-validation for robust evaluation