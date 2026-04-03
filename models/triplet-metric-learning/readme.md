# Deep Metric Learning for Time Series Clustering

A comprehensive framework for learning discriminative embeddings of time series data using triplet loss and deep neural networks, with extensive clustering evaluation capabilities.

## Overview

This project implements a deep metric learning approach to discover meaningful clusters in time series data. The model learns a latent space where similar time series are pulled together and dissimilar ones are pushed apart, enabling effective clustering even without labeled data. The framework includes comprehensive evaluation metrics and visualization tools to analyze clustering quality.

## Key Features

- **Triplet Loss Training**: Uses triplet margin loss to learn discriminative embeddings
- **Deep Encoder Architecture**: Multi-layer neural network with batch normalization and dropout
- **HDBSCAN Clustering**: Density-based clustering that handles noise and irregular cluster shapes
- **Comprehensive Evaluation**: Multiple internal and external clustering metrics
- **Feature Importance Analysis**: Older GLMNet-based feature selection and importance scoring, but for latest VIF version, refer to vif.py
- **Visualization Suite**: UMAP projections, training history, and cluster analysis plots
- **Robust Preprocessing**: Automatic outlier removal, variance thresholding, and correlation filtering

## Architecture

### Model Architecture
```python
TripletDSCModel(
    encoder: Sequential(
        Linear(input_dim → hidden_dim)
        BatchNorm1d + LeakyReLU + Dropout
        Linear(hidden_dim → hidden_dim)
        BatchNorm1d + LeakyReLU + Dropout  
        Linear(hidden_dim → hidden_dim//2)
        BatchNorm1d + LeakyReLU + Dropout
        Linear(hidden_dim//2 → latent_dim)
    )
)
```

### Training Pipeline
1. **Data Preparation**: 60/20/20 train/validation/test split with stratification
2. **Triplet Mining**: Online mining of anchor-positive-negative triplets during training
3. **Early Stopping**: Based on validation ARI with configurable threshold
4. **Clustering**: HDBSCAN on normalized latent embeddings
5. **Evaluation**: Multi-metric analysis on test set

## Installation

### Dependencies
```bash
pip install torch numpy pandas scikit-learn hdbscan umap-learn matplotlib shap seaborn glmnet
```

### Hardware Requirements
- CUDA-capable GPU recommended (automatically uses GPU if available)
- 16GB+ RAM for large datasets

## Usage

### Basic Usage
```python
from deep_time_cluster import train_with_triplet, ClusterAnalyzer

# Prepare your data
X = features_array  # (n_samples, n_features)
y = labels_array    # (n_samples,)
feature_names = feature_names_list

# Train the model
model, history, preds, importance_df, analysis = train_with_triplet(
    features=X,
    labels=y,
    feature_names=feature_names,
    output_dir="./results",
    hidden_dim=256,
    latent_dim=64,
    num_epochs=500,
    ari_threshold=0.85
)
```

### Command Line
```bash
python deep_time_cluster.py
```

## Configuration Parameters

### Training Parameters
- `hidden_dim`: Size of hidden layers (default: 128)
- `latent_dim`: Size of output embedding (default: 64)
- `num_epochs`: Maximum training epochs (default: 500)
- `lr`: Learning rate (default: 1e-3)
- `margin`: Triplet loss margin (default: 1.0)
- `batch_size`: Training batch size (default: 512)
- `triplet_weight`: Weight for triplet loss (default: 1.0)
- `ari_threshold`: Early stopping ARI threshold (default: 0.85)

### Data Preprocessing
- **Variance Thresholding**: Removes low-variance features (<0.001)
- **Correlation Filtering**: Drops features highly correlated (>0.66) with mean
- **Outlier Removal**: Optional IsolationForest-based outlier detection per class
- **Downsampling**: Optional stratified downsampling

## Evaluation Metrics

### External Metrics (requires true labels)
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Homogeneity, Completeness, V-measure
- Cluster Purity

### Internal Metrics
- Silhouette Score
- Calinski-Harabasz Index
- Davies-Bouldin Index
- Cluster Separation and Density

## Output Files

The framework generates comprehensive outputs:

```
results/
├── test_set_analysis.png          # Comprehensive cluster analysis
├── umap_comparison.png            # UMAP visualizations
├── training_history.png           # Training progress plots
├── training_history.csv           # Training metrics history
├── feature_importance.csv         # Feature importance scores
├── clusters.npy                   # Cluster assignments
├── model_weights.pt               # Trained model weights
└── training_metadata_TIMESTAMP.json # Training metadata
```

## Data Format

### Input Format
- **Features**: 2D numpy array or pandas DataFrame (samples × features)
- **Labels**: 1D array of class labels (for evaluation)
- **Feature Names**: List of feature names for interpretability

### Example Dataset
Use pseudolabelled datasets, as provided in /combined-ready-to-run


## Performance Tips

1. **Dataset Size**: For datasets >50k samples, consider using the `cluster_downsample` function
2. **Feature Selection**: Use variance thresholding and correlation filtering to reduce dimensionality
3. **Memory Management**: Adjust `batch_size` based on available GPU memory
4. **Early Stopping**: Monitor validation ARI and adjust `ari_threshold` based on dataset complexity

## Troubleshooting

### Common Issues
1. **Low ARI scores**: Try increasing `latent_dim` or adjusting HDBSCAN `min_cluster_size`
2. **Training instability**: Reduce learning rate or increase `batch_size`
3. **Memory errors**: Enable downsampling or reduce `hidden_dim`
4. **Poor clustering**: Check feature preprocessing and consider adding domain-specific features
- Review the documentation for parameter explanations

---

**Note**: This framework is designed for research purposes and may require tuning for specific applications. Always validate results with domain knowledge and multiple evaluation metrics.