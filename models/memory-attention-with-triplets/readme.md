# Memory-Augmented Attention Triplet Network for Sequential Protein Identification

## Overview
This repository implements a deep learning pipeline for cumulative protein identification using nanopore blockade signals. The system employs a novel architecture combining **triplet loss**, **feature attention mechanisms**, and **memory replay** to learn discriminative embeddings while preventing catastrophic forgetting as new proteins are introduced incrementally.

## Core Architecture

### MemoryAttentionTripletModel
A neural network that learns to:
1. **Identify important features** through attention mechanisms
2. **Generate discriminative embeddings** via triplet loss
3. **Maintain knowledge** across sequential learning steps using memory replay

```
Input Features → Feature Attention → Encoder → Latent Embeddings
                       ↓                              ↓
                 Attention Weights              Triplet Loss
                       ↓                              ↓
                 Feature Importance         Memory Bank Update
```

### Key Components

#### 1. Feature Attention Module
- **Architecture**: Bottleneck network (input_dim → hidden → input_dim)
- **Activation**: Tanh for intermediate layers, Sigmoid for final weights
- **Purpose**: Learn which tsfresh features are most discriminative
- **Regularization**: L1 penalty (λ=1e-4) for sparsity
- **Output**: Per-feature importance weights ∈ [0, 1]

#### 2. Encoder Network
- **Layers**: 4-layer fully connected network
- **Architecture**: 
  - Layer 1: input_dim → hidden_dim (256)
  - Layer 2: hidden_dim → hidden_dim
  - Layer 3: hidden_dim → hidden_dim/2
  - Layer 4: hidden_dim/2 → latent_dim (64)
- **Normalization**: BatchNorm after each linear layer
- **Activation**: LeakyReLU(0.2)
- **Dropout**: 0.3 after each activation

#### 3. Memory Bank Mechanism
- **Inspired by**: iCARL (Incremental Classifier and Representation Learning)
- **Storage**: 50 samples per protein class
- **Sampling**: Uniform random sampling with replacement
- **Purpose**: Preserve knowledge of previously learned proteins
- **Update Strategy**: Reservoir sampling for fixed-size memory

## Training Strategy

### Memory Replay Trainer
Implements continual learning with memory replay to mitigate catastrophic forgetting.

#### Loss Function
```
Total Loss = Current Triplet Loss + α × Memory Triplet Loss
```
- **Current Loss**: Triplet loss on current batch
- **Memory Loss**: Triplet loss on samples from memory bank
- **α (memory_replay_ratio)**: Weight for memory loss

#### Triplet Mining
- **Strategy**: random triplet mining within batches
- **Triplet Construction**:
  - Anchor: Current sample
  - Positive: Different sample, same protein
  - Negative: Sample from different protein
- **Margin**: 1.0 (Euclidean distance)

#### Training Procedure
1. Forward pass with attention-weighted features
2. Mine triplets from current batch
3. Sample triplets from memory bank
4. Compute combined loss
5. Update model parameters
6. Update memory bank with current embeddings

## Sequential Learning Pipeline

### SequentialAnalysis
Orchestrates the complete sequential protein identification workflow.

### Dataset Handling
- **Feature Alignment**: Finds common features across all datasets
- **Preprocessing Pipeline**:
  1. Remove columns with >10% NaN values
  2. Drop tsfresh-specific filtered features (c3, quantiles, etc.)
  3. Remove highly correlated features (r > 0.66 with mean)
  4. StandardScaler normalization
  5. Train/val/test split (60/20/20)

### Sequential Steps
Each step introduces a new protein while maintaining performance on previous proteins:
1. **Step 1**: Train on Protein A
2. **Step 2**: Train on Protein B (memory maintains A)
3. **Step 3**: Train on Protein C (memory maintains A+B)
4. **Step N**: Train on Protein N (memory maintains A+B+...+N-1)

### Evaluation Metrics

#### External Metrics (with ground truth)
- **ARI (Adjusted Rand Index)**: [-1, 1], 1 = perfect clustering
- **NMI (Normalized Mutual Information)**: [0, 1], measures clustering agreement
- **Homogeneity**: All cluster members from same class
- **Completeness**: All class members in same cluster
- **V-Measure**: Harmonic mean of homogeneity and completeness
- **Purity**: Fraction of correct assignments per cluster

#### Internal Metrics (no ground truth needed)
- **Silhouette Score**: [-1, 1], measures cluster separation
- **Calinski-Harabasz Index**: Ratio of between/within cluster dispersion
- **Davies-Bouldin Index**: Average cluster similarity (lower is better)

#### Distance Metrics
- **Intra-cluster Distance**: Compactness of clusters
- **Inter-cluster Distance**: Separation between clusters
- **Cluster Balance**: Standard deviation of cluster sizes

### Attention-Based Importance
- **Method**: Average attention weights across samples
- **Granularity**: Overall + per-protein importance
- **Output**: Feature ranking by attention scores

#### 3. Consistency Analysis
- **Metric**: Mean importance / Std deviation across steps
- **Purpose**: Identify features that remain important throughout learning
- **Output**: Ranked list of consistently discriminative features

## Visualizations

### 1. Training Diagnostics
- **Loss curves**: Total, current, and memory loss
- **Validation metrics**: ARI and NMI evolution
- **Performance comparison**: Train vs test metrics

### 2. UMAP Embeddings
- **True labels**: Ground truth protein assignments
- **Predicted clusters**: K-Means cluster assignments
- **Agreement map**: Overlay of predictions and ground truth
- **Evolution**: UMAP across all sequential steps

### 3. Feature Analysis
- **Top features**: Attention weights
- **GLMNet importance (Now Old)**: Top 30 by ElasticNet coefficients
- **Consistency heatmap**: Feature importance across steps
- **Per-protein importance**: Feature weights per class

### 4. Performance Analysis
- **Per-protein metrics**: Precision, recall, F1-score
- **Evolution plots**: ARI/NMI trajectory across steps
- **Generalization gap**: Train-test performance difference
- **Memory bank status**: Number of proteins in memory

## Usage

### Basic Training
```python
# Initialize analyzer
analyzer = SequentialAnalysis(output_dir="results/")

# Run sequential analysis
results = analyze.run_sequential_analysis()

# Test on unseen data
final_evaluation = analyze.test_on_unseen_data(model, device)
```

### Custom Configuration
```python
model = MemoryAttentionTripletModel(
    input_dim=200,          # Number of features
    hidden_dim=256,         # Encoder hidden size
    latent_dim=64,          # Embedding dimension
    dropout=0.3             # Dropout rate
)

trainer = MemoryReplayTrainer(model, device)

history = trainer.train_step(
    X_train, y_train, X_val, y_val,
    num_epochs=500,
    memory_replay_ratio=0.4,  # Weight for memory loss
    lr=1e-3                   # Learning rate
)
```

## Output Files

### Per-Step Outputs
- `step_X_analysis.png`: Comprehensive metrics and visualizations
- `step_X_umap_comparison.png`: UMAP embeddings (true vs predicted)
- `step_X_glmnet_importance.png`: GLMNet feature rankings
- `checkpoint_step_X.json`: Intermediate results

### Final Outputs
- `evolution_analysis.png`: Performance trajectory across steps
- `umap_evolution_all_steps.png`: UMAP progression
- `feature_consistency_analysis.png`: Top features across steps
- `feature_importance_analysis.json`: Complete feature rankings
- `final_summary_report.txt`: Text summary of all results
- `final_unseen_test.png`: Generalization test results

## Hyperparameters

### Model Architecture
```python
INPUT_DIM = 200        # Number of input features (tsfresh)
HIDDEN_DIM = 256       # Encoder hidden layer size
LATENT_DIM = 64        # Embedding space dimension
DROPOUT = 0.3          # Dropout probability
ATTENTION_L1 = 1e-4    # L1 regularization for attention
```

### Training
```python
BATCH_SIZE = 512                # Training batch size
LEARNING_RATE = 1e-3           # Initial learning rate (1e-4 after step 1)
NUM_EPOCHS = 500               # Maximum epochs per step
PATIENCE = 50                  # Early stopping patience
MEMORY_REPLAY_RATIO = 0.4      # Weight for memory loss
TRIPLET_MARGIN = 1.0           # Margin for triplet loss
WEIGHT_DECAY = 1e-4            # AdamW weight decay
```

### Memory Bank
```python
MEMORY_SIZE_PER_PROTEIN = 50   # Samples stored per protein
MEMORY_TRIPLETS = 50           # Number of memory triplets per batch
```

### Clustering
```python
UMAP_NEIGHBORS = 15            # UMAP n_neighbors
UMAP_MIN_DIST = 0.1           # UMAP min_dist
KMEANS_INIT = 20              # K-Means n_init
```

## Dependencies
```python
torch
scikit-learn 
umap-learn 
hdbscan
pandas, numpy
matplotlib, seaborn
```

## Theoretical Foundation

### Triplet Loss
Learns embeddings where:
- Distance(anchor, positive) < Distance(anchor, negative) + margin

### Memory
Mitigates catastrophic forgetting by:
1. Storing representative samples from past tasks
2. Replaying them during new task training
3. Balancing new and old task loss

### Feature Attention
Improves interpretability by:
1. Learning feature-wise importance weights
2. Weighting input features before encoding

- GPU recommended for training (CUDA support)
- Requires preprocessed tsfresh features
- Label mapping must be defined in analyzer
- Common features determined from all datasets
- Memory bank uses reservoir sampling for efficiency