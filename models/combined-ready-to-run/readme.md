## Quick Start

### 1. Set Data Location
export ML4NP_DATA_ROOT="/path/where/your/csv/is/located"

### 1. Set Data Location (for Rushang's delivery)
export ML4NP_DATA_ROOT="/mnt/data_share/rushang"

### Run Analysis
python triplet-metric-learning-combined.py

### Requirements
pip install torch numpy pandas scikit-learn umap-learn hdbscan matplotlib glmnet