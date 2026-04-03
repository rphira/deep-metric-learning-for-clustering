"""
Utility functions for saving results and setting random seeds
"""
import os
import random
import numpy as np
import pandas as pd
import torch

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True) 
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def save_results(output_dir, clusters, importance_df, model, history):
    """Save all training results to disk"""
    os.makedirs(output_dir, exist_ok=True)

    # Save cluster assignments
    np.save(os.path.join(output_dir, "clusters.npy"), clusters)

    # Save top features
    importance_df.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)

    # Save model weights
    torch.save(model.state_dict(), os.path.join(output_dir, "model_weights.pt"))

    # Save training history
    pd.DataFrame(history).to_csv(os.path.join(output_dir, "training_history.csv"), index=False)

    print(f"\nAll results saved to '{output_dir}'")