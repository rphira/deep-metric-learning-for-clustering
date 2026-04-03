"""
Feature importance calculation utilities. Old, and unused.
"""
import numpy as np
import pandas as pd
import torch
from glmnet import ElasticNet

'''This is the old feature importance calculation method, kept here for reference.'''
def calculate_feature_importance(model, features, feature_names, device, best_feature_scores):
    """GLMNet based feature importance calculation (OLD METHOD). Also includes model weights method."""
    
    # Method 1: Model weights
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "score_model_weights": best_feature_scores
    })
    
    # Method 2: GLMNet importance
    with torch.no_grad():
        X_full_tensor = torch.tensor(features, dtype=torch.float32, device=device)
        latent_full = model(X_full_tensor).cpu().numpy()
    
    coef_list = []
    for i in range(latent_full.shape[1]):
        m = ElasticNet(alpha=1.0, n_lambda=100)
        m.fit(features, latent_full[:, i])
        lambda_idx = np.argmin(np.abs(m.lambda_path_ - m.lambda_best_))
        coef_list.append(np.abs(m.coef_path_[:, lambda_idx]))
    
    glmnet_scores = np.mean(coef_list, axis=0)
    importance_df["score_glmnet"] = glmnet_scores
    
   # Combine scores by averaging (can be adjusted as needed)
    importance_df["score_combined"] = importance_df["score_glmnet"] + importance_df["score_model_weights"]/2
    
    return importance_df.sort_values(by="score_combined", ascending=False)