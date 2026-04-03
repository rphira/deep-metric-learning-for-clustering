import torch
import pandas as pd
import numpy as np
from glmnet import ElasticNet
import sys
sys.path.append('/home/rushang_phira/src/models/')
from improved_dsc_triplet import TripletDSCModel 
import os
import random
import ast
from datetime import datetime
import shap
import pandas as pd
import numpy as np
import torch
import hdbscan
import json
import umap
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold, f_classif
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from glmnet import ElasticNet
import warnings
from scipy.spatial.distance import pdist, cdist
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, homogeneity_score, completeness_score, v_measure_score
import matplotlib.gridspec as gridspec
import warnings
import random
from sklearn.cluster import MiniBatchKMeans
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning)

# post_hoc_elasticnet_analysis.py
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from glmnet import ElasticNet
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the same data used in original training"""
    print("Loading original data...")
    
    k8_k12_k16 =  pd.read_csv("/home/rushang_phira/src/data/complete_feature_sets/all_ladders_tsfresh.csv")
    #.rename(columns={'label': 'final_label'}).replace([float('inf'), float('-inf')], float('nan'))
    
    # k8_k12_k16 = k8_k12_k16[k8_k12_k16["final_label"].str[:3] == "PL6"]
    k8_k12_k16 = k8_k12_k16[k8_k12_k16['final_label'].notna()]
    nan_percentage = k8_k12_k16.isna().mean() * 100
    k8_k12_k16 = k8_k12_k16.loc[:, nan_percentage <= 10] # Drop columns where NaN percentage is greater than 50%
    k8_k12_k16 = k8_k12_k16.dropna()

    # cleaned_data = k8_k12_k16.loc[:, ~k8_k12_k16.columns.str.contains('c3|value__quantile|value__agg_linear_trend|large_standard_deviation|change_quantiles|cwt')]
    # to_drop = ['EventId', 'idxstart', 'idxend', 'risetime', 'value__standard_deviation', 'value__mean_second_derivative_central', 'value__linear_trend__attr_"intercept"', 'falltime', "I/Io", "Io", "Irms", "length", "value__maximum", "value__minimum", "log_length", 'value__variance_larger_than_standard_deviation', "value__median", "value__mean", 'value__root_mean_square', 'value__standard_deviation', 'value__variance', 'Imean', 'Isig', 'isBL?']
    # cleaned_data = cleaned_data.drop(columns=to_drop)

    cleaned_data = k8_k12_k16.loc[:, ~k8_k12_k16.columns.str.contains('c3|value__quantile|value__agg_linear_trend|large_standard_deviation|quantile|change_quantiles')]
    
    '''Dropping features with greater than 50% NaN values'''
    nan_threshold = 0.5 * len(cleaned_data)
    final_feature_df = cleaned_data.dropna(axis=1, thresh=nan_threshold)

    '''Dropping features with variance lower than 0,1'''
    numeric_features = final_feature_df.select_dtypes(include=['number'])
    selector = VarianceThreshold(threshold=0.001)
    reduced_features = selector.fit_transform(numeric_features)
    retained_columns = numeric_features.columns[selector.get_support()]
    final_feature_df = final_feature_df[retained_columns.tolist() + ['final_label']]  # Keep label

    '''Filling remaining NaNs with median of that feature'''
    # final_feature_df.loc[:, final_feature_df.columns != 'label'] = final_feature_df.loc[:, final_feature_df.columns != 'label'].fillna(final_feature_df.median(numeric_only=True))
    
    '''Going by label and removing 10% of outliers using IsolationForest. Checking if I can make the clusters more prominent'''
    
    def remove_outliers_by_label(df, label_column, contamination=0.3):
        cleaned_df = pd.DataFrame()  # To store the cleaned data
        
        # Iterate through each unique label
        for label in df[label_column].unique():
            # Filter the rows corresponding to the current label
            subset = df[df[label_column] == label]
            
            # Apply Isolation Forest for outlier detection
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            outliers = iso_forest.fit_predict(subset.drop(columns=[label_column]))  # Drop the label column
            
            # Outliers are labeled as -1, keep only the rows where outlier is 1 (inliers)
            subset_cleaned = subset[outliers == 1]
            
            # Append the cleaned subset to the result dataframe
            cleaned_df = pd.concat([cleaned_df, subset_cleaned], ignore_index=False)
        
        return cleaned_df

    # Apply the function to remove outliers
    # final_feature_df = remove_outliers_by_label(final_feature_df, label_column='final_label', contamination=0.3)
    
    cleaned_data = final_feature_df
    to_drop = ['EventId', 'idxstart', 'idxend', 'risetime', 'value__mean_abs_change', 'value__standard_deviation', 'value__linear_trend__attr_"intercept"', 'falltime', "I/Io", "Io", "Irms", "length", "value__maximum", "value__minimum", "log_length", "value__median", 'value__root_mean_square', 'value__standard_deviation', 'value__variance', 'Imean', 'isBL?','Isig']

    if 'value__mean' in cleaned_data.columns:
        print("Mean present")
        correlations = cleaned_data.drop(columns=['final_label']).corr()['value__mean'].abs()
        high_corr_features = correlations[correlations > 0.66].index.tolist()
        high_corr_features = [f for f in high_corr_features if f != 'value__mean']
        if high_corr_features:
            print("Highly correlated features with 'value__mean':")
            for feature in high_corr_features:
                print(f"{feature}: {correlations[feature]}")
        to_drop.extend(high_corr_features)
        print(f"Added {len(high_corr_features)} mean-correlated features to drop")
    columns_to_drop = ['value__mean'] + high_corr_features + to_drop
    cleaned_data = cleaned_data.drop(columns=[col for col in columns_to_drop if col in cleaned_data.columns])
    
    cleaned_data['final_label'] = cleaned_data['final_label'].str[3:]
    def cluster_downsample(df, label_column, clusters_per_class=6500):
        reduced = []
        for label, group in df.groupby(label_column):
            X = group.drop(columns=[label_column])
            n_clusters = min(clusters_per_class, len(group))
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(X)
            centers = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)
            centers[label_column] = label
            reduced.append(centers)
        return pd.concat(reduced, ignore_index=True)
    
    def stratified_downsample(df, label_column, samples_per_class=6500, random_state=42):
        sampled_dfs = []
        for label in df[label_column].unique():
            label_df = df[df[label_column] == label]
            n_samples = min(samples_per_class, len(label_df))
            sampled = label_df.sample(n=n_samples, random_state=random_state)
            sampled_dfs.append(sampled)
        return pd.concat(sampled_dfs, ignore_index=True)
    
    cleaned_data = stratified_downsample(cleaned_data, 'final_label', samples_per_class=6500)


    feature_names = cleaned_data.drop(columns="final_label").columns.tolist()
    
    print(f"Final dataset shape: {cleaned_data.shape}")
    print(f"Labels: {cleaned_data['final_label'].value_counts().to_dict()}")
    
    return cleaned_data, feature_names

# post_hoc_vif_analysis.py
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import HDBSCAN
import torch.nn.functional as F
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')


def load_trained_model(model_path, input_dim, hidden_dim=256, latent_dim=64):
    """Load trained model"""
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = TripletDSCModel(input_dim, hidden_dim, latent_dim).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model, device

def calculate_vif_importance(model, features, clusters, feature_names, device, 
                                  n_bootstrap=50, subsample_size=0.8, 
                                  alpha=0.1, n_lambda=100, standardize=True):
    """
    VIF analysis using glmnet ElasticNet like Köber et al.
    Köber used LASSO (L1), but ElasticNet gives L1 + L2 control. We use L1 here (l1_ratio=1.0)
    1. For each bootstrap iteration:
        a. Subsample data without replacement
        b. For each latent dimension:
            i. Fit ElasticNet to predict latent dim from features
            ii. Record which features have non-zero coefficients
    2. After all bootstraps, calculate VIF score for each feature as:
        VIF(feature) = (# times feature selected) / (total # models)
    3. Return DataFrame with features, VIF scores, and ranks
    4. Higher VIF = more stable/relevant feature
    5. Parameters:
        - model: trained model to get latent representations
        - features: feature matrix (numpy array or DataFrame)
        - clusters: cluster assignments for each sample
        - feature_names: list of feature names
        - device: torch device (cpu or cuda)
        - n_bootstrap: number of bootstrap subsamples
        - subsample_size: fraction of data to subsample each iteration
        - alpha: regularization strength for ElasticNet
        - n_lambda: number of lambda values to consider in glmnet
        - standardize: whether to standardize features before fitting
    6. Returns:
        - vif_df: DataFrame with features, VIF scores, and ranks
    7. Note: Assumes clusters are provided to exclude noise (-1) if needed
    """
    from glmnet import ElasticNet
    from sklearn.linear_model import Lasso, ElasticNet
    # Remove noise cluster (-1)
    mask = clusters != -1
    X = features[mask]
    
    n_samples, n_features = X.shape
    m = int(subsample_size * n_samples)
    
    # Standardize features
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    # Get latent representations
    with torch.no_grad():
        X_tensor = torch.tensor(X_std, dtype=torch.float32, device=device)
        latent_full = model(X_tensor)
        latent_full_norm = F.normalize(latent_full, p=2, dim=1).cpu().numpy()  # ← ADD NORMALIZATION

    n_latent_dims = latent_full_norm.shape[1]
    inclusion_counts = np.zeros(n_features)
    total_models = 0
    rng = np.random.default_rng(seed=42)

    print(f"VIF analysis: {n_latent_dims} latent dims, {n_bootstrap} subsamples")
    print(f"Using glmnet ElasticNet with alpha={alpha}")
    
    for bootstrap_idx in range(n_bootstrap):
        print(f"  Subsample {bootstrap_idx}/{n_bootstrap}")
            
        try:
            # Subsampling without replacement
            indices = rng.choice(n_samples, m, replace=False)
            X_sub = X_std[indices]
            latent_sub = latent_full_norm[indices]
            
            # For each latent dimension
            for latent_dim in range(n_latent_dims):
                y_target = latent_sub[:, latent_dim]  # Continuous target
            
                # we use lasso for feature "selection"
                lasso = ElasticNet(alpha=alpha, l1_ratio=1.0, max_iter=1000, random_state=42)
                lasso.fit(X_sub, y_target)
                
                significant_features = np.abs(lasso.coef_) > 0.001
                inclusion_counts[significant_features] += 1
                total_models += 1

    # Some error handling       
        except Exception as e:
            print(f"failed for bootstrap {bootstrap_idx}: {e}")
            continue
    
    if total_models == 0:
        raise ValueError("No successful subsample fits")
    
    # Calculate VIF scores
    vif_scores = inclusion_counts / total_models
    
    vif_df = pd.DataFrame({
        "feature": feature_names,
        "vif_score": vif_scores,
        "rank_vif": len(vif_scores) - np.argsort(np.argsort(vif_scores))
    }).sort_values("vif_score", ascending=False)
    
    print(f"Completed {total_models} models across {n_bootstrap} subsamples")
    return vif_df

def recreate_test_set_clusters(model, device, features, feature_names):
    """
    recreate exact test set and clusters from training
    """
    # recreate split
    X = features
    y = features['final_label'].values
    
    le = LabelEncoder()
    true_labels = le.fit_transform(y) if y is not None else None
    
    if true_labels is not None:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, true_labels, test_size=0.2, stratify=true_labels, random_state=42
        )
    else:
        X_temp, X_test = train_test_split(X, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_temp.drop(columns=['final_label']))
    X_test_scaled = scaler.transform(X_test.drop(columns=['final_label']))
    
    # Get cluster predictions
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32, device=device)
        latent_test = model(X_test_tensor)
        latent_test_norm = F.normalize(latent_test, p=2, dim=1).cpu().numpy()
        clusters_test = HDBSCAN(min_cluster_size=10).fit_predict(latent_test_norm)
    
    return X_test_scaled, clusters_test, le

def compare_with_og_importance(vif_results, original_importance_path, output_dir):
    """Compare VIF results with old (non vif version) feature importance"""
    original_importance = pd.read_csv(original_importance_path)
    
    comparison = original_importance.merge(
        vif_results[['feature', 'vif_score', 'rank_vif']], 
        on='feature', 
        how='left'
    )
    
    # calculate rank differences
    comparison['rank_difference'] = comparison['rank_vif'] - comparison.index
    comparison['abs_rank_difference'] = np.abs(comparison['rank_difference'])
    
    print("\nTop 20 features by VIF score:")
    print(comparison[['feature', 'vif_score', 'score_combined', 'rank_difference']].head(20))
    
    # correlation between methods
    valid_scores = comparison[['score_combined', 'vif_score']].dropna()
    if len(valid_scores) > 1:
        spearman_corr, p_value = spearmanr(valid_scores['score_combined'], valid_scores['vif_score'])
        print(f"\nSpearman correlation between original importance and VIF: {spearman_corr:.3f} (p={p_value:.3f})")
    
    comparison.to_csv(os.path.join(output_dir, "feature_importance_comparison.csv"), index=False)
    
    return comparison

def cluster_specific_vif(model, features, clusters, feature_names, device):
    """Calculate VIF separately for each cluster"""
    unique_clusters = np.unique(clusters[clusters != -1])
    cluster_vif_results = {}
    
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        X_cluster = features[cluster_mask]
        
        cluster_vif = calculate_vif_importance(
            model, X_cluster, clusters[cluster_mask], feature_names, device,
            n_bootstrap=100, subsample_size=0.8
        )
        cluster_vif_results[cluster_id] = cluster_vif
    
    return cluster_vif_results
def create_vif_visualizations(vif_results, comparison, output_dir):
    """visualizations for VIF analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Top features by VIF score
    top_vif = vif_results.head(20)
    axes[0, 0].barh(range(len(top_vif)), top_vif['vif_score'])
    axes[0, 0].set_yticks(range(len(top_vif)))
    axes[0, 0].set_yticklabels(top_vif['feature'], fontsize=8)
    axes[0, 0].set_xlabel('VIF Score')
    axes[0, 0].set_title('Top 20 Features by VIF Score')
    axes[0, 0].invert_yaxis()
    
    # Plot 2: VIF score distribution
    axes[0, 1].hist(vif_results['vif_score'], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('VIF Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of VIF Scores')
    axes[0, 1].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='VIF=0.5')
    axes[0, 1].axvline(x=0.8, color='orange', linestyle='--', alpha=0.7, label='VIF=0.8')
    axes[0, 1].legend()
    
    # Plot 3: Rank comparison
    valid_comparison = comparison.dropna(subset=['score_combined', 'vif_score'])
    if len(valid_comparison) > 0:
        axes[1, 0].scatter(valid_comparison['score_combined'], valid_comparison['vif_score'], alpha=0.6)
        axes[1, 0].set_xlabel('Original Importance Score')
        axes[1, 0].set_ylabel('VIF Score')
        axes[1, 0].set_title('VIF vs Original Importance')
        
        # Add correlation to plot
        spearman_corr, _ = spearmanr(valid_comparison['score_combined'], valid_comparison['vif_score'])
        axes[1, 0].text(0.05, 0.95, f'Spearman ρ = {spearman_corr:.3f}', 
                       transform=axes[1, 0].transAxes, fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Plot 4: Stability. I set here (arbitrarily) bins for stability. Although in practice, we do not see stable features
    stability_bins = [
        ('Very Stable (VIF > 0.8)', len(vif_results[vif_results['vif_score'] > 0.8])),
        ('Stable (0.5 < VIF ≤ 0.8)', len(vif_results[(vif_results['vif_score'] > 0.5) & (vif_results['vif_score'] <= 0.8)])),
        ('Moderate (0.2 < VIF ≤ 0.5)', len(vif_results[(vif_results['vif_score'] > 0.2) & (vif_results['vif_score'] <= 0.5)])),
        ('Unstable (VIF ≤ 0.2)', len(vif_results[vif_results['vif_score'] <= 0.2]))
    ]
    
    categories, counts = zip(*stability_bins)
    axes[1, 1].bar(categories, counts, color=['green', 'lightgreen', 'orange', 'red'])
    axes[1, 1].set_title('Feature Stability Categories')
    axes[1, 1].set_ylabel('Number of Features')
    plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    for i, count in enumerate(counts):
        axes[1, 1].text(i, count + 0.5, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "vif_analysis_visualization.png"), dpi=300, bbox_inches='tight')
    plt.show()

def main():
    output_dir = "/home/rushang_phira/src/report/all_ladders_stratified"
    model_path = os.path.join(output_dir, "model_weights.pt")
    cluster_path = os.path.join(output_dir, "clusters.npy")
    original_importance_path = os.path.join(output_dir, "feature_importance.csv")
    clusters = np.load(cluster_path)

    vif_subdir = os.path.join(output_dir, f"vif_analysis")
    os.makedirs(vif_subdir, exist_ok=True)
    print("Starting VIF analysis on existing results...")

    cleaned_data, feature_names = load_and_preprocess_data()
    print(f"Features shape: {cleaned_data.shape}")
    print(f"Number of features: {len(feature_names)}")
    
    input_dim = len(feature_names)
    model, device = load_trained_model(model_path, input_dim)
    print("Model loaded successfully!")
    
    X_test_scaled, clusters_test, le = recreate_test_set_clusters(model, device, cleaned_data, feature_names)
    print(f"Recreated test set: {X_test_scaled.shape}")
    print(f"Clusters: {np.unique(clusters_test, return_counts=True)}")

    cluster_info = pd.DataFrame({
        'cluster': clusters_test,
        'count': np.ones(len(clusters_test))
    })
    cluster_info.groupby('cluster').count().to_csv(os.path.join(vif_subdir, "cluster_distribution.csv"))



    vif_results = calculate_vif_importance(
        model=model,
        features=X_test_scaled,
        clusters=clusters_test,
        feature_names=feature_names,
        device=device,
        n_bootstrap=100,  
        subsample_size=0.8
    )
 
    comparison = compare_with_og_importance(vif_results, original_importance_path, vif_subdir)
    
    create_vif_visualizations(vif_results, comparison, vif_subdir)
    
    vif_results.to_csv(os.path.join(vif_subdir, "vif_analysis_results.csv"), index=False)
    
    print(f"Results saved to {vif_subdir}")
    # print(f"Stable features (VIF > 0.5): {len(vif_results[vif_results['vif_score'] > 0.5])}")
    # print(f"Very stable features (VIF > 0.8): {len(vif_results[vif_results['vif_score'] > 0.8])}")
    
    return vif_results, comparison

if __name__ == "__main__":
    vif_results, comparison = main()