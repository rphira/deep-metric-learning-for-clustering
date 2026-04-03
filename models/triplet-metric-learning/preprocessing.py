"""
Data preprocessing utilities for feature engineering
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import IsolationForest
from sklearn.cluster import MiniBatchKMeans


def preprocess_data(df, label_column='final_label'):
    """
    Complete preprocessing pipeline for the dataset
    """
    # Filter for specific labels if needed
    # df = df[df[label_column].str[:3] == "PL3"]
    df = df[df[label_column].notna()]

    # Drop columns with >10% NaN
    nan_percentage = df.isna().mean() * 100
    df = df.loc[:, nan_percentage <= 10]
    df = df.dropna()

    # Remove unwanted feature patterns
    df = df.loc[:, ~df.columns.str.contains(
        'c3|value__quantile|value__agg_linear_trend|large_standard_deviation|quantile|change_quantiles'
    )]
    
    # Drop features with variance lower than threshold
    nan_threshold = 0.5 * len(df)
    df = df.dropna(axis=1, thresh=nan_threshold)

    numeric_features = df.select_dtypes(include=['number'])
    selector = VarianceThreshold(threshold=0.1) # Threshold can be adjusted as needed
    reduced_features = selector.fit_transform(numeric_features)
    retained_columns = numeric_features.columns[selector.get_support()]
    df = df[retained_columns.tolist() + [label_column]]

    return df


def remove_redundant_features(df, label_column='final_label', corr_threshold=0.65):
    """
    Remove features highly correlated with value__mean
    """
    to_drop = [
        'EventId', 'idxstart', 'idxend', 'risetime', 'value__mean_abs_change',
        'value__standard_deviation', 'value__linear_trend__attr_"intercept"',
        'falltime', "I/Io", "Io", "Irms", "length", "value__maximum",
        "value__minimum", "log_length", "value__median", 'value__root_mean_square',
        'value__standard_deviation', 'value__variance', 'Imean', 'isBL?', 'Isig'
    ]

    high_corr_features = []
    if 'value__mean' in df.columns:
        print("Mean present")
        correlations = df.drop(columns=[label_column]).corr()['value__mean'].abs()
        high_corr_features = correlations[correlations > corr_threshold].index.tolist()
        high_corr_features = [f for f in high_corr_features if f != 'value__mean']
        
        if high_corr_features:
            print("Highly correlated features with 'value__mean':")
            for feature in high_corr_features:
                print(f"{feature}: {correlations[feature]}")
        print(f"Added {len(high_corr_features)} mean-correlated features to drop")

    columns_to_drop = ['value__mean'] + high_corr_features + to_drop
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    return df


def remove_outliers_by_label(df, label_column, contamination=0.3):
    """
    Remove outliers using IsolationForest per label
    """
    cleaned_df = pd.DataFrame()
    
    for label in df[label_column].unique():
        subset = df[df[label_column] == label]
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outliers = iso_forest.fit_predict(subset.drop(columns=[label_column]))
        subset_cleaned = subset[outliers == 1]
        cleaned_df = pd.concat([cleaned_df, subset_cleaned], ignore_index=False)
    
    return cleaned_df


def kmeans_downsample(df, label_column, clusters_per_class=6500):
    """
    Downsample using K-means cluster centers
    """
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