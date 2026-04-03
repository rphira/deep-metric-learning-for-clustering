import numpy as np
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score, 
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    fowlkes_mallows_score, adjusted_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score
)
from scipy.spatial.distance import pdist
from scipy.optimize import linear_sum_assignment


class MetricsCalculator:
    """Calculate clustering evaluation metrics. Going through a lot of them here. Not all are equally important.
    We use ARI and Silhouette as primary metrics."""
    
    @staticmethod
    def calculate_comprehensive_metrics(embeddings, true_labels, pred_labels, n_clusters):
        """
        Calculate extensive clustering evaluation metrics.
        """
        metrics = {}
        
        # EXTERNAL METRICS (compare to ground truth)
        metrics['ari'] = adjusted_rand_score(true_labels, pred_labels)
        metrics['nmi'] = normalized_mutual_info_score(true_labels, pred_labels)
        metrics['ami'] = adjusted_mutual_info_score(true_labels, pred_labels)
        metrics['fmi'] = fowlkes_mallows_score(true_labels, pred_labels)
        metrics['homogeneity'] = homogeneity_score(true_labels, pred_labels)
        metrics['completeness'] = completeness_score(true_labels, pred_labels)
        metrics['v_measure'] = v_measure_score(true_labels, pred_labels)
        
        # INTERNAL METRICS (no ground truth)
        try:
            metrics['silhouette'] = silhouette_score(embeddings, pred_labels)
        except:
            metrics['silhouette'] = 0.0
        
        try:
            metrics['calinski_harabasz'] = calinski_harabasz_score(embeddings, pred_labels)
        except:
            metrics['calinski_harabasz'] = 0.0
        
        try:
            metrics['davies_bouldin'] = davies_bouldin_score(embeddings, pred_labels)
        except:
            metrics['davies_bouldin'] = float('inf')
        
        # CLUSTER QUALITY METRICS
        intra_cluster_dists = []
        for cluster_id in np.unique(pred_labels):
            cluster_points = embeddings[pred_labels == cluster_id]
            if len(cluster_points) > 1:
                intra_cluster_dists.append(np.mean(pdist(cluster_points)))
        metrics['mean_intra_cluster_distance'] = np.mean(intra_cluster_dists) if intra_cluster_dists else 0.0
        
        # Inter-cluster distance (separation)
        cluster_centers = []
        for cluster_id in np.unique(pred_labels):
            cluster_points = embeddings[pred_labels == cluster_id]
            cluster_centers.append(np.mean(cluster_points, axis=0))
        cluster_centers = np.array(cluster_centers)
        
        if len(cluster_centers) > 1:
            inter_cluster_dists = pdist(cluster_centers)
            metrics['mean_inter_cluster_distance'] = np.mean(inter_cluster_dists)
            metrics['min_inter_cluster_distance'] = np.min(inter_cluster_dists)
        else:
            metrics['mean_inter_cluster_distance'] = 0.0
            metrics['min_inter_cluster_distance'] = 0.0
        
        # Dunn Index
        if intra_cluster_dists and len(cluster_centers) > 1:
            metrics['dunn_index'] = metrics['min_inter_cluster_distance'] / max(intra_cluster_dists)
        else:
            metrics['dunn_index'] = 0.0
        
        # CONFUSION-BASED METRICS
        cluster_purity = []
        for cluster_id in np.unique(pred_labels):
            cluster_mask = pred_labels == cluster_id
            cluster_true_labels = true_labels[cluster_mask]
            if len(cluster_true_labels) > 0:
                most_common = np.bincount(cluster_true_labels).argmax()
                purity = np.sum(cluster_true_labels == most_common) / len(cluster_true_labels)
                cluster_purity.append(purity)
        metrics['purity'] = np.mean(cluster_purity) if cluster_purity else 0.0
        
        # Cluster balance
        cluster_sizes = [np.sum(pred_labels == i) for i in np.unique(pred_labels)]
        metrics['cluster_size_std'] = np.std(cluster_sizes)
        metrics['cluster_size_cv'] = np.std(cluster_sizes) / np.mean(cluster_sizes) if np.mean(cluster_sizes) > 0 else 0.0
        
        return metrics
    
    @staticmethod
    def calculate_per_protein_metrics(embeddings_2d, true_labels, pred_labels, n_clusters, label_to_name_func):
        """Calculate per-protein metrics with protein names - fixed for HDBSCAN"""
        # Get unique predicted clusters
        unique_preds = np.unique(pred_labels)
        unique_trues = np.unique(true_labels)
        
        # Handle noise points from HDBSCAN
        valid_preds = pred_labels[pred_labels != -1]
        valid_trues = true_labels[pred_labels != -1]
        
        if len(valid_preds) == 0:
            return {}
        
        # confusion matrix
        n_pred_clusters = len(np.unique(valid_preds))
        n_true_clusters = len(unique_trues)
        
        confusion = np.zeros((n_pred_clusters, n_true_clusters))
        
        # Map cluster labels to 0-indexed
        pred_map = {label: idx for idx, label in enumerate(np.unique(valid_preds))}
        true_map = {label: idx for idx, label in enumerate(unique_trues)}
        
        for pred, true in zip(valid_preds, valid_trues):
            confusion[pred_map[pred], true_map[true]] += 1
        
        # Hungarian algorithm for assignment
        row_ind, col_ind = linear_sum_assignment(-confusion)
        
        per_protein_metrics = {}
        
        for pred_idx, true_idx in zip(row_ind, col_ind):
            pred_cluster = list(pred_map.keys())[list(pred_map.values()).index(pred_idx)]
            true_cluster = list(true_map.keys())[list(true_map.values()).index(true_idx)]
            
            pred_mask = valid_preds == pred_cluster
            true_mask = valid_trues == true_cluster
            
            tp = np.sum(pred_mask & true_mask)
            fp = np.sum(pred_mask & ~true_mask)
            fn = np.sum(~pred_mask & true_mask)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            protein_name = label_to_name_func(true_cluster)
            per_protein_metrics[protein_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'n_samples': np.sum(true_mask),
                'encoded_label': true_cluster,
                'assigned_cluster': pred_cluster
            }
        
        # handle unassigned proteins
        assigned_true_clusters = [list(true_map.keys())[list(true_map.values()).index(idx)] for idx in col_ind]
        all_true_clusters = list(true_map.keys())
        
        for true_cluster in all_true_clusters:
            if true_cluster not in assigned_true_clusters:
                protein_name = label_to_name_func(true_cluster)
                per_protein_metrics[protein_name] = {
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'n_samples': np.sum(valid_trues == true_cluster),
                    'encoded_label': true_cluster,
                    'assigned_cluster': None
                }
        
        return per_protein_metrics
    
    @staticmethod
    def print_clustering_metrics(metrics, dataset_name, data_split='test'):
        
        print(f"\n  EXTERNAL METRICS (Agreement with Ground Truth):")
        print(f"     ARI (Adjusted Rand Index):           {metrics['ari']:.4f} ")
        print(f"     NMI (Normalized Mutual Info):        {metrics['nmi']:.4f}")
        print(f"     AMI (Adjusted Mutual Info):          {metrics['ami']:.4f}")
        print(f"     FMI (Fowlkes-Mallows Index):         {metrics['fmi']:.4f}")
        print(f"     Homogeneity:                         {metrics['homogeneity']:.4f} ")
        print(f"     Completeness:                        {metrics['completeness']:.4f}")
        print(f"     V-Measure:                           {metrics['v_measure']:.4f}")
        print(f"     Purity:                              {metrics['purity']:.4f}")
        
        print(f"\n  INTERNAL METRICS (Cluster Quality):")
        print(f"     Silhouette Score:                    {metrics['silhouette']:.4f}")
        print(f"     Calinski-Harabasz Index:             {metrics['calinski_harabasz']:.2f}")
        print(f"     Davies-Bouldin Index:                {metrics['davies_bouldin']:.4f} ")
        print(f"     Dunn Index:                          {metrics['dunn_index']:.4f} ")
        
        print(f"\n  DISTANCE METRICS:")
        print(f"     Mean Intra-cluster Distance:         {metrics['mean_intra_cluster_distance']:.4f}  [Lower is better]")
        print(f"     Mean Inter-cluster Distance:         {metrics['mean_inter_cluster_distance']:.4f}  [Higher is better]")
        print(f"     Min Inter-cluster Distance:          {metrics['min_inter_cluster_distance']:.4f}  [Higher is better]")
        
        print(f"\n  BALANCE METRICS:")
        print(f"     Cluster Size Std Dev:                {metrics['cluster_size_std']:.2f}  [Lower is better]")
        print(f"     Cluster Size CV:                     {metrics['cluster_size_cv']:.4f}")
        
        overall_score = (
            metrics['ari'] + metrics['nmi'] + metrics['v_measure'] + 
            metrics['silhouette'] + (1 - min(metrics['davies_bouldin'], 2)/2)
        ) / 5
        # checking here the averages of the metrics above.

        
