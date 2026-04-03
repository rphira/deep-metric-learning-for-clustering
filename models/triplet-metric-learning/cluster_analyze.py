"""
clustering analysis and visualization
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    normalized_mutual_info_score, adjusted_rand_score, silhouette_score,
    calinski_harabasz_score, davies_bouldin_score,
    homogeneity_score, completeness_score, v_measure_score,
    confusion_matrix
)
from scipy.spatial.distance import pdist, cdist


class ClusterAnalyze:
    """
    Analyze clustering results with relevant metrics
    """
    def __init__(self):
        self.analysis_results = {}
    
    def analyze_clustering(self, features, le, true_labels, pred_labels, latent_embeddings, feature_names, dataset_name):
        """cluster analysis with multiple metrics"""
        
        analysis = {
            'dataset': dataset_name,
            'basic_metrics': self._calculate_basic_metrics(true_labels, pred_labels),
            'internal_metrics': self._calculate_internal_metrics(latent_embeddings, pred_labels),
            'cluster_characteristics': self._analyze_cluster_characteristics(features, true_labels, pred_labels),
            'confusion_analysis': self._analyze_confusion_matrix(true_labels, pred_labels, label_names=le.classes_)
        }
        
        self.analysis_results[dataset_name] = analysis
        return analysis
    
    def _calculate_basic_metrics(self, true_labels, pred_labels):
        """basic clustering metrics"""
        valid_mask = pred_labels != -1
        if np.sum(valid_mask) < 2:
            return {}
            
        true_clean = true_labels[valid_mask]
        pred_clean = pred_labels[valid_mask]
        
        if len(np.unique(pred_clean)) < 2:
            return {}
        
        return {
            'ari': adjusted_rand_score(true_clean, pred_clean),
            'nmi': normalized_mutual_info_score(true_clean, pred_clean),
            'homogeneity': homogeneity_score(true_clean, pred_clean),
            'completeness': completeness_score(true_clean, pred_clean),
            'v_measure': v_measure_score(true_clean, pred_clean),
            'purity': self._calculate_purity(true_clean, pred_clean),
            'noise_ratio': np.sum(pred_labels == -1) / len(pred_labels),
            'n_clusters_found': len(np.unique(pred_clean[pred_clean != -1])),
            'n_true_clusters': len(np.unique(true_clean))
        }
    
    def _calculate_internal_metrics(self, embeddings, pred_labels):
        """internal validation metrics"""
        valid_mask = pred_labels != -1
        if np.sum(valid_mask) < 2:
            return {}
            
        embeddings_clean = embeddings[valid_mask]
        pred_clean = pred_labels[valid_mask]
        
        if len(np.unique(pred_clean)) < 2:
            return {}
        
        return {
            'silhouette': silhouette_score(embeddings_clean, pred_clean),
            'calinski_harabasz': calinski_harabasz_score(embeddings_clean, pred_clean),
            'davies_bouldin': davies_bouldin_score(embeddings_clean, pred_clean),
            'cluster_separation': self._calculate_cluster_separation(embeddings_clean, pred_clean),
            'cluster_density': self._calculate_cluster_density(embeddings_clean, pred_clean)
        }
    
    def _calculate_purity(self, true_labels, pred_labels):
        """Calculate cluster purity"""
        contingency_matrix = np.zeros((len(np.unique(pred_labels)), len(np.unique(true_labels))))
        
        for i, true_label in enumerate(np.unique(true_labels)):
            for j, pred_label in enumerate(np.unique(pred_labels)):
                contingency_matrix[j, i] = np.sum((true_labels == true_label) & (pred_labels == pred_label))
        
        return np.sum(np.max(contingency_matrix, axis=1)) / len(true_labels)
    
    def _calculate_cluster_separation(self, embeddings, pred_labels):
        """Calculate minimum inter-cluster distance"""
        unique_labels = np.unique(pred_labels)
        if len(unique_labels) < 2:
            return 0
        
        cluster_centers = []
        for label in unique_labels:
            cluster_points = embeddings[pred_labels == label]
            cluster_centers.append(np.mean(cluster_points, axis=0))
        
        cluster_centers = np.array(cluster_centers)
        pairwise_distances = cdist(cluster_centers, cluster_centers)
        np.fill_diagonal(pairwise_distances, np.inf)
        
        return np.min(pairwise_distances)
    
    def _calculate_cluster_density(self, embeddings, pred_labels):
        """Calculate average cluster density"""
        densities = []
        for label in np.unique(pred_labels):
            cluster_points = embeddings[pred_labels == label]
            if len(cluster_points) > 1:
                avg_distance = np.mean(pdist(cluster_points))
                densities.append(1 / (1 + avg_distance))
            else:
                densities.append(1.0)
        
        return np.mean(densities)
    
    def _analyze_cluster_characteristics(self, features, true_labels, pred_labels):
        """Analyze characteristics of each cluster"""
        characteristics = {}
        unique_pred = np.unique(pred_labels)
        
        for cluster_id in unique_pred:
            if cluster_id == -1:
                continue
            
            cluster_mask = pred_labels == cluster_id
            if np.sum(cluster_mask) == 0:
                continue
                
            cluster_features = features[cluster_mask]
            cluster_true_labels = true_labels[cluster_mask]
            
            if len(cluster_true_labels) > 0:
                true_label_counts = {}
                for label in np.unique(cluster_true_labels):
                    true_label_counts[label] = np.sum(cluster_true_labels == label)
                dominant_label = max(true_label_counts, key=true_label_counts.get)
                purity = true_label_counts[dominant_label] / len(cluster_true_labels)
            else:
                dominant_label = "None"
                purity = 0
            
            characteristics[cluster_id] = {
                'size': len(cluster_features),
                'dominant_true_label': dominant_label,
                'purity': purity,
                'feature_means': np.mean(cluster_features, axis=0),
                'feature_stds': np.std(cluster_features, axis=0),
                'true_label_distribution': true_label_counts
            }
        
        return characteristics
    
    def _analyze_confusion_matrix(self, true_labels, pred_labels, label_names=None):
        """
        Analyze confusion between true and predicted labels by mapping clusters to most likely labels
        """
        valid_mask = pred_labels != -1
        if np.sum(valid_mask) < 2:
            return {}
        
        true_clean = np.array(true_labels)[valid_mask]
        pred_clean = np.array(pred_labels)[valid_mask]

        # Map each cluster to its majority true label
        cluster_to_label = {}
        cluster_accuracy = {}

        for cluster in np.unique(pred_clean):
            mask = pred_clean == cluster
            if np.sum(mask) == 0:
                continue
            
            cluster_true_labels = true_clean[mask]
            unique_labels, counts = np.unique(cluster_true_labels, return_counts=True)
            majority_idx = np.argmax(counts)
            majority_label = unique_labels[majority_idx]
            
            cluster_to_label[cluster] = majority_label
            cluster_accuracy[cluster] = counts[majority_idx] / np.sum(mask)

        # Map predicted cluster IDs to their majority labels
        pred_mapped = np.array([cluster_to_label.get(c, "unassigned") for c in pred_clean])

        # Build confusion matrix using human-readable labels
        all_labels = np.unique(np.concatenate((true_clean, pred_mapped)))
        cm = confusion_matrix(true_clean, pred_mapped, labels=all_labels)

        cm_df = pd.DataFrame(
            cm,
            index=[f"True: {lbl}" for lbl in label_names],
            columns=[f"Pred: {lbl}" for lbl in label_names]
        )

        # Print summary
        print("\nCluster-to-label mapping and accuracies:")
        for c, lbl in cluster_to_label.items():
            acc = cluster_accuracy[c]
            print(f"  Cluster {c} → {lbl}  (accuracy: {acc:.3f})")

        print("\nConfusion Matrix (rows = true labels, cols = predicted labels):")
        print(cm_df)

        return {
            'confusion_matrix': cm,
            'labels': label_names,
            'n_true_clusters': len(np.unique(true_clean)),
            'n_pred_clusters': len(np.unique(pred_clean))
        }
    
    def create_report(self, analysis, output_path=None):
        """Create comprehensive visualization report"""
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(3, 3, figure=fig)
        
        # Basic metrics
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_basic_metrics(ax1, analysis['basic_metrics'])
        
        # Internal metrics
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_internal_metrics(ax2, analysis['internal_metrics'])
        
        # Cluster sizes
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_cluster_sizes(ax3, analysis['cluster_characteristics'])
        
        # Cluster purity
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_cluster_purity(ax4, analysis['cluster_characteristics'])
        
        # Confusion matrix
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_confusion_matrix(ax5, analysis['confusion_analysis'])

        # Metrics comparison table
        ax7 = fig.add_subplot(gs[2, :])
        self._plot_metrics_table(ax7, analysis)
        
        plt.tight_layout()
        plt.suptitle(f'Cluster Analysis: {analysis["dataset"]}', fontsize=16, y=0.98)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"analysis saved: {output_path}")
        
        plt.show()
        
        # Print detailed results
        self._print_detailed_results(analysis)
    
    def _plot_basic_metrics(self, ax, metrics):
        """Plot basic metrics as radar chart"""
        if not metrics:
            ax.text(0.5, 0.5, 'No basic metrics', ha='center', va='center')
            ax.set_title('Basic Metrics')
            return
        
        categories = ['ARI', 'NMI', 'Purity', 'Homogeneity', 'Completeness']
        values = [metrics.get('ari', 0), metrics.get('nmi', 0), metrics.get('purity', 0),
                 metrics.get('homogeneity', 0), metrics.get('completeness', 0)]
        
        # Complete the circle
        values += values[:1]
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, values, 'o-', linewidth=2, label='Scores')
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Basic Clustering Metrics', size=14, y=1.1)
        ax.grid(True)
    
    def _plot_internal_metrics(self, ax, metrics):
        """Plot internal metrics as bar chart"""
        if not metrics:
            ax.text(0.5, 0.5, 'No internal metrics', ha='center', va='center')
            ax.set_title('Internal Metrics')
            return
        
        names = list(metrics.keys())
        values = list(metrics.values())
        
        # Normalize for better visualization
        if 'calinski_harabasz' in metrics:
            idx = names.index('calinski_harabasz')
            values[idx] = min(1.0, values[idx] / 1000)
        
        bars = ax.bar(names, values, color='lightgreen', alpha=0.7)
        ax.set_title('Internal Validation Metrics')
        ax.set_ylabel('Score')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    def _plot_cluster_sizes(self, ax, characteristics):
        """Plot cluster size distribution"""
        if not characteristics:
            ax.text(0.5, 0.5, 'No clusters', ha='center', va='center')
            ax.set_title('Cluster Sizes')
            return
        
        cluster_ids = list(characteristics.keys())
        sizes = [chars['size'] for chars in characteristics.values()]
        
        bars = ax.bar(cluster_ids, sizes, color='lightcoral', alpha=0.7)
        ax.set_title('Cluster Size Distribution')
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Number of Points')
        
        for bar, size in zip(bars, sizes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{size}', ha='center', va='bottom', fontsize=9)
    
    def _plot_cluster_purity(self, ax, characteristics):
        """Plot cluster purity"""
        if not characteristics:
            ax.text(0.5, 0.5, 'No clusters', ha='center', va='center')
            ax.set_title('Cluster Purity')
            return
        
        cluster_ids = list(characteristics.keys())
        purities = [chars['purity'] for chars in characteristics.values()]
        dominant_labels = [str(chars['dominant_true_label']) for chars in characteristics.values()]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_ids)))
        bars = ax.bar(cluster_ids, purities, color=colors, alpha=0.7)
        ax.set_title('Cluster Purity by Dominant Label')
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Purity')
        ax.set_ylim(0, 1)
        
        for bar, purity, label in zip(bars, purities, dominant_labels):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{purity:.2f}', ha='center', va='bottom', fontsize=8)
            # Add dominant label below
            ax.text(bar.get_x() + bar.get_width()/2, -0.1, label[:10],
                   ha='center', va='top', fontsize=7, rotation=45)
    
    def _plot_confusion_matrix(self, ax, confusion_analysis):
        """Plot confusion matrix"""
        if not confusion_analysis or 'confusion_matrix' not in confusion_analysis:
            ax.text(0.5, 0.5, 'No confusion matrix', ha='center', va='center')
            ax.set_title('Confusion Matrix')
            return
        
        cm = confusion_analysis['confusion_matrix']
        im = ax.imshow(cm, cmap='Blues', aspect='auto')
        ax.set_title('Confusion Matrix\n(True vs Predicted)')
        ax.set_xlabel('Predicted Cluster')
        ax.set_ylabel('True Label')
        
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', 
                       color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=8)
        
        plt.colorbar(im, ax=ax, shrink=0.6)
    
    def _plot_metrics_table(self, ax, analysis):
        ax.axis('off')
        
        table_data = []
        
        # Basic metrics
        if analysis['basic_metrics']:
            table_data.append(["BASIC METRICS", ""])
            for metric, value in analysis['basic_metrics'].items():
                if isinstance(value, (int, float)):
                    table_data.append([f"  {metric}", f"{value:.4f}"])
        
        # Internal metrics
        if analysis['internal_metrics']:
            table_data.append(["", ""])
            table_data.append(["INTERNAL METRICS", ""])
            for metric, value in analysis['internal_metrics'].items():
                if isinstance(value, (int, float)):
                    table_data.append([f"  {metric}", f"{value:.4f}"])
        
        # Cluster info
        table_data.append(["", ""])
        table_data.append(["Cluster Info", ""])
        table_data.append([f"  Clusters found", f"{len(analysis['cluster_characteristics'])}"])
        if analysis['basic_metrics']:
            table_data.append([f"  Noise ratio", f"{analysis['basic_metrics'].get('noise_ratio', 0):.4f}"])
        
        # Create table
        table = ax.table(cellText=table_data, 
                        cellLoc='left',
                        loc='center',
                        bbox=[0.1, 0.1, 0.8, 0.8])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Style header rows
        for i, row in enumerate(table_data):
            if row[1] == "":  # Header row
                for j in range(2):
                    table[(i, j)].set_facecolor('lightgray')
                    table[(i, j)].set_text_props(weight='bold')
    
    def _print_detailed_results(self, analysis):
        print(f"Comprehensive Cluster Analysis: {analysis['dataset']}")
        
        if analysis['basic_metrics']:
            print("\n Basic Metrics:")
            for metric, value in analysis['basic_metrics'].items():
                if isinstance(value, (int, float)):
                    print(f"  {metric:>20}: {value:.4f}")
        
        if analysis['internal_metrics']:
            print("\nInternal Metrics:")
            for metric, value in analysis['internal_metrics'].items():
                if isinstance(value, (int, float)):
                    print(f"  {metric:>20}: {value:.4f}")
        
        print(f"\n Cluster Characteristics:")
        for cluster_id, chars in analysis['cluster_characteristics'].items():
            print(f"  Cluster {cluster_id:2d}: {chars['size']:3d} points, "
                  f"purity: {chars['purity']:.3f}, dominant: {chars['dominant_true_label']}")