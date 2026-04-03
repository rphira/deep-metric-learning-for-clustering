'''This is a ready to run version, all combined into a single file of the metric learning pipeline with triplet loss.'''
import os
import sys
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
from pathlib import Path
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# CONFIGURATION SETUP
# ============================================================================

def get_config():
    """
    Get configuration from environment variables with sensible defaults.
    
    Environment variables to set:
    - ML4NP_DATA_ROOT: Path to your data directory (required)
    - ML4NP_OUTPUT_DIR: Path for output (defaults to './results')
    """
    
    # Get data root from environment (user must set this)
    data_root = os.environ.get("ML4NP_DATA_ROOT")
    if data_root is None:
        raise ValueError(
            "ML4NP_DATA_ROOT environment variable is not set.\n"
            "Please set it to the path containing your data files:\n"
            "  export ML4NP_DATA_ROOT=/path/to/your/data\n"
            "Or in your Python code:\n"
            "  import os; os.environ['ML4NP_DATA_ROOT'] = '/path/to/your/data'"
        )
    
    # Get output directory (defaults to ./results if not set)
    output_root = os.environ.get("ML4NP_OUTPUT_DIR", "./results")
    
    # Create Path objects
    data_root_path = Path(data_root)
    output_root_path = Path(output_root)
    
    # Ensure output directory exists
    output_root_path.mkdir(parents=True, exist_ok=True)
    
    # Define paths for specific files (adjust these based on your actual file structure)
    config = {
        'data_root': data_root_path,
        'output_root': output_root_path,
        'ladder_features': data_root_path / "complete_feature_sets/all_ladders_tsfresh.csv",
        # Add other file paths as needed
    }
    
    return config

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class TripletDSCModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.3):
        super(TripletDSCModel, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, latent_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        return latent  # No decoder since we're not using reconstruction

# ============================================================================
# CLUSTER ANALYSIS
# ============================================================================

class ClusterAnalyze:
    '''A class to analyze clustering results with multiple metrics and visualizations.'''
    def __init__(self):
        self.analysis_results = {}
    
    def analyze_clustering(self, features, le, true_labels, pred_labels, latent_embeddings, feature_names, dataset_name):
        """clustering analysis with multiple metrics"""
        
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
        """Calculate basic clustering metrics"""
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
        """Calculate internal validation metrics"""
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
        Analyze confusion between true and predicted labels, 
        mapping clusters to their most likely biological labels
        and computing per-cluster accuracy.
        """

        from sklearn.metrics import confusion_matrix
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

        # summary
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
    
    def create_comprehensive_report(self, analysis, output_path=None):
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

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_with_triplet(
    features,
    labels,
    feature_names,
    output_dir=None,
    n_clusters=None,
    hidden_dim=128,
    latent_dim=64,
    num_epochs=500,
    lr=1e-3,
    margin=1.0,
    l1_weight=1e-4,
    batch_size=512,
    triplet_weight=1.0,
    ari_threshold=0.85
):
    
    '''This is our main training function with triplet loss and early stopping based on validation ARI.'''
    le = LabelEncoder()
    true_labels = le.fit_transform(labels)

    # 60/20/20 split: Train / Validation / Test
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, true_labels, test_size=0.2, stratify=true_labels, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
    )
    rng = np.random.RandomState(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=device)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32, device=device)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32, device=device)
    
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(42)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)

    model = TripletDSCModel(features.shape[1], hidden_dim, latent_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    triplet_loss_fn = nn.TripletMarginLoss(margin=margin)

    # Enhanced history tracking
    history = {
        "epoch": [], "train_loss": [], "val_ari": [], "val_nmi": [], 
        "train_ari": [], "train_nmi": [], "test_ari": [], "test_nmi": [],
        "train_silhouette": [], "val_silhouette": []
    }

    patience = 150
    best_val_ari = -np.inf
    patience_counter = 0
    best_model_state = None
    best_feature_scores = None

    print(f"Data splits - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    print(f"Training on device: {device}")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for xb, yb in loader:
            optimizer.zero_grad()
            latent = model(xb) 
            triplet_generator = torch.Generator(device=latent.device)
            triplet_generator.manual_seed(42)

            # Triplet mining here.
            anchor, positive, negative = [], [], []
            for i in range(xb.size(0)):
                same = (yb == yb[i]).nonzero(as_tuple=True)[0]
                diff = (yb != yb[i]).nonzero(as_tuple=True)[0]
                if len(same) > 1 and len(diff) > 0:
                    pos_idx = same[same != i][rng.randint(0, len(same) - 1)]
                    neg_idx = diff[rng.randint(0, len(diff))]
                    anchor.append(latent[i])
                    positive.append(latent[pos_idx])
                    negative.append(latent[neg_idx])

            if anchor:
                anchor = torch.stack(anchor)
                positive = torch.stack(positive)
                negative = torch.stack(negative)
                triplet_loss = triplet_loss_fn(anchor, positive, negative)
            else:
                triplet_loss = torch.tensor(0.0, device=device)

            loss = triplet_weight * triplet_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # early stopping
        model.eval()
        with torch.no_grad():
            # Training set evaluation
            latent_train = model(X_train_tensor)
            latent_train_norm = F.normalize(latent_train, p=2, dim=1).cpu().numpy()
            train_preds = hdbscan.HDBSCAN(min_cluster_size=10).fit_predict(latent_train_norm)
            train_ari = adjusted_rand_score(y_train, train_preds) if len(np.unique(train_preds)) > 1 else 0
            train_nmi = normalized_mutual_info_score(y_train, train_preds) if len(np.unique(train_preds)) > 1 else 0
            train_sil = silhouette_score(latent_train_norm, train_preds) if len(np.unique(train_preds)) > 1 else -1

            # Validation set evaluation
            latent_val = model(X_val_tensor)
            latent_val_norm = F.normalize(latent_val, p=2, dim=1).cpu().numpy()
            val_preds = hdbscan.HDBSCAN(min_cluster_size=10).fit_predict(latent_val_norm)
            val_ari = adjusted_rand_score(y_val, val_preds) if len(np.unique(val_preds)) > 1 else 0
            val_nmi = normalized_mutual_info_score(y_val, val_preds) if len(np.unique(val_preds)) > 1 else 0
            val_sil = silhouette_score(latent_val_norm, val_preds) if len(np.unique(val_preds)) > 1 else -1

        history["epoch"].append(epoch)
        history["train_loss"].append(total_loss)
        history["val_ari"].append(val_ari)
        history["val_nmi"].append(val_nmi)
        history["train_ari"].append(train_ari)
        history["train_nmi"].append(train_nmi)
        history["train_silhouette"].append(train_sil)
        history["val_silhouette"].append(val_sil)

        print(f"[Epoch {epoch:03d}] Loss: {total_loss:.4f} | "
              f"Train ARI: {train_ari:.4f} | Val ARI: {val_ari:.4f} | "
              f"Train Sil: {train_sil:.4f} | Val Sil: {val_sil:.4f}")
        
        # early stopping on validation performance
        if val_ari > best_val_ari:
            best_val_ari = val_ari
            patience_counter = 0
            best_model_state = model.state_dict()
            with torch.no_grad():
                input_weights = model.encoder[0].weight.detach().cpu().numpy()
                best_feature_scores = np.sum(np.abs(input_weights), axis=0)
        else:
            patience_counter += 1

        if best_val_ari >= ari_threshold or patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} — Best Val ARI: {best_val_ari:.4f}")
            break

    # load best model from validation performance
    model.load_state_dict(best_model_state)

    # final evaluation on test set
    model.eval()
    with torch.no_grad():
        latent_test = model(X_test_tensor)
        latent_test_norm = F.normalize(latent_test, p=2, dim=1).cpu().numpy()
        preds_test = hdbscan.HDBSCAN(min_cluster_size=10).fit_predict(latent_test_norm)
        
        test_ari = adjusted_rand_score(y_test, preds_test)
        test_nmi = normalized_mutual_info_score(y_test, preds_test)

    history["test_ari"] = [test_ari] * len(history["epoch"])
    history["test_nmi"] = [test_nmi] * len(history["epoch"])

    print("Final Results:")
    print(f"Final Test ARI: {test_ari:.4f}")
    print(f"Final Test NMI: {test_nmi:.4f}")
    for i, label in enumerate(le.classes_):
        print(f"{i} → {label}")

    # analysis post clustering
    analyzer = ClusterAnalyze()
    
    test_analysis = analyzer.analyze_clustering(
        features=X_test_scaled, le=le, true_labels=y_test, pred_labels=preds_test,
        latent_embeddings=latent_test_norm, feature_names=feature_names,
        dataset_name="Test_Set"
    )

    if output_dir:
        test_report_path = os.path.join(output_dir, "test_set_analysis.png")
        analyzer.create_comprehensive_report(test_analysis, test_report_path)

    # umaps
    create_umap_plots(latent_test_norm, y_test, preds_test, le, output_dir)

    # Feature importance
    importance_df = calculate_feature_importance(model, features, feature_names, device, best_feature_scores)

    return model, history, preds_test, importance_df, {'test': test_analysis}

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_umap_plots(latent_test, y_test_encoded, preds_test, le, output_dir=None):
    """Create UMAP visualizations - True vs Predicted only"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 12))
    
    # UMAP transformation
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(latent_test)
    
    # Get true label names
    true_labels = le.inverse_transform(y_test_encoded)
    
    # Plot 1: True Labels
    for label in le.classes_:
        mask = true_labels == label
        ax1.scatter(embedding[mask, 0], embedding[mask, 1], label=label, s=50, alpha=1.0)
    ax1.set_title("True Labels")
    ax1.legend(
    bbox_to_anchor=(1.05, 1),  # Move legend outside the axes
    loc='upper left',          # Anchor it to the top-left of the legend box
    fontsize=8,                # Optional: smaller font if many labels
    frameon=False              # Optional: remove box
    )
    
    cmap = plt.get_cmap('tab20', len(np.unique(preds_test)))

    # Plot each cluster separately to get a legend entry
    for cluster in np.unique(preds_test):
        ax2.scatter(
            embedding[preds_test == cluster, 0],
            embedding[preds_test == cluster, 1],
            color=cmap(cluster),
            label=f"Cluster {cluster}",
            s=30,
            alpha=0.9
        )

    ax2.set_title("Predicted Clusters")
    ax2.legend(
        bbox_to_anchor=(1.05, 1),  # place legend outside the plot
        loc='upper left',
        title="Clusters",
        fontsize='small'
    )
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "umap_comparison.png"), dpi=300, bbox_inches='tight')
    
    plt.show()

def calculate_feature_importance(model, features, feature_names, device, best_feature_scores):
    """Calculate feature importance using multiple methods this is old now. Refer to vif.py for variable inclusion features"""
        
    # Method 1: Model weights
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "score_model_weights": best_feature_scores
    })
    
    # Method 2: GLMNet-based importance
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
    
    # Combined score
    importance_df["score_combined"] = importance_df["score_glmnet"]
    
    return importance_df.sort_values(by="score_combined", ascending=False)

def plot_training_history(history, output_dir=None):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history["epoch"], history["train_loss"], 'b-', linewidth=2)
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)
    
    # ARI comparison
    axes[0, 1].plot(history["epoch"], history["train_ari"], 'g-', label="Train ARI", linewidth=2)
    axes[0, 1].plot(history["epoch"], history["val_ari"], 'r-', label="Val ARI", linewidth=2)
    axes[0, 1].plot(history["epoch"], history["test_ari"], 'b-', label="Test ARI", linewidth=2)
    axes[0, 1].set_title("ARI Scores")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("ARI")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Silhouette scores
    axes[1, 0].plot(history["epoch"], history["train_silhouette"], 'g-', label="Train Silhouette", linewidth=2)
    axes[1, 0].plot(history["epoch"], history["val_silhouette"], 'r-', label="Val Silhouette", linewidth=2)
    axes[1, 0].set_title("Silhouette Scores")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Silhouette")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Final scores table
    axes[1, 1].axis('off')
    final_text = "Final Results:\n\n"
    final_text += f"Best Val ARI: {max(history['val_ari']):.4f}\n"
    final_text += f"Final Test ARI: {history['test_ari'][0]:.4f}\n"
    final_text += f"Final Test NMI: {history['test_nmi'][0]:.4f}\n"
    final_text += f"Training Epochs: {len(history['epoch'])}"
    
    axes[1, 1].text(0.1, 0.9, final_text, transform=axes[1, 1].transAxes, fontfamily='monospace',
                   verticalalignment='top', fontsize=12, linespacing=1.5)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "training_history.png"), dpi=300, bbox_inches='tight')
        print(f"Training history saved")
    
    plt.show()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_results(output_dir, clusters, importance_df, model, history):
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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True) 
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def remove_outliers_by_label(df, label_column, contamination=0.3):
    """Remove outliers using IsolationForest for each label separately"""
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

def cluster_downsample(df, label_column, samples_per_class=6500, random_state=42):
    sampled_dfs = []
    for label in df[label_column].unique():
        label_df = df[df[label_column] == label]
        n_samples = min(samples_per_class, len(label_df))
        sampled = label_df.sample(n=n_samples, random_state=random_state)
        sampled_dfs.append(sampled)
    return pd.concat(sampled_dfs, ignore_index=True)

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    # Set random seeds for reproducibility
    set_seed(42)
    
    # Get configuration
    config = get_config()
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set output directory for this run
    output_dir = config['output_root'] / f"ladder_1_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Data root: {config['data_root']}")
    print(f"Output directory: {output_dir}")
    print(f"Loading data from: {config['ladder_features']}")
    
    # Load data using configurable path
    try:
        k8_k12_k16 = pd.read_csv(config['ladder_features'])
    except FileNotFoundError as e:
        print(f"Error: Could not find data file at {config['ladder_features']}")
        print(f"Please make sure ML4NP_DATA_ROOT is set correctly and the file exists.")
        print(f"Current ML4NP_DATA_ROOT: {config['data_root']}")
        print(f"Expected file: {config['ladder_features']}")
        sys.exit(1)
    
    # Rename label column if needed
    k8_k12_k16 = k8_k12_k16.rename(columns={'label': 'final_label'}).replace([float('inf'), float('-inf')], float('nan'))

    # Filter for PL3 labels
    k8_k12_k16 = k8_k12_k16[k8_k12_k16["final_label"].str[:3] == "PL3"]  # keep only PL3 labels
    k8_k12_k16 = k8_k12_k16[k8_k12_k16['final_label'].notna()]

    # Drop columns with high NaN percentage
    nan_percentage = k8_k12_k16.isna().mean() * 100
    k8_k12_k16 = k8_k12_k16.loc[:, nan_percentage <= 10]  # Drop columns where NaN percentage is greater than 50%
    k8_k12_k16 = k8_k12_k16.dropna()

    # Feature selection: remove columns with certain patterns
    cleaned_data = k8_k12_k16.loc[:, ~k8_k12_k16.columns.str.contains('c3|value__quantile|value__agg_linear_trend|large_standard_deviation|quantile|change_quantiles')]
    
    # Drop features with too many NaNs
    nan_threshold = 0.5 * len(cleaned_data)
    final_feature_df = cleaned_data.dropna(axis=1, thresh=nan_threshold)

    # Remove low-variance features
    numeric_features = final_feature_df.select_dtypes(include=['number'])
    selector = VarianceThreshold(threshold=0.01)
    reduced_features = selector.fit_transform(numeric_features)
    retained_columns = numeric_features.columns[selector.get_support()]
    final_feature_df = final_feature_df[retained_columns.tolist() + ['final_label']]  # Keep label

    # Optional: Remove outliers (commented out by default)
    # final_feature_df = remove_outliers_by_label(final_feature_df, label_column='final_label', contamination=0.3)
    
    cleaned_data = final_feature_df
    
    # Define columns to drop
    to_drop = ['EventId', 'idxstart', 'idxend', 'risetime', 'value__mean_abs_change', 'value__standard_deviation', 'value__linear_trend__attr_"intercept"', 'falltime', "I/Io", "Io", "Irms", "length", "value__maximum", "value__minimum", "log_length", "value__median", 'value__root_mean_square', 'value__standard_deviation', 'value__variance', 'Imean', 'isBL?','Isig']

    # Remove highly correlated features with 'value__mean' if present
    high_corr_features = []
    if 'value__mean' in cleaned_data.columns:
        print("Mean present - checking for highly correlated features")
        correlations = cleaned_data.drop(columns=['final_label']).corr()['value__mean'].abs()
        high_corr_features = correlations[correlations > 0.66].index.tolist()
        high_corr_features = [f for f in high_corr_features if f != 'value__mean']
        if high_corr_features:
            print(f"Found {len(high_corr_features)} highly correlated features with 'value__mean'")
            for feature in high_corr_features:
                print(f"  {feature}: {correlations[feature]:.3f}")
    
    columns_to_drop = ['value__mean'] + high_corr_features + to_drop
    cleaned_data = cleaned_data.drop(columns=[col for col in columns_to_drop if col in cleaned_data.columns])

    # Optional: Downsample using clustering (commented out by default)
    # cleaned_data = cluster_downsample(cleaned_data, 'final_label', clusters_per_class=6500)
    
    # Prepare features and labels
    X = cleaned_data.drop(columns=['final_label']).values
    y = cleaned_data['final_label'].values
    feature_names = cleaned_data.drop(columns="final_label").columns.tolist()

    n_clusters = len(np.unique(y)) 
    
    # Save training metadata
    training_info = {
        "training_row_indices": cleaned_data.index.tolist(),
        "num_rows": len(cleaned_data),
        "feature_list": feature_names,
        "num_features": len(feature_names),
        "data_source": str(config['ladder_features']),
        "output_dir": str(output_dir),
        "timestamp": timestamp
    }

    json_path = os.path.join(output_dir, f"training_metadata_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(training_info, f, indent=4)
    
    print(f"Training metadata saved to: {json_path}")
    print(f"Starting training with {len(cleaned_data)} samples and {len(feature_names)} features")
    
    # Train the model
    model, history, preds, importance_df, analysis_results = train_with_triplet(
        X,
        y,
        feature_names,
        output_dir=output_dir,
        n_clusters=n_clusters,
        hidden_dim=256,
        latent_dim=64,
        num_epochs=500,
        lr=1e-3,
        margin=1.0,
        l1_weight=1e-4,
        batch_size=512,
        triplet_weight=1.0,
        ari_threshold=0.90
    )

    # Plot and save results
    plot_training_history(history, output_dir=output_dir)
    save_results(output_dir, preds, importance_df, model, history)
    
    print(f"\n=== RUN COMPLETE ===")
    print(f"Results saved to: {output_dir}")
    print(f"To reproduce, set ML4NP_DATA_ROOT to: {config['data_root']}")
    print(f"And run the same code.")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Triplet Metric Learning for Nanopore Data")
    print("=" * 70)
    print("\nIMPORTANT: Before running, set the ML4NP_DATA_ROOT environment variable:")
    print("  export ML4NP_DATA_ROOT=/path/to/your/data/directory")
    print("\nOr in Python:")
    print("  import os")
    print("  os.environ['ML4NP_DATA_ROOT'] = '/path/to/your/data/directory'")
    print("=" * 70)
    
    main()