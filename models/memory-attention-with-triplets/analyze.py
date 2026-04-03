import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import ElasticNet
import umap
from collections import defaultdict
import hdbscan
from models import MemoryAttentionTripletModel
from trainer import MemoryReplayTrainer
from data_utils import DataPreprocessor
from metrics import MetricsCalculator
from visualization import Visualizer
from reports import ReportGenerator


class SequentialAnalysis:
    """
    Main analyzer for sequential analyte identification with memory and attention.
    """
    def __init__(self, base_output_dir):
        self.base_output_dir = base_output_dir
        os.makedirs(base_output_dir, exist_ok=True)
        self.results = {}

        # Manual label encoding for consistency
        self.global_label_mapping = {
            'AS5': 0,
            'AS6': 1,
            'AS7': 2,
            'AS8': 3,
            'AS9': 4,
            'AS10': 5
        }

        self.label_to_name = {v: k for k, v in self.global_label_mapping.items()}
        self.evolution_history = {
            'steps': [],
            'metrics': [],
            'feature_importance': []
        }

        self.unseen_dataset_path = '/home/rushang_phira/src/data/complete_feature_sets/all_ladders_PL1.csv'
        self.common_features = None
        
        # Initialize helper classes
        self.data_preprocessor = DataPreprocessor()
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = Visualizer(base_output_dir, self.get_protein_name)
        self.report_generator = ReportGenerator(base_output_dir, self.get_protein_name)

    def get_protein_name(self, encoded_label):
        """Convert encoded label to protein name"""
        return self.label_to_name.get(encoded_label, f"Unknown_{encoded_label}")
    
    def get_protein_names(self, encoded_labels):
        """Convert array of encoded labels to protein names"""
        return [self.get_protein_name(label) for label in encoded_labels]
    
    def load_sequential_datasets(self):
        """Load datasets in cumulative sequential order"""
        protein_files = [
            ('AS6', 'Mrunal_L2_AS6_tsfresh.csv'),
            ('AS7', 'Mrunal_L2_AS7_tsfresh.csv'),
            ('AS8', 'Mrunal_L2_AS8_tsfresh.csv'),
            ('AS9', 'Mrunal_L2_AS9_tsfresh.csv'),
            ('AS10', 'Mrunal_L2_AS10_tsfresh.csv')
        ]
        
        print("\nLoading all datasets to determine common features...")
        all_dataframes = []
        for protein_name, filename in protein_files:
            df = self.data_preprocessor.load_and_preprocess(
                f'/home/rushang_phira/src/data/classified/feature_set/{filename}'
            )
            all_dataframes.append(df)
            print(f"  {protein_name}: {df.shape[1]-1} features, {len(df)} samples")
        
        # Find common features
        common_features = set(all_dataframes[0].columns)
        for df in all_dataframes[1:]:
            common_features = common_features.intersection(set(df.columns))
        
        if 'final_label' not in common_features:
            common_features.add('final_label')
        
        common_features = sorted(list(common_features))
        print(f"\nCommon features across all datasets: {len(common_features)-1}")
        
        self.common_features = [f for f in sorted(list(common_features)) if f != 'final_label']
        
        # Align all datasets
        aligned_dataframes = []
        for df in all_dataframes:
            aligned_df = df[common_features].copy()
            aligned_dataframes.append(aligned_df)
        
        datasets = {}
        for i, (protein_name, _) in enumerate(protein_files, start=1):
            current_data = aligned_dataframes[i-1]
            protein_list = '+'.join([protein_files[j][0] for j in range(i)])
            datasets[f'step_{i}_{protein_list}'] = current_data.copy()
            
            print(f"\nStep {i}: Using {protein_name}")
            print(f"  Cumulative proteins: {protein_list}")
            print(f"  Total samples: {len(current_data)}")
        
        return datasets
    
    def run_sequential_analysis(self):
        """Main sequential analysis pipeline"""
        print("Sequential (actually cumulative) Addition With Memory and Attention")
        datasets = self.load_sequential_datasets()
        
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {device}")
        
        model = None
        trainer = None
        
        for step_idx, (dataset_name, data) in enumerate(datasets.items(), start=1):
            print(f"STEP {step_idx}: {dataset_name}")
        
            X, y, feature_names, le = self.data_preprocessor.prepare_dataset(data)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.25, stratify=y_train, random_state=42
            )
            
            # Initialize model on first step
            if model is None:
                model = MemoryAttentionTripletModel(
                    input_dim=X_train.shape[1],
                    hidden_dim=256,
                    latent_dim=64,
                    dropout=0.3
                ).to(device)
                trainer = MemoryReplayTrainer(model, device)
                print(f"initialized model with {X_train.shape[1]} features")
            
            # Train
            print(f"training on {len(np.unique(y_train))} proteins...")
            history = trainer.train_step(
                X_train, y_train, X_val, y_val,
                num_epochs=500,
                memory_replay_ratio=0.2,
                lr=1e-3 if step_idx == 1 else 1e-4
            )
            
            # Evaluate
            print(f"evaluating")
            evaluation = self.evaluate_step(
                model, X_train, y_train, X_test, y_test, feature_names, device
            )

            # Visualizations
            print(f"  Generating UMAP visualizations...")
            self.visualizer.visualize_umap_comparison(
                embeddings=evaluation['umap_data']['test_embeddings'],
                true_labels=evaluation['umap_data']['test_true_labels'],
                pred_labels=evaluation['umap_data']['test_pred_labels'],
                step_name=dataset_name,
                step_idx=step_idx
            )
            
            # Feature importance
            print(f"  Extracting feature importance...")
            feature_importance = model.get_feature_importance(
                X_train, y_train, feature_names, device
            )
            
            # Store results
            self.results[dataset_name] = {
                'step': step_idx,
                'history': history,
                'evaluation': evaluation,
                'feature_importance': feature_importance,
                'feature_names': feature_names,
                'n_proteins': len(np.unique(y_train))
            }
            
            # Update evolution tracking
            self.evolution_history['steps'].append(dataset_name)
            self.evolution_history['metrics'].append(evaluation)
            self.evolution_history['feature_importance'].append(feature_importance)
            
            # Generate visualizations
            self.visualizer.visualize_step(dataset_name, step_idx, self.results[dataset_name])
            print(f"  Generating GLMNet visualization...")
            self.visualizer.visualize_glmnet_importance(dataset_name, step_idx, self.results[dataset_name])
            
            # Save checkpoint
            self.save_checkpoint(dataset_name)
            
            print(f"  Step {step_idx} completed!")
            print(f"  Test ARI: {evaluation['test_ari']:.4f}")
        
        # Generate final reports
        self.report_generator.generate_final_report(self.results, self.evolution_history)

        # Unseen dataset test
        final_evaluation = self.test_on_unseen_data(model, device)
        self.results['final_unseen_test'] = {'evaluation': final_evaluation}
        
        return self.results
  
    def evaluate_step(self, model, X_train, y_train, X_test, y_test, feature_names, device):
        """evaluation for one step"""
        model.eval()
        
        # Prepare data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            train_embeddings = model(X_train_tensor).cpu().numpy()
            test_embeddings = model(X_test_tensor).cpu().numpy()
            
            train_embeddings_norm = train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True)
            test_embeddings_norm = test_embeddings / np.linalg.norm(test_embeddings, axis=1, keepdims=True)

        # GLMNet feature importance (we dont actually use it anymore. See vif.py for updated feature analysis)
        glmnet_importance = self.calculate_glmnet_importance(
            model, X_train_scaled, feature_names, device
        )
        
        # Clustering
        n_clusters = len(np.unique(y_train))
        train_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        train_embeddings_2d = train_reducer.fit_transform(train_embeddings_norm)

        test_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        test_embeddings_2d = test_reducer.fit_transform(test_embeddings_norm)

        # I also test with KMeans but I leave HDBSCAN here
        kmeans_test = hdbscan.HDBSCAN(min_cluster_size=5)
        # KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
        test_preds = kmeans_test.fit_predict(test_embeddings_2d)
        
        kmeans_train = hdbscan.HDBSCAN(min_cluster_size=5)
        # KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
        train_preds = kmeans_train.fit_predict(train_embeddings_2d)
        
        # Comprehensive metrics
        metrics = self.metrics_calculator.calculate_comprehensive_metrics(
            test_embeddings_2d, y_test, test_preds, n_clusters
        )
        
        train_metrics = self.metrics_calculator.calculate_comprehensive_metrics(
            train_embeddings_2d, y_train, train_preds, n_clusters
        )
        
        # Per-analyte metrics
        per_analyte = self.metrics_calculator.calculate_per_analyte_metrics(
            test_embeddings_norm, y_test, test_preds, n_clusters, self.get_protein_name
        )
        
        # UMAP data
        umap_data = {
            'train_embeddings': train_embeddings,
            'train_true_labels': y_train,
            'train_pred_labels': train_preds,
            'test_embeddings': test_embeddings,
            'test_true_labels': y_test,
            'test_pred_labels': test_preds
        }
        
        return {
            'train_metrics': train_metrics,
            'test_metrics': metrics,
            'test_ari': metrics['ari'],
            'test_nmi': metrics['nmi'],
            'test_silhouette': metrics['silhouette'],
            'train_ari': train_metrics['ari'],
            'train_nmi': train_metrics['nmi'],
            'train_silhouette': train_metrics['silhouette'],
            'n_clusters': n_clusters,
            'per_analyte': per_analyte,
            'umap_data': umap_data,
            'glmnet_importance': glmnet_importance
        }
        
    def calculate_glmnet_importance(self, model, features, feature_names, device):
        """Calculate feature importance using GLMNet/ElasticNet"""
        model.eval()
        with torch.no_grad():
            features_tensor = torch.tensor(features, dtype=torch.float32, device=device)
            latent_full = model(features_tensor).cpu().numpy()

        coef_list = []
        for i in range(latent_full.shape[1]):
            try:
                m = ElasticNet(alpha=1.0, max_iter=100, random_state=42)
                m.fit(features, latent_full[:, i])
                coef_list.append(np.abs(m.coef_))
            except:
                coef_list.append(np.zeros(len(feature_names)))
        
        if coef_list:
            glmnet_importance_scores = np.mean(coef_list, axis=0)
        else:
            glmnet_importance_scores = np.zeros(len(feature_names))
        
        glmnet_importance = {
            feature_names[i]: float(glmnet_importance_scores[i])
            for i in range(len(feature_names))
        }
        
        return glmnet_importance

    def test_on_unseen_data(self, model, device):
        """Test on completely unseen dataset"""
        print("Generalization test: Unseen Data")
        
        print(f"  Loading unseen dataset: {self.unseen_dataset_path}")
        unseen_data = self.data_preprocessor.load_and_preprocess(self.unseen_dataset_path)
        
        X_unseen, y_unseen, feature_names, le = self.data_preprocessor.prepare_dataset(unseen_data)
        
        X_unseen_aligned = DataPreprocessor.align_features(X_unseen, feature_names, self.common_features)
        
        print(f"  Unseen dataset: {len(X_unseen_aligned)} samples")
        print(f"  Proteins: {len(np.unique(y_unseen))} classes")
        
        evaluation = self.evaluate_step_simple(model, X_unseen_aligned, y_unseen, 
                                               X_unseen_aligned, y_unseen, feature_names, device)
        
        self.visualizer.visualize_final_unseen_test(evaluation)
        
        return evaluation

    def evaluate_step_simple(self, model, X_train, y_train, X_test, y_test, feature_names, device):
        """evaluation without GLMNet"""
        model.eval()
        n_clusters = len(np.unique(y_train))

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            train_embeddings = model(X_train_tensor).cpu().numpy()
            test_embeddings = model(X_test_tensor).cpu().numpy()
            
            train_embeddings_norm = train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True)
            test_embeddings_norm = test_embeddings / np.linalg.norm(test_embeddings, axis=1, keepdims=True)
        
        train_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        train_embeddings_2d = train_reducer.fit_transform(train_embeddings_norm)
        
        test_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        test_embeddings_2d = test_reducer.fit_transform(test_embeddings_norm)
        
        kmeans_test = hdbscan.HDBSCAN(min_cluster_size=5)
        test_preds = kmeans_test.fit_predict(test_embeddings_2d)
        
        kmeans_train = hdbscan.HDBSCAN(min_cluster_size=5)
        train_preds = kmeans_train.fit_predict(train_embeddings_2d)
        
        metrics = self.metrics_calculator.calculate_comprehensive_metrics(
            test_embeddings_2d, y_test, test_preds, n_clusters
        )
        train_metrics = self.metrics_calculator.calculate_comprehensive_metrics(
            train_embeddings_2d, y_train, train_preds, n_clusters
        )
        
        per_analyte = self.metrics_calculator.calculate_per_analyte_metrics(
            test_embeddings_2d, y_test, test_preds, n_clusters, self.get_protein_name
        )
        
        umap_data = {
            'train_embeddings': train_embeddings_2d,
            'train_true_labels': y_train,
            'train_pred_labels': train_preds,
            'test_embeddings': test_embeddings_2d,
            'test_true_labels': y_test,
            'test_pred_labels': test_preds
        }
        
        return {
            'train_metrics': train_metrics,
            'test_metrics': metrics,
            'test_ari': metrics['ari'],
            'test_nmi': metrics['nmi'],
            'test_silhouette': metrics['silhouette'],
            'train_ari': train_metrics['ari'],
            'train_nmi': train_metrics['nmi'],
            'train_silhouette': train_metrics['silhouette'],
            'n_clusters': n_clusters,
            'per_analyte': per_analyte,
            'umap_data': umap_data,
            'glmnet_importance': {}
        }

    def save_checkpoint(self, dataset_name):
        """Save checkpoint with current results"""
        checkpoint = {
            'dataset': dataset_name,
            'results': {
                k: {
                    'step': v['step'],
                    'test_ari': v['evaluation']['test_ari'],
                    'test_nmi': v['evaluation']['test_nmi'],
                    'n_proteins': v['n_proteins']
                }
                for k, v in self.results.items()
            },
            'evolution_history': self.evolution_history
        }
        
        checkpoint_path = os.path.join(self.base_output_dir, f'checkpoint_{dataset_name}.json')
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
