import os
import numpy as np
import pandas as pd
import torch
import datetime
from matplotlib import gridspec
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors
import umap
import matplotlib.cm as cm
from matplotlib import cm as mplt_cm
from matplotlib.colors import Normalize
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import random
from scipy import stats
from scipy.spatial.distance import cdist, pdist
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings('ignore')

class MemoryAttentionTripletModel(nn.Module):
    """
    Modification on my initial triplet-loss network with addition of:
    1. Feature attention for identifying important (tsfresh) features
    2. Memory bank
    """

    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.3):
        super(MemoryAttentionTripletModel, self).__init__()
        
        '''Attention mechanism'''
        self.feature_attention = nn.Sequential(
            nn.Linear(input_dim, max(32, input_dim // 4)),  # Bottleneck
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(max(32, input_dim // 4), input_dim),
            nn.Sigmoid()
        )
        self.attention_l1_lambda = 1e-4

        # Encoder architecture
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
        
        '''Memory bank mechanism (Similar to iCARL) but just random sampling instead'''
        self.protein_memory = {}  # {protein_id: [embeddings]}
        self.memory_size_per_protein = 50  # Store 50 samples per
        self.attention_history = {}  # Track attention patterns per

    def forward(self, x, return_attention=False):
        """
        Forward pass with attention-weighted features. Part of feature selection since attention weights can identify important features
        
        Args:
            x: Input features [batch_size, input_dim]
            return_attention: If True, also return attention weights
        
        Returns:
            embeddings: Latent representations [batch_size, latent_dim]
            attention_weights (optional): Feature importance [batch_size, input_dim]
        """

        attention_weights = self.feature_attention(x)  # [batch, input_dim]
        attended_features = x * attention_weights
        embeddings = self.encoder(attended_features) # Encode
        
        if return_attention:
            return embeddings, attention_weights
        return embeddings

    def update_memory(self, embeddings, labels, attention_weights=None):
        """Update memory bank, store as tensors with gradient capability"""
        rng = np.random.RandomState(42) 
        embeddings_np = embeddings.detach().cpu().numpy() # Unused right now
        labels_np = labels.cpu().numpy()
        
        for protein_id in np.unique(labels_np):
            protein_mask = labels_np == protein_id
            protein_embeddings = embeddings[protein_mask]
            
            if protein_id not in self.protein_memory:
                self.protein_memory[protein_id] = []
            
            # Add tensor samples (detached but with grad capability)
            for i, emb in enumerate(protein_embeddings):
                if len(self.protein_memory[protein_id]) < self.memory_size_per_protein:

                    # Store detached clone but ensure it's on same device and requires grad
                    memory_tensor = emb.detach().clone().requires_grad_(True)
                    self.protein_memory[protein_id].append(memory_tensor)
                else:
                    idx = rng.randint(0, len(self.protein_memory[protein_id]))
                    memory_tensor = emb.detach().clone().requires_grad_(True)
                    self.protein_memory[protein_id][idx] = memory_tensor

    def sample_from_memory(self, n_triplets, device):
        """Sample triplets and return tensors with gradient computation enabled. This is sampling from memory, not current PTM"""
        rng = np.random.RandomState(42) 
        if len(self.protein_memory) < 2: # need at least 2 PTMs to sample triplets
            return None, None, None
        
        protein_ids = list(self.protein_memory.keys())
        anchors, positives, negatives = [], [], []
        
        for _ in range(n_triplets):
            anchor_protein = rng.choice(protein_ids)
            anchor_samples = self.protein_memory[anchor_protein]
            
            if len(anchor_samples) < 2:
                continue
            
            anchor_idx, pos_idx = rng.choice(len(anchor_samples), 2, replace=False)
            anchor = anchor_samples[anchor_idx]
            positive = anchor_samples[pos_idx]
            
            neg_proteins = [p for p in protein_ids if p != anchor_protein]
            if neg_proteins:
                neg_protein = rng.choice(neg_proteins)
                neg_samples = self.protein_memory[neg_protein]
                neg_idx =  rng.randint(0, len(neg_samples))
                negative = neg_samples[neg_idx]
                
                anchors.append(anchor)
                positives.append(positive)
                negatives.append(negative)
        
        if anchors:
            anchors_tensor = torch.stack(anchors).to(device)
            positives_tensor = torch.stack(positives).to(device)
            negatives_tensor = torch.stack(negatives).to(device)
            
            anchors_tensor.requires_grad_(True)
            positives_tensor.requires_grad_(True)
            negatives_tensor.requires_grad_(True)
            
            return anchors_tensor, positives_tensor, negatives_tensor
        return None, None, None

    def get_feature_importance(self, X, y, feature_names, device):
        """
        Extract feature importance using attention weights.
        
        Args:
            X: Input features [n_samples, n_features]
            y: Labels [n_samples]
            feature_names: List of feature names
            device: torch device
        
        Returns:
            Dict with overall and per-protein feature importance
        """
        self.eval()
        importance_dict = {
            'overall': {},
            'per_protein': {}
        }
        
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
            
            # Get attention weights for all samples
            _, attention_weights = self.forward(X_tensor, return_attention=True)
            attention_np = attention_weights.cpu().numpy()
            
            # Overall importance (average across all samples)
            overall_importance = attention_np.mean(axis=0)
            importance_dict['overall'] = {
                feature_names[i]: float(overall_importance[i])
                for i in range(len(feature_names))
            }
            
            # Per-PTM importance
            for protein_id in np.unique(y):
                protein_mask = y == protein_id
                protein_attention = attention_np[protein_mask]
                mean_attention = protein_attention.mean(axis=0)
                
                protein_name = f'protein_{int(protein_id)}'
                importance_dict['per_protein'][protein_name] = {
                    feature_names[i]: float(mean_attention[i])
                    for i in range(len(feature_names))
                }
        
        return importance_dict

class MemoryReplayTrainer:
    """
    Trainer that implements memory replay to prevent catastrophic forgetting.
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def train_step(self, X_train, y_train, X_val, y_val, 
                   num_epochs=100, memory_replay_ratio=0.4, lr=1e-3):
        """
        Train one sequential step with memory replay.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            num_epochs: Number of training epochs
            memory_replay_ratio: Weight for memory replay loss (0-1)
            lr: Learning rate
        
        Returns:
            training_history: Dict with training metrics
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32, device=self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=self.device)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32, device=self.device)
        
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(42)
        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        loader = DataLoader(dataset, batch_size=512, shuffle=True, worker_init_fn=seed_worker, generator=g)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        triplet_loss_fn = nn.TripletMarginLoss(margin=1.0)
        
        # tracking
        best_val_ari = -1
        patience = 50
        patience_counter = 0
        best_model_state = None
        history = {
            'train_loss': [],
            'current_loss': [],
            'memory_loss': [],
            'val_ari': [],
            'val_nmi': []
        }
        
        print(f"Training with {len(self.model.protein_memory)} proteins in memory")
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0
            epoch_current_loss = 0
            epoch_memory_loss = 0
            n_batches = 0
            
            for xb, yb in loader:
                optimizer.zero_grad()
                
                # Forward pass with attention
                latent, attention_weights = self.model(xb, return_attention=True)
                
                # Loss 1: Current Data Triplet Loss
                anchor, positive, negative = self._mine_triplets(latent, yb)
                current_loss = torch.tensor(0.0, device=self.device)
                if anchor is not None:
                    current_loss = triplet_loss_fn(anchor, positive, negative)
                    epoch_current_loss += current_loss.item()
                
                # Loss 2: Memory Replay Loss
                memory_loss = torch.tensor(0.0, device=self.device)
                if len(self.model.protein_memory) > 0:

                    # Sample triplets from memory
                    mem_anchor, mem_pos, mem_neg = self.model.sample_from_memory(
                        n_triplets=min(50, len(xb)), 
                        device=self.device
                    )
                    if mem_anchor is not None:
                        memory_loss = triplet_loss_fn(mem_anchor, mem_pos, mem_neg)
                        epoch_memory_loss += memory_loss.item()
                
                # Combined loss with memory
                total_loss = current_loss + memory_replay_ratio * memory_loss
                
                if total_loss > 0:
                    total_loss.backward()
                    optimizer.step()
                    epoch_loss += total_loss.item()
                    n_batches += 1
            
            # Update memory bank after each epoch
            self.model.eval()
            with torch.no_grad():
                train_embeddings, train_attention = self.model(X_train_tensor, return_attention=True)
                self.model.update_memory(train_embeddings, y_train_tensor, train_attention)
            
            # Validation
            val_ari, val_nmi = self._validate(X_val_tensor, y_val)
            
            # Track metrics
            avg_loss = epoch_loss / max(n_batches, 1)
            avg_current = epoch_current_loss / max(n_batches, 1)
            avg_memory = epoch_memory_loss / max(n_batches, 1)
            
            history['train_loss'].append(avg_loss)
            history['current_loss'].append(avg_current)
            history['memory_loss'].append(avg_memory)
            history['val_ari'].append(val_ari)
            history['val_nmi'].append(val_nmi)
            
            if epoch % 10 == 0:
                print(f"    Epoch {epoch:3d} | Loss: {avg_loss:.4f} "
                      f"(Current: {avg_current:.4f}, Memory: {avg_memory:.4f}) "
                      f"| Val ARI: {val_ari:.4f} | Val NMI: {val_nmi:.4f}")
            
            # early stopping
            if val_ari > best_val_ari:
                best_val_ari = val_ari
                patience_counter = 0
                best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch}")
                break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        history['best_val_ari'] = best_val_ari
        return history
    
    def _mine_triplets(self, latent, labels):
        """Mine triplets from current batch"""
        anchor, positive, negative = [], [], []
        rng = np.random.RandomState(42)
        generator = torch.Generator(device=latent.device)
        generator.manual_seed(42)
        for i in range(latent.size(0)):
            same = (labels == labels[i]).nonzero(as_tuple=True)[0]
            diff = (labels != labels[i]).nonzero(as_tuple=True)[0]
            
            if len(same) > 1 and len(diff) > 0:
                pos_idx = same[same != i][rng.randint(0, len(same) - 1)]
                neg_idx = diff[rng.randint(0, len(diff))]
                anchor.append(latent[i])
                positive.append(latent[pos_idx])
                negative.append(latent[neg_idx])
        
        if anchor:
            return torch.stack(anchor), torch.stack(positive), torch.stack(negative)
        return None, None, None
    
    def _validate(self, X_val_tensor, y_val):
        """Validate using K-Means clustering"""
        self.model.eval()
        with torch.no_grad():
            val_embeddings = self.model(X_val_tensor)
            val_embeddings_norm = F.normalize(val_embeddings, p=2, dim=1).cpu().numpy()
            
            n_clusters = len(np.unique(y_val))
            # kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
            val_embeddings_2d = reducer.fit_transform(val_embeddings_norm)
            kmeans = hdbscan.HDBSCAN(min_cluster_size=5)
            val_preds = kmeans.fit_predict(val_embeddings_2d)
            
            val_ari = adjusted_rand_score(y_val, val_preds)
            val_nmi = normalized_mutual_info_score(y_val, val_preds)
            
        return val_ari, val_nmi

# ============================================================================
# Cumulative PTM Analysis Pipeline
# ============================================================================

class ProteinSequentialAnalyzer:
    """
    Main analysis for cumulative protein identification with memory and attention.
    """
    def __init__(self, base_output_dir):
        self.base_output_dir = base_output_dir
        os.makedirs(base_output_dir, exist_ok=True)
        self.results = {}

        # I manually encode labels here for consistency
        self.global_label_mapping = {
            'AS5': 0,
            'AS6': 1,
            'AS7': 2,
            'AS8': 3,
            'AS9': 4,
            'AS10': 5
            # 'H4fAcK8': 0,
            # 'H4fAcK2': 1,
            # 'H4fAcK16': 2
        }

        # Reverse mapping here so its easier to handle during plotting
        self.label_to_name = {v: k for k, v in self.global_label_mapping.items()}
        self.evolution_history = {
            'steps': [],
            'metrics': [],
            'feature_importance': []
        }

        # For when we want to add an unseen dataset for testing
        self.unseen_dataset_path = '/home/rushang_phira/src/data/complete_feature_sets/all_ladders_PL1.csv'
        self.common_features = None

    def get_protein_name(self, encoded_label):
        """Convert encoded label (0,1,2,3,4) to protein name (AS6,AS7,...)"""
        return self.label_to_name.get(encoded_label, f"Unknown_{encoded_label}")
    
    def get_protein_names(self, encoded_labels):
        """Convert array of encoded labels to protein names"""
        return [self.get_protein_name(label) for label in encoded_labels]
    
    def load_sequential_datasets(self):
        """Load datasets in cumulative sequential order. Special handling is needed because
        each dataset has different features. We find common features across all datasets"""

        protein_files = [
            ('AS6', 'Mrunal_L2_AS6_tsfresh.csv'),
            ('AS7', 'Mrunal_L2_AS7_tsfresh.csv'),
            ('AS8', 'Mrunal_L2_AS8_tsfresh.csv'),
            ('AS9', 'Mrunal_L2_AS9_tsfresh.csv'),
            ('AS10', 'Mrunal_L2_AS10_tsfresh.csv')
            # ('K12', 'Mrunal_K8_K12_tsfresh.csv'),
            # ('K16', 'Mrunal_K8_K12_K16_tsfresh.csv')
        ]
        
        print("\nLoading all datasets to determine common features...")
        all_dataframes = []
        for protein_name, filename in protein_files:
            df = self.load_and_preprocess(
                f'/home/rushang_phira/src/data/classified/feature_set/{filename}'
            )
            all_dataframes.append(df)
            print(f"  {protein_name}: {df.shape[1]-1} features, {len(df)} samples")
        
        # This step finds common features across all datasets
        common_features = set(all_dataframes[0].columns)
        for df in all_dataframes[1:]:
            common_features = common_features.intersection(set(df.columns))
        
        # Keep final_label in common features
        if 'final_label' not in common_features:
            common_features.add('final_label')
        
        common_features = sorted(list(common_features))
        print(f"\nCommon features across all datasets: {len(common_features)-1}")
        print(f"  (excluding 'final_label' column)")
        
        # self.common_features = sorted(list(common_features))
 
        self.common_features = [f for f in sorted(list(common_features)) if f != 'final_label']
        
        # Align all datasets to common features
        aligned_dataframes = []
        for df in all_dataframes:
            aligned_df = df[common_features].copy()
            aligned_dataframes.append(aligned_df)
        
        # An older step for cumulative datasets. I ended up deciding against it 
        datasets = {}
        cumulative_data = None
        
        for i, (protein_name, _) in enumerate(protein_files, start=1):
            # Use the current dataset directly
            current_data = aligned_dataframes[i-1]
            
            protein_list = '+'.join([protein_files[j][0] for j in range(i)])
            datasets[f'step_{i}_{protein_list}'] = current_data.copy()
            
            print(f"\nStep {i}: Using {protein_name} (contains all previous data implicitly)")
            print(f"  Cumulative proteins: {protein_list}")
            print(f"  Total samples: {len(current_data)}")
            print(f"  Features: {current_data.shape[1]-1}")
            print(f"  Total proteins: {len(np.unique(current_data['final_label']))}")
        
        return datasets
    
    def load_and_preprocess(self, data_path):
        """preprocessing pipeline"""
        data = pd.read_csv(data_path)
        data = data.rename(columns={'label': 'final_label'})

        data = data[data['final_label'].notna()]
        
        # NaN filtering
        nan_percentage = data.isna().mean() * 100
        data = data.loc[:, nan_percentage <= 10]
        data = data.dropna()
        
        # Column filtering
        cleaned_data = data.loc[:, ~data.columns.str.contains(
            'c3|value__quantile|value__agg_linear_trend|large_standard_deviation|change_quantiles'
        )]
        
        # Drop specific columns
        to_drop = [
            'EventId', 'idxstart', 'idxend', 'risetime', 'value__standard_deviation', 
            'value__mean_second_derivative_central', 'value__linear_trend__attr_"intercept"', 
            'falltime', "I/Io", "Io", "Irms", "length", "value__maximum", "value__minimum", 
            "log_length", 'value__variance_larger_than_standard_deviation', "value__median",
            'value__root_mean_square', 'value__standard_deviation', 
            'value__variance', 'Imean', 'Isig', 'isBL?',
            'value__count_above_mean',
            'value__count_below_mean',
            'value__longest_strike_above_mean', 
            'value__longest_strike_below_mean',
        ]
        
        if 'value__mean' in cleaned_data.columns:
            print("Mean present")
            correlations = cleaned_data.drop(columns=['final_label']).corr()['value__mean'].abs()
            high_corr_features = correlations[correlations > 0.66].index.tolist()
            # high_corr_features = [f for f in high_corr_features if f != 'value__mean' and f != 'benford_correlation']
            to_drop.extend(high_corr_features)
            print(f"Added {len(high_corr_features)} mean-correlated features to drop")

        to_drop_final = ['value__mean'] + high_corr_features + to_drop
        cleaned_data = cleaned_data.drop(columns=[col for col in to_drop_final if col in cleaned_data.columns])
        # cleaned_data = cleaned_data.drop(columns=['value__mean'] + high_corr_features + to_drop)
        # cleaned_data = cleaned_data.drop(columns=[col for col in to_drop if col in cleaned_data.columns])
        
        return cleaned_data
    
    def prepare_dataset(self, cleaned_data, label_col='final_label'):
        """Convert DataFrame to features and labels"""
        X = cleaned_data.drop(columns=[label_col]).values
        y = cleaned_data[label_col].values
        feature_names = cleaned_data.drop(columns=label_col).columns.tolist()
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        return X, y_encoded, feature_names, le
    
    def run_sequential_analysis(self):
        """Main cumulative analysis pipeline"""
        print("Cumulative Addition With Memory and Attention")
        
        datasets = self.load_sequential_datasets()
        
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {device}")
        
        model = None
        trainer = None
        
        # Process each sequential step
        for step_idx, (dataset_name, data) in enumerate(datasets.items(), start=1):
            print(f"\n{'='*80}")
            print(f"STEP {step_idx}: {dataset_name}")
            print(f"{'='*80}")
            
            # Prepare data
            X, y, feature_names, le = self.prepare_dataset(data)
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
                print(f"  Initialized model with {X_train.shape[1]} features")
            
            # Train
            print(f"Training on {len(np.unique(y_train))} classes")
            history = trainer.train_step(
                X_train, y_train, X_val, y_val,
                num_epochs=500,
                memory_replay_ratio=0.2,
                lr=1e-3 if step_idx == 1 else 1e-4  # Lower LR after first step
            )
            
            # Evaluate
            print(f"Evaluating")
            evaluation = self.evaluate_step(
                model, X_train, y_train, X_test, y_test, feature_names, device
            )

            # Generate UMAP visualizations
            self.visualize_umap_comparison(
                embeddings=evaluation['umap_data']['test_embeddings'],
                true_labels=evaluation['umap_data']['test_true_labels'],
                pred_labels=evaluation['umap_data']['test_pred_labels'],
                step_name=dataset_name,
                step_idx=step_idx
            )
            #  feature importance
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
  
            self.evolution_history['steps'].append(dataset_name)
            self.evolution_history['metrics'].append(evaluation)
            self.evolution_history['feature_importance'].append(feature_importance)
            
       
            self.visualize_step(dataset_name, step_idx)
            print(f"  Generating GLMNet visualization...")
            self.visualize_glmnet_importance(dataset_name, step_idx)
            # Save checkpoint
            self.save_checkpoint(dataset_name)
            
            print(f"  Step {step_idx} completed!")
            print(f"  Test ARI: {evaluation['test_ari']:.4f}")
            print(f"  Memory proteins: {len(model.protein_memory)}")
        
        # final reports
        self.generate_final_report()

        # Unseen dataset 
        final_evaluation = self.test_on_unseen_data(model, device)
        self.results['final_unseen_test'] = {'evaluation': final_evaluation}
        return self.results
  
    def evaluate_step(self, model, X_train, y_train, X_test, y_test, feature_names, device):
        """Comprehensive evaluation for one step"""
        model.eval()
        
        # Prepare data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            # Get embeddings
            train_embeddings = model(X_train_tensor).cpu().numpy()
            test_embeddings = model(X_test_tensor).cpu().numpy()
            
            # Normalize
            train_embeddings_norm = train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True)
            test_embeddings_norm = test_embeddings / np.linalg.norm(test_embeddings, axis=1, keepdims=True)

        # GLMNet (old now. Refer to attention, and the vif.py or updated feature selection)
        from sklearn.linear_model import ElasticNet
        glmnet_importance = self.calculate_glmnet_importance(
            model, X_train_scaled, feature_names, device
        )
        # Clustering evaluation
        n_clusters = len(np.unique(y_train))
        train_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        train_embeddings_2d = train_reducer.fit_transform(train_embeddings_norm)

        test_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        test_embeddings_2d = test_reducer.fit_transform(test_embeddings_norm)

        # tested with KMeans for a bit. Comment out KMeans and use HDBSCAN if needed (I used HDBSCAN)
        kmeans_test = hdbscan.HDBSCAN(min_cluster_size=5) 
        # KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
        test_preds = kmeans_test.fit_predict(test_embeddings_2d)
        
        kmeans_train = hdbscan.HDBSCAN(min_cluster_size=5) 
        train_preds = kmeans_train.fit_predict(train_embeddings_2d)
        metrics = self.calculate_comprehensive_metrics(
            test_embeddings_2d, y_test, test_preds, n_clusters
        )
        
        train_metrics = self.calculate_comprehensive_metrics(
            train_embeddings_2d, y_train, train_preds, n_clusters
        )
        
        # Per-PTM metrics
        per_protein = self.calculate_per_protein_metrics(
            test_embeddings_norm, y_test, test_preds, n_clusters
        )
        
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
            'per_protein': per_protein,
            'umap_data': umap_data,
            'glmnet_importance': glmnet_importance
        }
        
    def calculate_glmnet_importance(self, model, features, feature_names, device):
        """Calculate feature importance using GLMNet/ElasticNet (Now old)"""
        from sklearn.linear_model import ElasticNet
        
        model.eval()
        with torch.no_grad():
            # Get latent embeddings for all features
            features_tensor = torch.tensor(features, dtype=torch.float32, device=device)
            latent_full = model(features_tensor).cpu().numpy()

        # Fit ElasticNet for each latent dimension
        coef_list = []
        for i in range(latent_full.shape[1]):
            try:
                m = ElasticNet(alpha=1.0, max_iter=100, random_state=42)
                m.fit(features, latent_full[:, i])
                coef_list.append(np.abs(m.coef_))
            except:
                # Fallback if ElasticNet fails
                coef_list.append(np.zeros(len(feature_names)))
        
        # Average absolute coefficients across latent dimensions
        if coef_list:
            glmnet_importance_scores = np.mean(coef_list, axis=0)
        else:
            glmnet_importance_scores = np.zeros(len(feature_names))
        
        # Create importance dictionary
        glmnet_importance = {
            feature_names[i]: float(glmnet_importance_scores[i])
            for i in range(len(feature_names))
        }
        
        return glmnet_importance
    
    def visualize_glmnet_importance(self, dataset_name, step_idx):
        """Create separate plot for GLMNet feature importance only"""
        result = self.results[dataset_name]
        
        glmnet_importance = result['evaluation']['glmnet_importance']
        feature_names = result['feature_names']
        
        sorted_features = sorted(
            glmnet_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:30]  # Top 30
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        feature_names_top = [f[0][:50] for f in sorted_features] 
        glmnet_scores = [f[1] for f in sorted_features]
        
        y_pos = np.arange(len(feature_names_top))
        
        bars = ax.barh(y_pos, glmnet_scores, alpha=0.8, color='coral', edgecolor='darkred', linewidth=0.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names_top, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('GLMNet Importance Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Step {step_idx}: Top 30 Features by GLMNet Importance\n(Dataset: {dataset_name})', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, glmnet_scores)):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{score:.4f}', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        
        glmnet_path = os.path.join(self.base_output_dir, f'step_{step_idx}_glmnet_importance.png')
        plt.savefig(glmnet_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    GLMNet importance plot saved: {glmnet_path}")
        
    def calculate_comprehensive_metrics(self, embeddings, true_labels, pred_labels, n_clusters):
        """
        Calculate extensive clustering evaluation metrics.
        """
        from sklearn.metrics import (
            adjusted_rand_score, normalized_mutual_info_score, 
            silhouette_score, calinski_harabasz_score, davies_bouldin_score,
            fowlkes_mallows_score, adjusted_mutual_info_score,
            homogeneity_score, completeness_score, v_measure_score
        )
        
        metrics = {}
        
        # EXTERNAL METRICS

        metrics['ari'] = adjusted_rand_score(true_labels, pred_labels)
      
        metrics['nmi'] = normalized_mutual_info_score(true_labels, pred_labels)
        
        metrics['ami'] = adjusted_mutual_info_score(true_labels, pred_labels)
        
        metrics['fmi'] = fowlkes_mallows_score(true_labels, pred_labels)
        
        metrics['homogeneity'] = homogeneity_score(true_labels, pred_labels)
        
        metrics['completeness'] = completeness_score(true_labels, pred_labels)
        
        metrics['v_measure'] = v_measure_score(true_labels, pred_labels)
        
        # INTERNAL METRICS 

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
        
        # SOME ADDITIONAL CLUSTER QUALITY METRICS
        # Intra-cluster distance
        intra_cluster_dists = []
        for cluster_id in np.unique(pred_labels):
            cluster_points = embeddings[pred_labels == cluster_id]
            if len(cluster_points) > 1:
                intra_cluster_dists.append(np.mean(pdist(cluster_points)))
        metrics['mean_intra_cluster_distance'] = np.mean(intra_cluster_dists) if intra_cluster_dists else 0.0
        
        # Inter-cluster distance
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
        
        # Dunn Index: ratio of min inter-cluster to max intra-cluster distance
        if intra_cluster_dists and len(cluster_centers) > 1:
            metrics['dunn_index'] = metrics['min_inter_cluster_distance'] / max(intra_cluster_dists)
        else:
            metrics['dunn_index'] = 0.0
        
        # CONFUSION-BASED METRICS 
        # Purity: fraction of total samples where cluster's most common class
        cluster_purity = []
        for cluster_id in np.unique(pred_labels):
            cluster_mask = pred_labels == cluster_id
            cluster_true_labels = true_labels[cluster_mask]
            if len(cluster_true_labels) > 0:
                most_common = np.bincount(cluster_true_labels).argmax()
                purity = np.sum(cluster_true_labels == most_common) / len(cluster_true_labels)
                cluster_purity.append(purity)
        metrics['purity'] = np.mean(cluster_purity) if cluster_purity else 0.0
        
        # Cluster balance: std of cluster sizes (lower = more balanced)
        cluster_sizes = [np.sum(pred_labels == i) for i in np.unique(pred_labels)]
        metrics['cluster_size_std'] = np.std(cluster_sizes)
        metrics['cluster_size_cv'] = np.std(cluster_sizes) / np.mean(cluster_sizes) if np.mean(cluster_sizes) > 0 else 0.0
        
        return metrics

    def print_clustering_metrics(self, metrics, dataset_name, data_split='test'):
        """Print comprehensive metrics with formatting"""
        print(f"  COMPREHENSIVE CLUSTERING METRICS - {dataset_name} ({data_split.upper()})")
        
        print(f"\n  EXTERNAL METRICS (Agreement with Ground Truth):")
        print(f"     ARI (Adjusted Rand Index):           {metrics['ari']:.4f}")
        print(f"     NMI (Normalized Mutual Info):        {metrics['nmi']:.4f}")
        print(f"     AMI (Adjusted Mutual Info):          {metrics['ami']:.4f}")
        print(f"     FMI (Fowlkes-Mallows Index):         {metrics['fmi']:.4f}")
        print(f"     Homogeneity:                         {metrics['homogeneity']:.4f}")
        print(f"     Completeness:                        {metrics['completeness']:.4f}")
        print(f"     V-Measure:                           {metrics['v_measure']:.4f}")
        print(f"     Purity:                              {metrics['purity']:.4f}")
        
        print(f"\n  INTERNAL METRICS (Cluster Quality):")
        print(f"     Silhouette Score:                    {metrics['silhouette']:.4f}")
        print(f"     Calinski-Harabasz Index:             {metrics['calinski_harabasz']:.2f}")
        print(f"     Davies-Bouldin Index:                {metrics['davies_bouldin']:.4f}")
        print(f"     Dunn Index:                          {metrics['dunn_index']:.4f}")
        
        print(f"\n  DISTANCE METRICS:")
        print(f"     Mean Intra-cluster Distance:         {metrics['mean_intra_cluster_distance']:.4f}  [Lower is better]")
        print(f"     Mean Inter-cluster Distance:         {metrics['mean_inter_cluster_distance']:.4f}  [Higher is better]")
        print(f"     Min Inter-cluster Distance:          {metrics['min_inter_cluster_distance']:.4f}  [Higher is better]")
        
        print(f"\n  BALANCE METRICS:")
        print(f"     Cluster Size Std Dev:                {metrics['cluster_size_std']:.2f}  [Lower is better]")
        print(f"     Cluster Size CV:                     {metrics['cluster_size_cv']:.4f} ")
        
        # Overall assessment
        overall_score = (
            metrics['ari'] + metrics['nmi'] + metrics['v_measure'] + 
            metrics['silhouette'] + (1 - min(metrics['davies_bouldin'], 2)/2)
        ) / 5
    
        
    def align_features(self, X_unseen, unseen_feature_names, unseen_data):
        """Align unseen data features with training feature set"""
        if self.common_features is None:
            raise ValueError("Common features not set - run training first")
        
        unseen_df = pd.DataFrame(X_unseen, columns=unseen_feature_names)
        
        # Ensure we have all the common features (add missing ones with zeros)
        aligned_df = pd.DataFrame()
        
        # Use self.common_features (which already excludes final_label)
        for feature in self.common_features:
            if feature in unseen_df.columns:
                aligned_df[feature] = unseen_df[feature]
            else:
                # Feature missing in unseen data - fill with zeros
                aligned_df[feature] = 0.0
                print(f"    Warning: Feature '{feature}' missing in unseen data, filled with zeros")
        
        # Ensure correct feature order
        aligned_df = aligned_df[self.common_features]
        
        X_aligned = aligned_df.values
        
        print(f"    Aligned unseen data: {X_unseen.shape[1]} → {X_aligned.shape[1]} features")
        
        return X_aligned

    def test_on_unseen_data(self, model, device):
        """Test on completely unseen dataset containing AS4-AS10"""
        print("Generalization test: Unseen Data")
        
        # Load the unseen dataset
        print(f"  Loading unseen dataset: {self.unseen_dataset_path}")
        unseen_data = self.load_and_preprocess(self.unseen_dataset_path)
        
        X_unseen, y_unseen, feature_names, le = self.prepare_dataset(unseen_data)
        
        # Align features with trained model
        X_unseen_aligned = self.align_features(X_unseen, feature_names, unseen_data)
        
        print(f"  Unseen dataset: {len(X_unseen_aligned)} samples")
        print(f"  Proteins: {len(np.unique(y_unseen))} classes")
        print(f"  Protein distribution: {dict(zip(*np.unique(y_unseen, return_counts=True)))}")
        
        # Use a simplified evaluation that doesn't call calculate_glmnet_importance since we dont care about it anymore
        evaluation = self.evaluate_step_simple(model, X_unseen_aligned, y_unseen, X_unseen_aligned, y_unseen, feature_names, device)        
    
        self.visualize_final_unseen_test(evaluation)
        
        return evaluation

    def evaluate_step_simple(self, model, X_train, y_train, X_test, y_test, feature_names, device):
        """Simplified evaluation without GLMNet"""
        model.eval()
        n_clusters =  len(np.unique(y_train))

        # Prepare data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            # Get embeddings
            train_embeddings = model(X_train_tensor).cpu().numpy()
            test_embeddings = model(X_test_tensor).cpu().numpy()
            
            # Normalize
            train_embeddings_norm = train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True)
            test_embeddings_norm = test_embeddings / np.linalg.norm(test_embeddings, axis=1, keepdims=True)
        

        train_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        train_embeddings_2d = train_reducer.fit_transform(train_embeddings_norm)
        
        test_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        test_embeddings_2d = test_reducer.fit_transform(test_embeddings_norm)
        
        # Cluster in 2D space
        kmeans_test = hdbscan.HDBSCAN(min_cluster_size=5)
        test_preds = kmeans_test.fit_predict(test_embeddings_2d)
        
        kmeans_train = hdbscan.HDBSCAN(min_cluster_size=5)
        train_preds = kmeans_train.fit_predict(train_embeddings_2d)
        
        # METRICS
        metrics = self.calculate_comprehensive_metrics(test_embeddings_2d, y_test, test_preds, n_clusters)
        train_metrics = self.calculate_comprehensive_metrics(train_embeddings_2d, y_train, train_preds, n_clusters)
        
        # Per-PTM metrics
        per_protein = self.calculate_per_protein_metrics(test_embeddings_2d, y_test, test_preds, len(np.unique(y_train)))
        
        # UMAP data
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
            'n_clusters': len(np.unique(y_train)),
            'per_protein': per_protein,
            'umap_data': umap_data,
            'glmnet_importance': {}  # Empty dict instead of calculated GLMNet
        }

    def visualize_final_unseen_test(self, evaluation):
        """Create visualization for unseen test results with protein name legend"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # UMAP plot with legend
        umap_data = evaluation['umap_data']
        true_labels = umap_data['test_true_labels']
        
        # Create scatter plot with protein names in legend
        scatter = axes[0].scatter(umap_data['test_embeddings'][:, 0], 
                                umap_data['test_embeddings'][:, 1],
                                c=true_labels, cmap='tab10', s=7, alpha=0.2)
        
        # Create legend with names instead of numbers
        unique_labels = np.unique(true_labels)
        legend_elements = []
        for label in unique_labels:
            protein_name = self.get_protein_name(label)
            color = scatter.cmap(scatter.norm(label))
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', 
                        markerfacecolor=color, markersize=8, label=protein_name)
            )
        
        axes[0].legend(handles=legend_elements, loc='upper right', 
                    bbox_to_anchor=(1.15, 1), fontsize=8, framealpha=0.9)
        
        axes[0].set_title('Unseen Data: AS4-AS10\n(Completely Unseen Test Set)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('UMAP 1')
        axes[0].set_ylabel('UMAP 2')
        axes[0].grid(True, alpha=0.3)
        

        axes[1].axis('off')
        
        per_protein = evaluation.get('per_protein', {})
        
        summary_text = f"UNSEEN TEST RESULTS\n"
        summary_text += f"Overall Metrics:\n"
        summary_text += f"• ARI: {evaluation['test_ari']:.4f}\n"
        summary_text += f"• NMI: {evaluation['test_nmi']:.4f}\n"
        summary_text += f"• Silhouette: {evaluation['test_silhouette']:.4f}\n\n"
        
        summary_text += f"Dataset Info:\n"
        summary_text += f"• Samples: {len(true_labels)}\n"
        summary_text += f"• Proteins: {len(unique_labels)}\n\n"
        
        # Add per-protein performance if available
        if per_protein:
            summary_text += f"Per-Protein F1 Scores:\n"
            for protein_name, metrics in list(per_protein.items())[:6]:  # Show top 6
                summary_text += f"• {protein_name}: {metrics.get('f1', 0):.3f}\n"

        
        axes[1].text(0.1, 0.95, summary_text, transform=axes[1].transAxes, 
                    fontfamily='monospace', fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8))
        
        # Add some arbitrary performance assessment numbers
        ari_score = evaluation['test_ari']
        if ari_score > 0.8:
            assessment = "EXCELLENT"
            color = "green"
        elif ari_score > 0.6:
            assessment = "GOOD" 
            color = "orange"
        elif ari_score > 0.4:
            assessment = "MODERATE"
            color = "orange"
        else:
            assessment = "POOR"
            color = "red"
        
        assessment_text = f"Generalization Assessment:\n{assessment}\n(ARI = {ari_score:.3f})"
        axes[1].text(0.5, 0.1, assessment_text, transform=axes[1].transAxes,
                    fontsize=12, fontweight='bold', color=color, ha='center',
                    bbox=dict(boxstyle="round", facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.base_output_dir, 'final_unseen_test.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Final unseen test visualization saved with protein legend")

    def calculate_per_protein_metrics(self, embeddings_2d, true_labels, pred_labels, n_clusters):
        from scipy.optimize import linear_sum_assignment
        
        # Get unique predicted clusters (HDBSCAN can have many clusters + noise label -1)
        unique_preds = np.unique(pred_labels)
        unique_trues = np.unique(true_labels)
        
        # Handle noise points from HDBSCAN (label = -1)
        valid_preds = pred_labels[pred_labels != -1]
        valid_trues = true_labels[pred_labels != -1]
        
        if len(valid_preds) == 0:
            return {}  # No valid clusters found
        
        # Create confusion matrix between valid predicted clusters and true labels
        n_pred_clusters = len(np.unique(valid_preds))
        n_true_clusters = len(unique_trues)
        
        confusion = np.zeros((n_pred_clusters, n_true_clusters))
        
        # Map cluster labels to 0-indexed for confusion matrix
        pred_map = {label: idx for idx, label in enumerate(np.unique(valid_preds))}
        true_map = {label: idx for idx, label in enumerate(unique_trues)}
        
        for pred, true in zip(valid_preds, valid_trues):
            confusion[pred_map[pred], true_map[true]] += 1
        
        # Hungarian algorithm for optimal assignment
        row_ind, col_ind = linear_sum_assignment(-confusion)
        
        per_protein_metrics = {}
        
        for pred_idx, true_idx in zip(row_ind, col_ind):
            # Get original labels back from mapping
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
            
            # Use protein NAME instead of number
            protein_name = self.get_protein_name(true_cluster)
            per_protein_metrics[protein_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'n_samples': np.sum(true_mask),
                'encoded_label': true_cluster,
                'assigned_cluster': pred_cluster
            }
        
        # Handle unassigned proteins (if any)
        assigned_true_clusters = [list(true_map.keys())[list(true_map.values()).index(idx)] for idx in col_ind]
        all_true_clusters = list(true_map.keys())
        
        for true_cluster in all_true_clusters:
            if true_cluster not in assigned_true_clusters:
                protein_name = self.get_protein_name(true_cluster)
                per_protein_metrics[protein_name] = {
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'n_samples': np.sum(valid_trues == true_cluster),
                    'encoded_label': true_cluster,
                    'assigned_cluster': None
                }
        
        return per_protein_metrics
    
    def visualize_umap_comparison(self, embeddings, true_labels, pred_labels, step_name, step_idx):
        """UMAP visualization with protein names instead of numbers"""
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings_norm)
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Normalize color range
        norm = mcolors.Normalize(vmin=np.min(true_labels), vmax=np.max(true_labels))
        
        # === PLOT 1: True Labels with Protein Names ===
        scatter1 = axes[0].scatter(
            embedding_2d[:, 0], embedding_2d[:, 1],
            c=true_labels, cmap='tab10', s=50, alpha=0.7,
            edgecolors='black', linewidth=0.5, norm=norm
        )

        # === Create ScalarMappable for consistent legend colors ===
        sm = plt.cm.ScalarMappable(cmap='tab10', norm=norm)
        sm.set_array([])  # To avoid warning when creating legend

        # Create legend labels
        unique_labels = np.unique(true_labels)
        legend_labels = [self.get_protein_name(label) for label in unique_labels]

        # === Legend using ScalarMappable ===
        legend_handles = [
            plt.Line2D([0], [0], marker='o', color='w', 
                    markerfacecolor=sm.to_rgba(label),  # Use ScalarMappable for consistent color
                    markersize=10, label=name)
            for label, name in zip(unique_labels, legend_labels)
        ]
        
        # Add the legend with consistent colors
        axes[0].legend(handles=legend_handles, loc='best', fontsize=10, framealpha=0.9)
        axes[0].set_title(f'True Protein Labels ({len(np.unique(true_labels))} proteins)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('UMAP 1')
        axes[0].set_ylabel('UMAP 2')
        axes[0].grid(True, alpha=0.3)
        
        # === PLOT 2: Predicted Clusters ===
        scatter2 = axes[1].scatter(
            embedding_2d[:, 0], embedding_2d[:, 1],
            c=pred_labels, cmap='Set2', s=50, alpha=0.7,
            edgecolors='black', linewidth=0.5
        )
        
        # === PLOT 3: Agreement Map ===
        scatter3 = axes[2].scatter(
            embedding_2d[:, 0], embedding_2d[:, 1],
            c=pred_labels, cmap='Set2', s=80, alpha=0.6, edgecolors='none'
        )

        axes[2].set_title('Agreement Map\n(Fill=Predicted, Border=True)', 
                        fontsize=14, fontweight='bold')
        axes[2].set_xlabel('UMAP 1')
        axes[2].set_ylabel('UMAP 2')
        axes[2].grid(True, alpha=0.3)
        
        # === Save Figure ===
        plt.tight_layout()
        
        umap_path = os.path.join(self.base_output_dir, f'step_{step_idx}_umap_comparison.png')
        plt.savefig(umap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    UMAP visualization saved: {umap_path}")
    
    def visualize_step(self, dataset_name, step_idx):
        """Generate comprehensive visualizations for current step"""
        result = self.results[dataset_name]
        
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig)
        
        # 1. Training history
        ax1 = fig.add_subplot(gs[0, 0])
        history = result['history']
        ax1.plot(history['train_loss'], label='Total Loss')
        ax1.plot(history['current_loss'], label='Current Loss')
        ax1.plot(history['memory_loss'], label='Memory Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Validation metrics
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(history['val_ari'], label='ARI', marker='o')
        ax2.plot(history['val_nmi'], label='NMI', marker='s')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.set_title('Validation Metrics')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Performance comparison
        ax3 = fig.add_subplot(gs[0, 2])
        eval_metrics = result['evaluation']
        metrics = ['ARI', 'NMI', 'Silhouette']
        train_scores = [eval_metrics['train_ari'], eval_metrics['train_nmi'], eval_metrics['train_silhouette']]
        test_scores = [eval_metrics['test_ari'], eval_metrics['test_nmi'], eval_metrics['test_silhouette']]
        
        x = np.arange(len(metrics))
        width = 0.35
        ax3.bar(x - width/2, train_scores, width, label='Train', alpha=0.8)
        ax3.bar(x + width/2, test_scores, width, label='Test', alpha=0.8)
        ax3.set_ylabel('Score')
        ax3.set_title('Train vs Test Performance')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics)
        ax3.legend()
        ax3.set_ylim(0, 1)
        
        # 4. Top 20 Feature Importance
        ax4 = fig.add_subplot(gs[1, :])
        importance = result['feature_importance']['overall']
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
        feature_names_top = [f[:40] for f, _ in sorted_features]  # Truncate long names
        feature_scores = [s for _, s in sorted_features]
        
        y_pos = np.arange(len(feature_names_top))
        ax4.barh(y_pos, feature_scores, alpha=0.8, color='steelblue')
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(feature_names_top, fontsize=8)
        ax4.invert_yaxis()
        ax4.set_xlabel('Attention Weight')
        ax4.set_title('Top 20 Most Important Features (Overall)')
        ax4.grid(True, alpha=0.3, axis='x')
        
        # 5. Per-protein performance
        ax5 = fig.add_subplot(gs[2, 0])
        per_protein = eval_metrics['per_protein']
        protein_ids = list(per_protein.keys())
        f1_scores = [per_protein[p]['f1'] for p in protein_ids]
        
        ax5.bar(protein_ids, f1_scores, alpha=0.8, color='green')
        ax5.set_ylabel('F1 Score')
        ax5.set_title('Per-Protein F1 Score (Test Set)')
        ax5.set_ylim(0, 1)
        ax5.tick_params(axis='x', rotation=45)
        
        # 6. Per-protein precision/recall
        ax6 = fig.add_subplot(gs[2, 1])
        precisions = [per_protein[p]['precision'] for p in protein_ids]
        recalls = [per_protein[p]['recall'] for p in protein_ids]
        
        x = np.arange(len(protein_ids))
        width = 0.35
        ax6.bar(x - width/2, precisions, width, label='Precision', alpha=0.8)
        ax6.bar(x + width/2, recalls, width, label='Recall', alpha=0.8)
        ax6.set_ylabel('Score')
        ax6.set_title('Per-Protein Precision & Recall')
        ax6.set_xticks(x)
        ax6.set_xticklabels(protein_ids, rotation=45)
        ax6.legend()
        ax6.set_ylim(0, 1)
        
        # 7. Memory bank status
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        memory_text = f"MEMORY BANK STATUS\n\n"
        memory_text += f"Step: {step_idx}\n"
        memory_text += f"Proteins in memory: {result['n_proteins']}\n\n"
        memory_text += f"Performance:\n"
        memory_text += f"  Test ARI: {eval_metrics['test_ari']:.4f}\n"
        memory_text += f"  Test NMI: {eval_metrics['test_nmi']:.4f}\n"
        memory_text += f"  Test Silhouette: {eval_metrics['test_silhouette']:.4f}\n\n"
        memory_text += f"Best Val ARI: {history['best_val_ari']:.4f}\n"
        
        ax7.text(0.1, 0.9, memory_text, transform=ax7.transAxes, 
                fontfamily='monospace', verticalalignment='top', fontsize=11)
        
        plt.suptitle(f'Step {step_idx}: {dataset_name}', fontsize=16, y=0.98)
        plt.tight_layout()
        
        # Save
        plot_path = os.path.join(self.base_output_dir, f'step_{step_idx}_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Visualization saved: {plot_path}")
    
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
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        print(f"\n{'='*80}")
        print("GENERATING FINAL REPORT")
        print(f"{'='*80}")
        
        self.plot_evolution()
                
        self.plot_umap_evolution()

        self.analyze_feature_consistency()

        self.create_summary_report()
        
        print(f"All results saved to: {self.base_output_dir}")
    
    def plot_evolution(self):
        """Plot performance evolution across steps"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        steps = [m['step'] for k, m in self.results.items()]
        step_names = list(self.results.keys())
        
        # Extract metrics
        train_ari = [self.results[k]['evaluation']['train_ari'] for k in step_names]
        test_ari = [self.results[k]['evaluation']['test_ari'] for k in step_names]
        train_nmi = [self.results[k]['evaluation']['train_nmi'] for k in step_names]
        test_nmi = [self.results[k]['evaluation']['test_nmi'] for k in step_names]
        
        # Plot ARI evolution
        axes[0, 0].plot(steps, train_ari, 'o-', label='Train', linewidth=2, markersize=8)
        axes[0, 0].plot(steps, test_ari, 's-', label='Test', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('ARI Score')
        axes[0, 0].set_title('ARI Evolution Across Steps')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 1)
        
        # Plot NMI evolution
        axes[0, 1].plot(steps, train_nmi, 'o-', label='Train', linewidth=2, markersize=8)
        axes[0, 1].plot(steps, test_nmi, 's-', label='Test', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('NMI Score')
        axes[0, 1].set_title('NMI Evolution Across Steps')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1)
        
        # Plot generalization gap
        axes[1, 0].plot(steps, np.array(train_ari) - np.array(test_ari), 
                       'o-', linewidth=2, markersize=8, color='red')
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Train - Test ARI')
        axes[1, 0].set_title('Generalization Gap')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot number of proteins
        n_proteins = [self.results[k]['n_proteins'] for k in step_names]
        axes[1, 1].bar(steps, n_proteins, alpha=0.8, color='green')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Number of Proteins')
        axes[1, 1].set_title('Cumulative Proteins')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Sequential Learning Evolution', fontsize=16, y=0.98)
        plt.tight_layout()
        
        evolution_path = os.path.join(self.base_output_dir, 'evolution_analysis.png')
        plt.savefig(evolution_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Evolution plot saved: {evolution_path}")

    def plot_umap_evolution(self):
        """
        Create a comprehensive UMAP evolution plot showing all steps.
        """
        print("\n  Creating UMAP evolution visualization...")
        
        n_steps = len(self.results)
        fig = plt.figure(figsize=(20, 4 * ((n_steps + 1) // 2)))
        
        for idx, (step_name, result) in enumerate(self.results.items(), start=1):
            umap_data = result['evaluation']['umap_data']
            
            # Get test data
            embeddings = umap_data['test_embeddings']
            true_labels = umap_data['test_true_labels']
            pred_labels = umap_data['test_pred_labels']
            
            # Normalize and compute UMAP
            embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
            embedding_2d = reducer.fit_transform(embeddings_norm)
            
            # Create subplot with 2 panels (true vs predicted)
            ax1 = plt.subplot(n_steps, 2, 2*idx - 1)
            ax2 = plt.subplot(n_steps, 2, 2*idx)
            
            # True labels
            scatter1 = ax1.scatter(
                embedding_2d[:, 0], embedding_2d[:, 1],
                c=true_labels, cmap='tab10', s=7, alpha=0.5
            )
            ax1.set_title(f'Step {idx}: True Labels', fontsize=12, fontweight='bold')
            ax1.set_xlabel('UMAP 1')
            ax1.set_ylabel('UMAP 2')
            ax1.grid(True, alpha=0.3)
            
            # Predicted labels
            scatter2 = ax2.scatter(
                embedding_2d[:, 0], embedding_2d[:, 1],
                c=pred_labels, cmap='Set2', s=7, alpha=0.5
            )
            ax2.set_title(f'Step {idx}: Predicted (ARI={result["evaluation"]["test_ari"]:.3f})', 
                        fontsize=12, fontweight='bold')
            ax2.set_xlabel('UMAP 1')
            ax2.set_ylabel('UMAP 2')
            ax2.grid(True, alpha=0.3)
        
        plt.suptitle('UMAP Evolution Across All Sequential Steps', fontsize=16, y=0.995)
        plt.tight_layout()
        
        evolution_umap_path = os.path.join(self.base_output_dir, 'umap_evolution_all_steps.png')
        plt.savefig(evolution_umap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  UMAP evolution plot saved: {evolution_umap_path}")

    def analyze_feature_consistency(self):
        """Analyze which features are consistently important across all steps. Not same as feature importance, might differ."""
        print("\nfeature consistency")
        
        # Collect importance scores across all steps
        all_features = self.results[list(self.results.keys())[0]]['feature_names']
        feature_scores_across_steps = defaultdict(list)
        
        for step_name, result in self.results.items():
            importance = result['feature_importance']['overall']
            for feature in all_features:
                feature_scores_across_steps[feature].append(importance.get(feature, 0))
        
        # Calculate consistency metrics
        consistency_analysis = {}
        for feature, scores in feature_scores_across_steps.items():
            consistency_analysis[feature] = {
                'mean_importance': np.mean(scores),
                'std_importance': np.std(scores),
                'consistency_score': np.mean(scores) / (np.std(scores) + 1e-8),
                'min_importance': np.min(scores),
                'max_importance': np.max(scores),
                'scores_per_step': scores
            }
        
        # Rank by consistency
        ranked_features = sorted(
            consistency_analysis.items(),
            key=lambda x: x[1]['consistency_score'],
            reverse=True
        )
        
        # Print top features
        print("\n  TOP 30 MOST CONSISTENTLY IMPORTANT FEATURES:")
        print("  " + "-"*100)
        print(f"  {'Rank':<6}{'Feature':<60}{'Mean':<10}{'Std':<10}{'Consistency':<12}")
        print("  " + "-"*100)
        
        for i, (feature, metrics) in enumerate(ranked_features[:30], 1):
            print(f"  {i:<6}{feature[:58]:<60}{metrics['mean_importance']:<10.4f}"
                  f"{metrics['std_importance']:<10.4f}{metrics['consistency_score']:<12.2f}")
        
        # Save full analysis
        analysis_path = os.path.join(self.base_output_dir, 'feature_importance_analysis.json')
        with open(analysis_path, 'w') as f:
            json.dump({
                'ranked_features': [
                    {
                        'feature': feature,
                        'rank': i,
                        'metrics': metrics
                    }
                    for i, (feature, metrics) in enumerate(ranked_features, 1)
                ],
                'summary': {
                    'total_features': len(all_features),
                    'highly_consistent_features': len([f for f, m in ranked_features 
                                                      if m['consistency_score'] > 10]),
                    'moderately_consistent_features': len([f for f, m in ranked_features 
                                                          if 5 < m['consistency_score'] <= 10])
                }
            }, f, indent=2)
        
        print(f"\n  Full feature analysis saved: {analysis_path}")
        
        # Visualize top features
        self.visualize_top_features(ranked_features[:30])
    
    def visualize_top_features(self, top_features):
        """Visualize top important features across steps"""
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # Extract data
        feature_names = [f[:40] for f, _ in top_features[:20]]  # Top 20
        consistency_scores = [m['consistency_score'] for _, m in top_features[:20]]
        mean_importance = [m['mean_importance'] for _, m in top_features[:20]]
        
        # Plot 1: Consistency scores
        y_pos = np.arange(len(feature_names))
        axes[0].barh(y_pos, consistency_scores, alpha=0.8, color='steelblue')
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(feature_names, fontsize=9)
        axes[0].invert_yaxis()
        axes[0].set_xlabel('Consistency Score (Mean/Std)')
        axes[0].set_title('Top 20 Most Consistent Features Across All Steps')
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Plot 2: Heatmap of importance across steps
        step_names = [f"Step {i+1}" for i in range(len(self.results))]
        importance_matrix = np.array([
            top_features[i][1]['scores_per_step'] 
            for i in range(min(20, len(top_features)))
        ])
        
        im = axes[1].imshow(importance_matrix, aspect='auto', cmap='YlOrRd')
        axes[1].set_yticks(np.arange(len(feature_names)))
        axes[1].set_yticklabels(feature_names, fontsize=9)
        axes[1].set_xticks(np.arange(len(step_names)))
        axes[1].set_xticklabels(step_names)
        axes[1].set_title('Feature Importance Heatmap Across Steps')
        plt.colorbar(im, ax=axes[1], label='Attention Weight')
        
        plt.tight_layout()
        
        feature_viz_path = os.path.join(self.base_output_dir, 'feature_consistency_analysis.png')
        plt.savefig(feature_viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature consistency visualization saved: {feature_viz_path}")
    
    def create_summary_report(self):
        """Create final text summary report"""
        summary_lines = []
        summary_lines.append("FINAL SUMMARY")
  
        summary_lines.append("")

        summary_lines.append("OVERALL PERFORMANCE:")

        for step_name, result in self.results.items():
            eval_metrics = result['evaluation']
            summary_lines.append(f"\n{step_name}:")
            summary_lines.append(f"  Proteins: {result['n_proteins']}")
            summary_lines.append(f"  Train ARI: {eval_metrics['train_ari']:.4f}")
            summary_lines.append(f"  Test ARI:  {eval_metrics['test_ari']:.4f}")
            summary_lines.append(f"  Test NMI:  {eval_metrics['test_nmi']:.4f}")
            summary_lines.append(f"  Test Silhouette: {eval_metrics['test_silhouette']:.4f}")
        
        # Final step detailed analysis
        final_step_name = list(self.results.keys())[-1]
        final_result = self.results[final_step_name]
        
        summary_lines.append("\n")
        summary_lines.append(f"FINAL STEP ANALYSIS ({final_step_name}):")
        
        per_protein = final_result['evaluation']['per_protein']
        summary_lines.append("\nPer-Protein Performance:")
        summary_lines.append("-"*80)
        summary_lines.append(f"{'Protein':<15}{'Precision':<12}{'Recall':<12}{'F1 Score':<12}{'Samples':<10}")
        summary_lines.append("-"*80)
        
        for protein_id, metrics in per_protein.items():
            summary_lines.append(
                f"{protein_id:<15}"
                f"{metrics['precision']:<12.4f}"
                f"{metrics['recall']:<12.4f}"
                f"{metrics['f1']:<12.4f}"
                f"{metrics['n_samples']:<10}"
            )
        
        # Learning trajectory
        summary_lines.append("\n")
        summary_lines.append("LEARNING TRAJECTORY:")
        
        test_aris = [self.results[k]['evaluation']['test_ari'] for k in self.results.keys()]
        initial_ari = test_aris[0]
        final_ari = test_aris[-1]
        ari_change = final_ari - initial_ari
        
        summary_lines.append(f"\nInitial Test ARI (Step 1): {initial_ari:.4f}")
        summary_lines.append(f"Final Test ARI (Step {len(test_aris)}): {final_ari:.4f}")
        summary_lines.append(f"Change: {ari_change:+.4f}")
        
        # Save report
        report_path = os.path.join(self.base_output_dir, 'final_summary_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        # Also print to console
        print("\n" + '\n'.join(summary_lines))
        print(f"\nSummary report saved: {report_path}")

def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("\n")
    print(" MEMORY ATTENTION TRIPLET NETWORK")
    print("\n")
    
    # Initialize analyzer
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/home/rushang_phira/src/report/attention_tanh_1.0_no_benford_ALL_PL1_Unseen"
    analyzer = ProteinSequentialAnalyzer(output_dir)
    
    # Run sequential analysis
    results = analyzer.run_sequential_analysis()
    
    print("\n")
    print("COMPLETED")
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - step_X_analysis.png: Detailed analysis for each step")
    print("  - evolution_analysis.png: Performance evolution across steps")
    print("  - feature_consistency_analysis.png: Top features visualization")
    print("  - feature_importance_analysis.json: Complete feature rankings")
    print("  - final_summary_report.txt: Text summary of results")
    print("  - checkpoint_*.json: Intermediate checkpoints")
    print("\n")

if __name__ == "__main__":
    main()