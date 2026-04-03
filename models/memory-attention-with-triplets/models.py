import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MemoryAttentionTripletModel(nn.Module):
    """
    Modification on my initial triplet-loss network with additions of:
    1. Feature attention for identifying important (tsfresh) features
    2. Memory
    """

    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.3):
        super(MemoryAttentionTripletModel, self).__init__()
        
        '''Attention'''
        self.feature_attention = nn.Sequential(
            nn.Linear(input_dim, max(32, input_dim // 4)),  # bottleneck here
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(max(32, input_dim // 4), input_dim),
            nn.Sigmoid()
        )
        self.attention_l1_lambda = 1e-4
        
        # Same encoder from triplet-loss-based pipeline
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
        
        '''Memory mechanism (Similar to iCARL) but just random sampling instead'''
        self.protein_memory = {}  # {protein_id: [embeddings]}
        self.memory_size_per_protein = 50  # Store 50 samples per protein. Adjust as needed. Tried 100 but no significant difference.
        self.attention_history = {}  # Track attention patterns

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
        embeddings = self.encoder(attended_features)  # Encode
        
        if return_attention:
            return embeddings, attention_weights
        return embeddings

    def update_memory(self, embeddings, labels, attention_weights=None):
        """Update memory bank, store as tensors with gradient capability"""
        rng = np.random.RandomState(42) 
        embeddings_np = embeddings.detach().cpu().numpy()  # Unused right now
        labels_np = labels.cpu().numpy()
        
        for protein_id in np.unique(labels_np):
            protein_mask = labels_np == protein_id
            protein_embeddings = embeddings[protein_mask]  # keep as tensor
            
            if protein_id not in self.protein_memory:
                self.protein_memory[protein_id] = []
            
            # Add tensor samples (detached but with grad capability)
            for i, emb in enumerate(protein_embeddings):
                if len(self.protein_memory[protein_id]) < self.memory_size_per_protein:
                    # Store detached clone but ensure it's on same device and requires grad
                    memory_tensor = emb.detach().clone().requires_grad_(True)  # setting gradients
                    self.protein_memory[protein_id].append(memory_tensor)
                else:
                    idx = rng.randint(0, len(self.protein_memory[protein_id]))
                    memory_tensor = emb.detach().clone().requires_grad_(True)
                    self.protein_memory[protein_id][idx] = memory_tensor

    def sample_from_memory(self, n_triplets, device):
        """Sample triplets and return tensors with gradient computation enabled. This is sampling from memory, not current analyte"""
        rng = np.random.RandomState(42) 
        if len(self.protein_memory) < 2:  # need at least 2 ananlytes to sample triplets
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
                neg_idx = rng.randint(0, len(neg_samples))
                negative = neg_samples[neg_idx]
                
                anchors.append(anchor)
                positives.append(positive)
                negatives.append(negative)
        
        if anchors:
            # stack existing tensors. ensure they're on right device
            anchors_tensor = torch.stack(anchors).to(device)
            positives_tensor = torch.stack(positives).to(device)
            negatives_tensor = torch.stack(negatives).to(device)
            
            # ensure require gradients
            anchors_tensor.requires_grad_(True)
            positives_tensor.requires_grad_(True)
            negatives_tensor.requires_grad_(True)
            
            return anchors_tensor, positives_tensor, negatives_tensor
        return None, None, None

    def get_feature_importance(self, X, y, feature_names, device):
        """
         feature importance from attention weights as seen in figure 32 of report.
        
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
            
            # get attention weights for all samples
            _, attention_weights = self.forward(X_tensor, return_attention=True)
            attention_np = attention_weights.cpu().numpy()
            
            # vverall importance (average across all samples)
            overall_importance = attention_np.mean(axis=0)
            importance_dict['overall'] = {
                feature_names[i]: float(overall_importance[i])
                for i in range(len(feature_names))
            }
            
            # per-analyte importance
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
