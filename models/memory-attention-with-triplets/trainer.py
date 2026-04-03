import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np
import random
import umap
import hdbscan
from sklearn.cluster import KMeans


class MemoryReplayTrainer:
    """
    Trainer that implements memory replay to prevent catastrophic forgetting.
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def train_step(self, X_train, y_train, X_val, y_val, 
                   num_epochs=1000, memory_replay_ratio=0.4, lr=1e-3):
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
        
        # track training
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
                
                # 1st loss: triplet loss
                anchor, positive, negative = self._mine_triplets(latent, yb)
                current_loss = torch.tensor(0.0, device=self.device)
                if anchor is not None:
                    current_loss = triplet_loss_fn(anchor, positive, negative)
                    epoch_current_loss += current_loss.item()
                
                # 2nd loss: memory replay loss
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
                
                # combine loss with memory. Just a basic weighted sum. Possible to explore more complex strategies.
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
        """Validation."""
        self.model.eval()
        with torch.no_grad():
            val_embeddings = self.model(X_val_tensor)
            val_embeddings_norm = F.normalize(val_embeddings, p=2, dim=1).cpu().numpy()
            
            n_clusters = len(np.unique(y_val))
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
            val_embeddings_2d = reducer.fit_transform(val_embeddings_norm)
            kmeans = hdbscan.HDBSCAN(min_cluster_size=5)
            val_preds = kmeans.fit_predict(val_embeddings_2d)
            
            val_ari = adjusted_rand_score(y_val, val_preds)
            val_nmi = normalized_mutual_info_score(y_val, val_preds)
            
        return val_ari, val_nmi
