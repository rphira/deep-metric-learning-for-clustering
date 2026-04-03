"""
Training function
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import hdbscan

from model import TripletModel


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
    """
    Train model
    """
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
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    
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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                       worker_init_fn=seed_worker, generator=g)

    model = TripletModel(features.shape[1], hidden_dim, latent_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    triplet_loss_fn = nn.TripletMarginLoss(margin=margin)

    # track history
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

    print(f"Splits. Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    print(f"device: {device}")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for xb, yb in loader:
            optimizer.zero_grad()
            latent = model(xb)
            
            # triplet mining
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

        # Early stopping
        model.eval()
        with torch.no_grad():
            # Train set evaluation
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
        
        # Early stopping
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
            print(f"Early stopping at epoch {epoch}. Best val ARI: {best_val_ari:.4f}")
            break

    # Load best model
    model.load_state_dict(best_model_state)

    # Final evaluation on held out test set
    model.eval()
    with torch.no_grad():
        latent_test = model(X_test_tensor)
        latent_test_norm = F.normalize(latent_test, p=2, dim=1).cpu().numpy()
        preds_test = hdbscan.HDBSCAN(min_cluster_size=10).fit_predict(latent_test_norm)
        
        test_ari = adjusted_rand_score(y_test, preds_test)
        test_nmi = normalized_mutual_info_score(y_test, preds_test)

    history["test_ari"] = [test_ari] * len(history["epoch"])
    history["test_nmi"] = [test_nmi] * len(history["epoch"])

    print("Final results:")
    print(f"Final Test ARI: {test_ari:.4f}")
    print(f"Final Test NMI: {test_nmi:.4f}")
    for i, label in enumerate(le.classes_):
        print(f"{i} → {label}")
    
    return {
        'model': model,
        'history': history,
        'preds_test': preds_test,
        'best_feature_scores': best_feature_scores,
        'label_encoder': le,
        'scaler': scaler,
        'test_data': {
            'X_test_scaled': X_test_scaled,
            'y_test': y_test,
            'latent_test_norm': latent_test_norm
        }
    }