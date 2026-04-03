"""
Visualization utilities for clustering results
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import umap


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
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=8,
        frameon=False
    )
    
    cmap = plt.get_cmap('tab20', len(np.unique(preds_test)))

    # Plot 2: Predicted Clusters
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
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        title="Clusters",
        fontsize='small'
    )
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "umap_comparison.png"), 
                   dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_history(history, output_dir=None):
    """Plot comprehensive training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history["epoch"], history["train_loss"], 'b-', linewidth=2)
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)
    
    # ARI comparison
    axes[0, 1].plot(history["epoch"], history["train_ari"], 'g-', 
                   label="Train ARI", linewidth=2)
    axes[0, 1].plot(history["epoch"], history["val_ari"], 'r-', 
                   label="Val ARI", linewidth=2)
    axes[0, 1].plot(history["epoch"], history["test_ari"], 'b-', 
                   label="Test ARI", linewidth=2)
    axes[0, 1].set_title("ARI Scores")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("ARI")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Silhouette scores
    axes[1, 0].plot(history["epoch"], history["train_silhouette"], 'g-', 
                   label="Train Silhouette", linewidth=2)
    axes[1, 0].plot(history["epoch"], history["val_silhouette"], 'r-', 
                   label="Val Silhouette", linewidth=2)
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
    
    axes[1, 1].text(0.1, 0.9, final_text, transform=axes[1, 1].transAxes, 
                   fontfamily='monospace', verticalalignment='top', 
                   fontsize=12, linespacing=1.5)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "training_history.png"), 
                   dpi=300, bbox_inches='tight')
        print(f"Training history saved")
    
    plt.show()