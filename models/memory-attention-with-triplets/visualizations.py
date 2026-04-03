import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib import gridspec
import umap
from collections import defaultdict


class Visualizer:
    """all visualizations"""
    
    def __init__(self, base_output_dir, get_protein_name_func):
        self.base_output_dir = base_output_dir
        self.get_protein_name = get_protein_name_func
    
    def visualize_step(self, dataset_name, step_idx, result):
        """Generate comprehensive visualizations for current step"""
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
        feature_names_top = [f[:40] for f, _ in sorted_features]
        feature_scores = [s for _, s in sorted_features]
        
        y_pos = np.arange(len(feature_names_top))
        ax4.barh(y_pos, feature_scores, alpha=0.8, color='steelblue')
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(feature_names_top, fontsize=8)
        ax4.invert_yaxis()
        ax4.set_xlabel('Attention Weight')
        ax4.set_title('Top 20 Most Important Features (Overall)')
        ax4.grid(True, alpha=0.3, axis='x')
        
        # 5. Per-PTM performance
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
        
        plot_path = os.path.join(self.base_output_dir, f'step_{step_idx}_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Visualization saved: {plot_path}")
    
    def visualize_glmnet_importance(self, dataset_name, step_idx, result):
        """Create separate plot for GLMNet feature importance only. Not VIF. Older method."""
        glmnet_importance = result['evaluation']['glmnet_importance']
        
        sorted_features = sorted(
            glmnet_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:30]
        
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
        
        for i, (bar, score) in enumerate(zip(bars, glmnet_scores)):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{score:.4f}', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        
        glmnet_path = os.path.join(self.base_output_dir, f'step_{step_idx}_glmnet_importance.png')
        plt.savefig(glmnet_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    GLMNet importance plot saved: {glmnet_path}")
    
    def visualize_umap_comparison(self, embeddings, true_labels, pred_labels, step_name, step_idx):
        """UMAP visualization"""
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings_norm)
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        norm = mcolors.Normalize(vmin=np.min(true_labels), vmax=np.max(true_labels))
        
        # Plot 1: True Labels
        scatter1 = axes[0].scatter(
            embedding_2d[:, 0], embedding_2d[:, 1],
            c=true_labels, cmap='tab10', s=50, alpha=0.7,
            edgecolors='black', linewidth=0.5, norm=norm
        )

        sm = plt.cm.ScalarMappable(cmap='tab10', norm=norm)
        sm.set_array([])

        unique_labels = np.unique(true_labels)
        legend_labels = [self.get_protein_name(label) for label in unique_labels]

        legend_handles = [
            plt.Line2D([0], [0], marker='o', color='w', 
                    markerfacecolor=sm.to_rgba(label),
                    markersize=10, label=name)
            for label, name in zip(unique_labels, legend_labels)
        ]
        
        axes[0].legend(handles=legend_handles, loc='best', fontsize=10, framealpha=0.9)
        axes[0].set_title(f'True Protein Labels ({len(np.unique(true_labels))} proteins)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('UMAP 1')
        axes[0].set_ylabel('UMAP 2')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Predicted Clusters
        scatter2 = axes[1].scatter(
            embedding_2d[:, 0], embedding_2d[:, 1],
            c=pred_labels, cmap='Set2', s=50, alpha=0.7,
            edgecolors='black', linewidth=0.5
        )
        axes[1].set_title('Predicted Clusters', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('UMAP 1')
        axes[1].set_ylabel('UMAP 2')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Agreement Map
        scatter3 = axes[2].scatter(
            embedding_2d[:, 0], embedding_2d[:, 1],
            c=pred_labels, cmap='Set2', s=80, alpha=0.6, edgecolors='none'
        )
        axes[2].set_title('Agreement Map\n(Fill=Predicted, Border=True)', 
                        fontsize=14, fontweight='bold')
        axes[2].set_xlabel('UMAP 1')
        axes[2].set_ylabel('UMAP 2')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        umap_path = os.path.join(self.base_output_dir, f'step_{step_idx}_umap_comparison.png')
        plt.savefig(umap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    UMAP visualization saved: {umap_path}")
    
    def visualize_final_unseen_test(self, evaluation):
        """Create visualization for unseen test results"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        umap_data = evaluation['umap_data']
        true_labels = umap_data['test_true_labels']
        
        scatter = axes[0].scatter(umap_data['test_embeddings'][:, 0], 
                                umap_data['test_embeddings'][:, 1],
                                c=true_labels, cmap='tab10', s=7, alpha=0.2)
        
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
        
        # Performance summary
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
        
        if per_protein:
            summary_text += f"Per-Protein F1 Scores:\n"
            for protein_name, metrics in list(per_protein.items())[:6]:
                summary_text += f"• {protein_name}: {metrics.get('f1', 0):.3f}\n"

        axes[1].text(0.1, 0.95, summary_text, transform=axes[1].transAxes, 
                    fontfamily='monospace', fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.base_output_dir, 'final_unseen_test.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
