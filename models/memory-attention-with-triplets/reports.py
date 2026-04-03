import os
import json
import numpy as np
import matplotlib.pyplot as plt
import umap
from collections import defaultdict


class ReportGenerator:
    """Generate reports and plots"""
    
    def __init__(self, base_output_dir, get_protein_name_func):
        self.base_output_dir = base_output_dir
        self.get_protein_name = get_protein_name_func
    
    def generate_final_report(self, results, evolution_history):
        """Generate final report."""
        print("HERE GENERATING FINAL REPORT")
        
        self.plot_evolution(results)
        self.plot_umap_evolution(results)
        self.analyze_feature_consistency(results)
        self.create_summary_report(results)
        
        print(f"All results saved to: {self.base_output_dir}")
    
    def plot_evolution(self, results):
        """Plot performance evolution across steps"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        steps = [m['step'] for k, m in results.items() if 'step' in m]
        step_names = [k for k in results.keys() if 'step' in results[k]]
        
        train_ari = [results[k]['evaluation']['train_ari'] for k in step_names]
        test_ari = [results[k]['evaluation']['test_ari'] for k in step_names]
        train_nmi = [results[k]['evaluation']['train_nmi'] for k in step_names]
        test_nmi = [results[k]['evaluation']['test_nmi'] for k in step_names]
        
        # ARI evolution
        axes[0, 0].plot(steps, train_ari, 'o-', label='Train', linewidth=2, markersize=8)
        axes[0, 0].plot(steps, test_ari, 's-', label='Test', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('ARI Score')
        axes[0, 0].set_title('ARI Evolution Across Steps')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 1)
        
        # NMI evolution
        axes[0, 1].plot(steps, train_nmi, 'o-', label='Train', linewidth=2, markersize=8)
        axes[0, 1].plot(steps, test_nmi, 's-', label='Test', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('NMI Score')
        axes[0, 1].set_title('NMI Evolution Across Steps')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1)
        
        # Plotting train and test ARIs to check generalization quality of the model
        axes[1, 0].plot(steps, np.array(train_ari) - np.array(test_ari), 
                       'o-', linewidth=2, markersize=8, color='red')
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Train - Test ARI')
        axes[1, 0].set_title('Generalization Gap')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Number of Analyte PTMs
        n_proteins = [results[k]['n_proteins'] for k in step_names]
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
        
        print(f"evolution plot saved to: {evolution_path}")

    def plot_umap_evolution(self, results):
        """Create UMAP evolution plot showing all steps"""
        print("\n  Creating UMAP evolution visualization...")
        
        step_names = [k for k in results.keys() if 'step' in results[k]]
        n_steps = len(step_names)
        fig = plt.figure(figsize=(20, 4 * ((n_steps + 1) // 2)))
        
        for idx, step_name in enumerate(step_names, start=1):
            result = results[step_name]
            umap_data = result['evaluation']['umap_data']
            
            embeddings = umap_data['test_embeddings']
            true_labels = umap_data['test_true_labels']
            pred_labels = umap_data['test_pred_labels']
            
            embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
            embedding_2d = reducer.fit_transform(embeddings_norm)
            
            ax1 = plt.subplot(n_steps, 2, 2*idx - 1)
            ax2 = plt.subplot(n_steps, 2, 2*idx)
            
            # true labels
            scatter1 = ax1.scatter(
                embedding_2d[:, 0], embedding_2d[:, 1],
                c=true_labels, cmap='tab10', s=7, alpha=0.5
            )
            ax1.set_title(f'Step {idx}: True Labels', fontsize=12, fontweight='bold')
            ax1.set_xlabel('UMAP 1')
            ax1.set_ylabel('UMAP 2')
            ax1.grid(True, alpha=0.3)
            
            # cluster predictions
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

    def analyze_feature_consistency(self, results):
        """Analyze which features are consistently important. It is just checking which features have high attention weights across all steps."""
        print("\n  Analyzing feature consistency")
        
        step_names = [k for k in results.keys() if 'step' in results[k]]
        all_features = results[step_names[0]]['feature_names']
        feature_scores_across_steps = defaultdict(list)
        
        for step_name in step_names:
            result = results[step_name]
            importance = result['feature_importance']['overall']
            for feature in all_features:
                feature_scores_across_steps[feature].append(importance.get(feature, 0))
        
        # check consistency metrics
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
        
        print("\n TOP 30 MOST CONSISTENTLY IMPORTANT FEATURES:")
        print(f"{'Rank':<6}{'Feature':<60}{'Mean':<10}{'Std':<10}{'Consistency':<12}")
    
        
        for i, (feature, metrics) in enumerate(ranked_features[:30], 1):
            print(f"  {i:<6}{feature[:58]:<60}{metrics['mean_importance']:<10.4f}"
                  f"{metrics['std_importance']:<10.4f}{metrics['consistency_score']:<12.2f}")
        

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
        
        
        # Visualize top features
        self.visualize_top_features(ranked_features[:30], results)
    
    def visualize_top_features(self, top_features, results):
        """Visualize top important features"""
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        feature_names = [f[:40] for f, _ in top_features[:20]]
        consistency_scores = [m['consistency_score'] for _, m in top_features[:20]]
        
        # Plot 1: Consistency scores
        y_pos = np.arange(len(feature_names))
        axes[0].barh(y_pos, consistency_scores, alpha=0.8, color='steelblue')
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(feature_names, fontsize=9)
        axes[0].invert_yaxis()
        axes[0].set_xlabel('Consistency Score (Mean/Std)')
        axes[0].set_title('Top 20 Most Consistent Features Across All Steps')
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Plot 2: Heatmap
        step_names = [k for k in results.keys() if 'step' in results[k]]
        step_labels = [f"Step {i+1}" for i in range(len(step_names))]
        importance_matrix = np.array([
            top_features[i][1]['scores_per_step'] 
            for i in range(min(20, len(top_features)))
        ])
        
        im = axes[1].imshow(importance_matrix, aspect='auto', cmap='YlOrRd')
        axes[1].set_yticks(np.arange(len(feature_names)))
        axes[1].set_yticklabels(feature_names, fontsize=9)
        axes[1].set_xticks(np.arange(len(step_labels)))
        axes[1].set_xticklabels(step_labels)
        axes[1].set_title('Feature Importance Heatmap Across Steps')
        plt.colorbar(im, ax=axes[1], label='Attention Weight')
        
        plt.tight_layout()
        
        feature_viz_path = os.path.join(self.base_output_dir, 'feature_consistency_analysis.png')
        plt.savefig(feature_viz_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_summary_report(self, results):
        """A text summary report for a quick overview of performance"""
        print("\n  Creating summary report...")
        
        summary_lines = []
        summary_lines.append("SEQUENTIAL PROTEIN IDENTIFICATION - FINAL SUMMARY")

        
        summary_lines.append("OVERALL PERFORMANCE:")
        
        step_names = [k for k in results.keys() if 'step' in results[k]]
        for step_name in step_names:
            result = results[step_name]
            eval_metrics = result['evaluation']
            summary_lines.append(f"\n{step_name}:")
            summary_lines.append(f"  Proteins: {result['n_proteins']}")
            summary_lines.append(f"  Train ARI: {eval_metrics['train_ari']:.4f}")
            summary_lines.append(f"  Test ARI:  {eval_metrics['test_ari']:.4f}")
            summary_lines.append(f"  Test NMI:  {eval_metrics['test_nmi']:.4f}")
            summary_lines.append(f"  Test Silhouette: {eval_metrics['test_silhouette']:.4f}")
        
        # Final step analysis
        final_step_name = step_names[-1]
        final_result = results[final_step_name]
        
        # some extra beautification
        summary_lines.append("\n")

        summary_lines.append(f"FINAL STEP DETAILED ANALYSIS ({final_step_name}):")

        
        per_protein = final_result['evaluation']['per_protein']
        summary_lines.append("\nPer-Analyte PTM Performance:")
        summary_lines.append(f"{'Analyte PTM':<15}{'Precision':<12}{'Recall':<12}{'F1 Score':<12}{'Samples':<10}")
    
        
        for protein_id, metrics in per_protein.items():
            summary_lines.append(
                f"{protein_id:<15}"
                f"{metrics['precision']:<12.4f}"
                f"{metrics['recall']:<12.4f}"
                f"{metrics['f1']:<12.4f}"
                f"{metrics['n_samples']:<10}"
            )

        
        test_aris = [results[k]['evaluation']['test_ari'] for k in step_names]
        initial_ari = test_aris[0]
        final_ari = test_aris[-1]
        ari_change = final_ari - initial_ari
        
        summary_lines.append(f"\nInitial Test ARI (Step 1): {initial_ari:.4f}")
        summary_lines.append(f"Final Test ARI (Step {len(test_aris)}): {final_ari:.4f}")
        summary_lines.append(f"Change: {ari_change:+.4f}")
        summary_lines.append("\n")
        
        report_path = os.path.join(self.base_output_dir, 'final_summary_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        print("\n" + '\n'.join(summary_lines))
        print(f"\nSummary report saved: {report_path}")
