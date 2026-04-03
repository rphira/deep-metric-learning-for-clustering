"""
Main script
"""
import os
import json
from datetime import datetime
import warnings
import pandas as pd
import numpy as np
from utils import set_seed, save_results
from preprocessing import preprocess_data, remove_redundant_features
from trainer import train_with_triplet
from cluster_analyze import ClusterAnalyze
from visualize import create_umap_plots, plot_training_history
from feature_importance import calculate_feature_importance

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    # Set random seed for reproducibility
    set_seed(42)
    
    # Set output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/home/rushang_phira/src/report/some_name_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data")
    k8_k12_k16 = pd.read_csv("/home/rushang_phira/src/data/complete_feature_sets/all_ladders_tsfresh.csv")
    k8_k12_k16 = k8_k12_k16.rename(columns={'label': 'final_label'}).replace(
        [float('inf'), float('-inf')], float('nan')
    )
    
    print("Preprocessing data")
    cleaned_data = preprocess_data(k8_k12_k16, label_column='final_label')
    cleaned_data = remove_redundant_features(cleaned_data, label_column='final_label')
    
    # outlier removal(uncomment when needed. We do not do it for smaller datasets)
    # cleaned_data = remove_outliers_by_label(cleaned_data, 'final_label', contamination=0.3)
    
    # Downsampling
    # cleaned_data = stratified_downsample(cleaned_data, 'final_label', samples_per_class=6500)
    
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
        "num_features": len(feature_names)
    }
    
    json_path = os.path.join(output_dir, f"training_metadata_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(training_info, f, indent=4)
    
    print(f"Training on {len(X)} samples with {len(feature_names)} features")
    print(f"Number of classes: {n_clusters}")
    
    # Train model
    print("\nBeginning training...")
    train_results = train_with_triplet(
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
    
    # Extract results
    model = train_results['model']
    history = train_results['history']
    preds_test = train_results['preds_test']
    best_feature_scores = train_results['best_feature_scores']
    le = train_results['label_encoder']
    test_data = train_results['test_data']
    
    # cluster analysis
    print("\ncluster analysis")
    analyze = ClusterAnalyze()
    test_analysis = analyze.analyze_clustering(
        features=test_data['X_test_scaled'],
        le=le,
        true_labels=test_data['y_test'],
        pred_labels=preds_test,
        latent_embeddings=test_data['latent_test_norm'],
        feature_names=feature_names,
        dataset_name="Test_Set"
    )
    
    # Create analysis report
    test_report_path = os.path.join(output_dir, "test_set_analysis.png")
    analyze.create_report(test_analysis, test_report_path)
    
    # Create UMAP visualizations
    print("\nhere doing UMAPs")
    create_umap_plots(
        test_data['latent_test_norm'],
        test_data['y_test'],
        preds_test,
        le,
        output_dir
    )
    
    # feature importance
    print("\nhere calculating feature importance")
    device = next(model.parameters()).device
    importance_df = calculate_feature_importance(
        model, X, feature_names, device, best_feature_scores
    )
    
    # training history
    print("\nhere plotting training history")
    plot_training_history(history, output_dir=output_dir)
    
    # zave results
    print("\nhere saving results")
    save_results(output_dir, preds_test, importance_df, model, history)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()