import random
import numpy as np
import torch
import datetime
import warnings
warnings.filterwarnings('ignore')

from analyze import SequentialAnalysis


def set_seeds(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    set_seeds(42)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/home/rushang_phira/src/report/attention_tanh_1.0_no_benford_ALL_PL1_Unseen"
    analyzer = SequentialAnalysis(output_dir)
    

    results = analyzer.run_sequential_analysis()
    
    print(f"\noutputs saved to: {output_dir}")
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
