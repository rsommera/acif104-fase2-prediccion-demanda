
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "results"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def main():
    print("Generating Partial Plots...")
    
    # 1. Model Comparison (ML only for now)
    ml_path = os.path.join(RESULTS_DIR, "ml_metrics.csv")
    imbalance_path = os.path.join(RESULTS_DIR, "imbalance_metrics.csv")
    
    metrics = []
    if os.path.exists(ml_path):
        metrics.append(pd.read_csv(ml_path))
    if os.path.exists(imbalance_path):
        metrics.append(pd.read_csv(imbalance_path))
        
    if metrics:
        all_metrics = pd.concat(metrics)
        plt.figure(figsize=(10, 6))
        # Check if RMSE column exists
        if "RMSE" in all_metrics.columns:
            sns.barplot(data=all_metrics, x="Model", y="RMSE", palette="viridis")
            plt.title("Model Compliance: RMSE Comparison (ML Models)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, "model_comparison_rmse.png"))
            print("Generated model_comparison_rmse.png")
        else:
            print("RMSE column missing in metrics.")
            
    # 2. Convergence Plot Placeholder
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, "Deep Learning Training in Progress...\nPlease wait.", 
             horizontalalignment='center', verticalalignment='center', fontsize=20)
    plt.title("Convergence Plot")
    plt.axis('off')
    plt.savefig(os.path.join(PLOTS_DIR, "convergence_plot.png"))
    print("Generated placeholder convergence_plot.png")

if __name__ == "__main__":
    main()
