
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from data_processing import load_data

# Config
DATA_PATH = os.path.join("data", "Retail_Dataset2.csv")
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def main():
    print("Loading Data for Hierarchical Clustering...")
    df = load_data(DATA_PATH)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Order_Demand" in numeric_cols:
        numeric_cols.remove("Order_Demand")
    
    X = df[numeric_cols].values
    
    # Subsample strongly for Dendrogram (cannot plot 100k points)
    # Usually < 200 points for readable dendrogram, or calculate linkage on < 10000 but plot truncated.
    # Plan: Calculate linkage on subsample (e.g. 100) to keep it fast and readable.
    if len(X) > 100:
        print("Subsampling data to 100 points for readable Dendrogram...")
        indices = np.random.choice(len(X), 100, replace=False)
        X = X[indices]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    methods = ['single', 'complete', 'average']
    
    for method in methods:
        print(f"Running Hierarchical Clustering ({method})...")
        Z = linkage(X_scaled, method=method)
        
        plt.figure(figsize=(12, 6))
        dendrogram(Z, labels=indices if 'indices' in locals() else None)
        plt.title(f'Dendrogram ({method} linkage)')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.savefig(os.path.join(PLOTS_DIR, f"dendrogram_{method}.png"))
        plt.close()
        
    print("Done.")

if __name__ == "__main__":
    main()
