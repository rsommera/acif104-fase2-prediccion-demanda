
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from data_processing import load_data

try:
    from sklearn_extra.cluster import KMedoids
    HAS_KMEDOIDS = True
except ImportError:
    HAS_KMEDOIDS = False
    print("Aviso: scikit-learn-extra no encontrado. Se saltará KMedoids.")

# Config
DATA_PATH = os.path.join("data", "Retail_Dataset2.csv")
RESULTS_DIR = "results"
PLOTS_DIR = "plots"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def main():
    print("Cargando datos para clustering...")
    df = load_data(DATA_PATH)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Order_Demand" in numeric_cols:
        # Decisión Técnica:
        # Generalmente clusterizamos usando solo las FEATURES para encontrar segmentos naturales
        # sin sesgarnos por la variable objetivo (Demanda).
        numeric_cols.remove("Order_Demand")
    
    X = df[numeric_cols].values
    
    # Subsampleo si los datos son demasiados (> 5000)
    # Silhouette Score y KMedoids son O(N^2), muy lentos con Big Data.
    if len(X) > 5000:
        print("Sub-muestreando a 5000 puntos para optimizar velocidad de Clustering...")
        indices = np.random.choice(len(X), 5000, replace=False)
        X = X[indices]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # --- Método del Codo (Elbow Method) ---
    # Busca el punto de inflexión donde aumentar K ya no reduce significativamente la inercia (varianza).
    print("Ejecutando Método del Codo (Elbow)...")
    inertias = []
    K_range = range(2, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertias, 'bo-')
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Inercia')
    plt.title('Método del Codo para k óptimo')
    plt.savefig(os.path.join(PLOTS_DIR, "clustering_elbow.png"))
    plt.close()
    
    # --- Silhouette Score ---
    # Mide qué tan similar es un objeto a su propio cluster comparado con otros clusters.
    # Rango [-1, 1]. Mayor es mejor.
    print("Calculando Silhouette Scores...")
    sil_scores = []
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        sil_scores.append(score)
        
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, sil_scores, 'go-')
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores por k')
    plt.savefig(os.path.join(PLOTS_DIR, "clustering_silhouette.png"))
    plt.close()
    
    # Elegimos K óptimo (máximo silhoutte)
    best_k = K_range[np.argmax(sil_scores)]
    print(f"K Óptimo basado en Silhouette: {best_k}")
    
    # --- Clustering Final (KMeans) ---
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    
    # --- Visualización PCA (Reducción de Dimensionalidad) ---
    # Reducimos de N dimensiones a 2 para poder graficar en un plano XY.
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_labels, palette="viridis", s=50)
    plt.title(f'K-Means Clustering (k={best_k}) - PCA')
    plt.savefig(os.path.join(PLOTS_DIR, "clustering_kmeans_pca.png"))
    plt.close()
    
    # --- K-Medoids (PAM) ---
    # K-Medoids usa puntos reales como centros (medoides), es más robusto a outliers que K-Means.
    if HAS_KMEDOIDS:
        print(f"Ejecutando K-Medoids (k={best_k})...")
        try:
            kmedoids = KMedoids(n_clusters=best_k, method='pam', random_state=42)
            kmedoids_labels = kmedoids.fit_predict(X_scaled)
            
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmedoids_labels, palette="plasma", s=50)
            plt.title(f'K-Medoids (PAM) Clustering (k={best_k}) - PCA')
            plt.savefig(os.path.join(PLOTS_DIR, "clustering_kmedoids_pca.png"))
            plt.close()
        except Exception as e:
            print(f"Fallo en KMedoids: {e}")
            
    # Guardar métricas en CSV
    results = {
        "K": K_range,
        "Inertia": inertias,
        "Silhouette": sil_scores
    }
    pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR, "clustering_results.csv"), index=False)
    print("Resultados de Clustering guardados.")

if __name__ == "__main__":
    main()
