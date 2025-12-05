
import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from data_processing import load_data
from xgboost import XGBRegressor

# Configuración
DATA_PATH = os.path.join("data", "Retail_Dataset2.csv")
MODELS_DIR = "models"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def main():
    print("Cargando Datos para SHAP...")
    df = load_data(DATA_PATH)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Order_Demand" in numeric_cols:
        numeric_cols.remove("Order_Demand")
    
    X = df[numeric_cols].values
    
    # Cargar Scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler_X.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)
    else:
        print("Scaler no encontrado, usando datos crudos.")
        X_scaled = X

    # Cargar Nombres de Features
    features_path = os.path.join(MODELS_DIR, "feature_names.txt")
    if os.path.exists(features_path):
        with open(features_path, "r") as f:
            feature_names = [line.strip() for line in f.readlines()]
    else:
        feature_names = numeric_cols

    # SHAP es computacionalmente costoso, usamos una muestra pequeña (100 filas)
    X_sample = X_scaled[:100]
    
    # Intentar cargar XGBoost primero (producido por run_xgboost_final.py)
    xgboost_path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
    svr_path = os.path.join(MODELS_DIR, "svr_model.pkl")
    
    if os.path.exists(xgboost_path):
        print(f"Modelo XGBoost cargado desde {xgboost_path}")
        model = joblib.load(xgboost_path)
        # TreeExplainer está optimizado para modelos basados en árboles (muy rápido)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
    elif os.path.exists(svr_path):
        print(f"Modelo SVR cargado desde {svr_path}")
        model = joblib.load(svr_path)
        # KernelExplainer funciona con cualquier modelo (caja negra) pero es LENTO
        print("Usando KernelExplainer... esto puede tardar.")
        explainer = shap.KernelExplainer(model.predict, X_sample)
        shap_values = explainer.shap_values(X_sample)
    else:
        print("No se encontró modelo válido (xgboost o svr). Entrene primero.")
        return
    
    # 1. Summary Plot: Muestra impacto y dirección de cada feature en la predicción.
    print("Generando Summary Plot...")
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.savefig(os.path.join(PLOTS_DIR, "shap_summary.png"), bbox_inches='tight')
    plt.close()
    
    # 2. Bar Plot: Importancia global (valor absoluto del impacto medio).
    print("Generando Bar Plot...")
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
    plt.savefig(os.path.join(PLOTS_DIR, "shap_bar.png"), bbox_inches='tight')
    plt.close()
    
    # 3. Dependence Plot: Relación entre valor de la feature y su impacto SHAP.
    if len(feature_names) > 0:
        print("Generando Dependence Plot...")
        try:
            # Graficamos para la primera feature más importante
            shap.dependence_plot(0, shap_values, X_sample, feature_names=feature_names, show=False)
            plt.savefig(os.path.join(PLOTS_DIR, "shap_dependence.png"), bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Dependence plot falló: {e}")

    print("Gráficos SHAP Guardados.")

if __name__ == "__main__":
    main()
