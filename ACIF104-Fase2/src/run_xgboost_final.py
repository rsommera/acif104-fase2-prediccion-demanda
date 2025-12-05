
import os
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from data_processing import load_data, temporal_split
from models_ml import train_xgboost
from train import evaluate

# Configuración
DATA_PATH = os.path.join("data", "Retail_Dataset2.csv")
MODELS_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    print("Entrenando Modelo XGBoost Final de Producción...")
    df = load_data(DATA_PATH)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Order_Demand" in numeric_cols:
        numeric_cols.remove("Order_Demand")
    
    # División Temporal
    train, test = temporal_split(df)
    
    X_train = train[numeric_cols].values
    y_train = train["Order_Demand"].values
    X_test = test[numeric_cols].values
    y_test = test["Order_Demand"].values
    
    # Manejo de Scaler (Debe ser consistente con training general)
    scaler_path = os.path.join(MODELS_DIR, "scaler_X.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = StandardScaler()
        scaler.fit(X_train)
        joblib.dump(scaler, scaler_path)
        
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar XGBoost
    model = train_xgboost(X_train_scaled, y_train)
    
    # Guardar Artefacto .pkl
    joblib.dump(model, os.path.join(MODELS_DIR, "xgboost_model.pkl"))
    print("Modelo XGBoost Guardado (models/xgboost_model.pkl).")
    
    # Evaluar
    y_pred = model.predict(X_test_scaled)
    metrics = evaluate(y_test, y_pred, "XGBoost_Final")
    print("Métricas Finales:", metrics)
    
    # Adjuntar métricas al archivo CSV
    metrics_path = os.path.join(RESULTS_DIR, "ml_metrics.csv")
    if os.path.exists(metrics_path):
        current_df = pd.read_csv(metrics_path)
        new_row = pd.DataFrame([metrics])
        # Concat y guardar
        current_df = pd.concat([current_df, new_row], ignore_index=True)
        current_df.to_csv(metrics_path, index=False)
    else:
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

if __name__ == "__main__":
    main()
