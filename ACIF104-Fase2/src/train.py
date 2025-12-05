
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

from data_processing import load_data, temporal_split, create_sliding_windows, handle_imbalance
from models_ml import train_knn, train_svr, train_xgboost
from models_dl import build_lstm, build_gru, build_transformer

import tensorflow as tf
from tensorflow import keras

# Configuración
DATA_PATH = os.path.join("data", "Retail_Dataset2.csv")
MODELS_DIR = "models"
RESULTS_DIR = "results"
PLOTS_DIR = "plots"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def evaluate(y_true, y_pred, model_name):
    """
    Evalúa el rendimiento del modelo usando métricas estándar de regresión.
    
    Métricas:
    - MSE (Mean Squared Error): Penaliza errores grandes cuadráticamente.
    - RMSE (Root MSE): En las mismas unidades que la variable objetivo.
    - MAE (Mean Absolute Error): Error promedio absoluto (más robusto a outliers).
    - R2 (Coeficiente de Determinación): Qué tanto de la varianza explica el modelo (1.0 es perfecto).
    - MAPE (Mean Absolute Percentage Error): Error porcentual promedio.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    return {
        "Model": model_name,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE": mape
    }

def main():
    print("Cargando Datos...")
    # Intentamos detectar la columna de fecha si existe
    df = pd.read_csv(DATA_PATH)
    date_col = 'Date' if 'Date' in df.columns else None
    
    df_clean = load_data(DATA_PATH, date_col=date_col)
    
    # Selección de Features (Numéricas)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if "Order_Demand" in numeric_cols:
        numeric_cols.remove("Order_Demand")
    
    X = df_clean[numeric_cols].values
    y = df_clean["Order_Demand"].values
    
    # División Estratificada/Temporal (Train vs Test)
    # Importante: Respetamos el orden temporal para evitar data leakage.
    print("Dividiendo Datos (Train/Test)...")
    train_df, test_df = temporal_split(df_clean, date_col=date_col, test_size=0.2)
    
    X_train = train_df[numeric_cols].values
    y_train = train_df["Order_Demand"].values
    X_test = test_df[numeric_cols].values
    y_test = test_df["Order_Demand"].values
    
    # Escalado de Datos (Normalización Z-Score)
    # Fundamental para modelos basados en distancia (KNN, SVR) y Redes Neuronales.
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Guardamos Scaler y Nombres de Features para uso en API
    joblib.dump(scaler_X, os.path.join(MODELS_DIR, "scaler_X.pkl"))
    with open(os.path.join(MODELS_DIR, "feature_names.txt"), "w") as f:
        for name in numeric_cols:
            f.write(name + "\n")

    ml_metrics = []
    
    # --- Parte 1: Experimentos de Desbalance (usando XGBoost como sonda) ---
    print("\n--- Ejecutando Experimentos de Desbalance ---")
    imbalance_methods = ['none', 'ros', 'rus', 'smote']
    
    for method in imbalance_methods:
        print(f"Probando Método de Desbalance: {method}")
        X_res, y_res = handle_imbalance(X_train_scaled, y_train, method=method)
        
        # Entrenar XGBoost rápido para comparar impacto
        model = train_xgboost(X_res, y_res)
        y_pred = model.predict(X_test_scaled)
        
        metrics = evaluate(y_test, y_pred, f"XGBoost_{method.upper()}")
        ml_metrics.append(metrics)
        
    
    # Guardar Resultados de Desbalance
    pd.DataFrame(ml_metrics).to_csv(os.path.join(RESULTS_DIR, "imbalance_metrics.csv"), index=False)
    
    # --- Parte 2: Comparación de Modelos ML (Evaluación) ---
    print("\n--- Entrenando Regresores ML ---")
    
    # KNN
    print("Entrenando KNN...")
    knn = train_knn(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    ml_metrics.append(evaluate(y_test, y_pred, "KNN"))
    joblib.dump(knn, os.path.join(MODELS_DIR, "knn_model.pkl"))
    
    # SVR
    print("Entrenando SVR...")
    svr = train_svr(X_train_scaled, y_train)
    y_pred = svr.predict(X_test_scaled)
    ml_metrics.append(evaluate(y_test, y_pred, "SVR"))
    joblib.dump(svr, os.path.join(MODELS_DIR, "svr_model.pkl"))
    
    # Guardar Métricas ML
    ml_df = pd.DataFrame(ml_metrics)
    ml_df.to_csv(os.path.join(RESULTS_DIR, "ml_metrics.csv"), index=False)
    print("Métricas ML Guardadas.")
    
    # --- Parte 3: Deep Learning (Modelos Secuenciales) ---
    print("\n--- Entrenando Modelos Secuenciales (DL) ---")
    dl_metrics = []
    
    # Generar Ventanas Deslizantes
    # DL necesita set de Validación para EarlyStopping
    val_split_idx = int(len(X_train_scaled) * 0.8)
    X_t, y_t = X_train_scaled[:val_split_idx], y_train[:val_split_idx]
    X_v, y_v = X_train_scaled[val_split_idx:], y_train[val_split_idx:]
    
    window_size = 10
    X_train_seq, y_train_seq = create_sliding_windows(X_t, y_t, window_size)
    X_val_seq, y_val_seq = create_sliding_windows(X_v, y_v, window_size)
    X_test_seq, y_test_seq = create_sliding_windows(X_test_scaled, y_test, window_size)
    
    input_shape = (window_size, X_train_seq.shape[2])
    
    epochs = 20 # Mantener bajo para demo, aumentar en producción
    batch_size = 32
    # EarlyStopping: Detiene el entrenamiento si val_loss no mejora en 5 épocas
    callbacks = [keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    
    models_dl = {
        "LSTM": build_lstm(input_shape),
        "GRU": build_gru(input_shape),
        "Transformer": build_transformer(input_shape)
    }
    
    history_dict = {}
    
    for name, model in models_dl.items():
        print(f"Entrenando {name}...")
        history = model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        history_dict[name] = history.history
        
        y_pred = model.predict(X_test_seq).flatten()
        dl_metrics.append(evaluate(y_test_seq, y_pred, name))
        
        model.save(os.path.join(MODELS_DIR, f"{name.lower()}_model.h5"))
    
    # Guardar Métricas DL
    dl_df = pd.DataFrame(dl_metrics)
    dl_df.to_csv(os.path.join(RESULTS_DIR, "dl_metrics.csv"), index=False)
    print("Métricas DL Guardadas.")
    
    # --- Gráficos ---
    print("Generando Gráficos...")
    
    # 1. Gráfico de Comparación RMSE
    all_metrics = pd.concat([ml_df, dl_df])
    plt.figure(figsize=(12, 6))
    sns.barplot(data=all_metrics, x="Model", y="RMSE", hue="Model", palette="viridis", legend=False)
    plt.title("Comparación de Modelos por RMSE (Menor es mejor)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "model_comparison_rmse.png"))
    plt.close()
    
    # 2. Gráfico de Convergencia (Pérdida vs Épocas)
    plt.figure(figsize=(10, 6))
    for name, hist in history_dict.items():
        plt.plot(hist['loss'], label=f'{name} Train')
        plt.plot(hist['val_loss'], linestyle='--', label=f'{name} Val')
    plt.title("Convergencia de Entrenamiento DL (Loss/MSE)")
    plt.xlabel("Época")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, "convergence_plot.png"))
    plt.close()
    
    print("Proceso Finalizado.")

if __name__ == "__main__":
    main()
