
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib


DATA_PATH = os.path.join("data", "Retail_Dataset2.csv")
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def cargar_datos():
    df = pd.read_csv(DATA_PATH)

    # Aseguramos tipo numérico
    df["Order_Demand"] = pd.to_numeric(df["Order_Demand"], errors="coerce")
    df = df.dropna(subset=["Order_Demand"])

    # Solo columnas numéricas como features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Order_Demand" in numeric_cols:
        numeric_cols.remove("Order_Demand")

    X = df[numeric_cols].values
    y = df["Order_Demand"].values

    print("Columnas usadas como features:", numeric_cols)
    print("Shape X:", X.shape, "Shape y:", y.shape)

    return X, y, numeric_cols


def build_mlp_intermedio(input_dim: int):
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def main():
    X, y, feature_names = cargar_datos()

    # División 60/20/20
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    model = build_mlp_intermedio(X_train_scaled.shape[1])

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    history = model.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=100,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1,
    )

    # Evaluación en test
    y_test_pred = model.predict(X_test_scaled).flatten()

    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100

    print("\n=== Métricas en TEST (modelo MLP intermedio) ===")
    print(f"MSE:  {mse:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"MAE:  {mae:,.2f}")
    print(f"R²:   {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")

    # Guardar modelo y scaler
    model_path = os.path.join(MODELS_DIR, "modelo_mlp_intermedio.h5")
    scaler_path = os.path.join(MODELS_DIR, "scaler_X.pkl")
    features_path = os.path.join(MODELS_DIR, "feature_names.txt")

    model.save(model_path)
    joblib.dump(scaler_X, scaler_path)
    with open(features_path, "w") as f:
        for name in feature_names:
            f.write(name + "\n")

    print(f"\n✅ Modelo guardado en: {model_path}")
    print(f"✅ Scaler guardado en: {scaler_path}")
    print(f"✅ Nombres de features guardados en: {features_path}")


if __name__ == "__main__":
    main()
