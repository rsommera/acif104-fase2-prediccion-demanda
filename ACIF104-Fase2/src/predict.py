
import os
import numpy as np
import joblib
import tensorflow as tf

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "modelo_mlp_intermedio.h5")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler_X.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_names.txt")


class DemandPredictor:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"No se encontró el scaler en {SCALER_PATH}")
        if not os.path.exists(FEATURES_PATH):
            raise FileNotFoundError(f"No se encontró el archivo de features en {FEATURES_PATH}")

        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.scaler_X = joblib.load(SCALER_PATH)

        with open(FEATURES_PATH, "r") as f:
            self.feature_names = [line.strip() for line in f if line.strip()]

    def predict(self, feature_dict):
        """
        feature_dict: diccionario con {nombre_feature: valor}
        """
        x = []
        for name in self.feature_names:
            if name not in feature_dict:
                raise ValueError(f"Falta la feature requerida: {name}")
            x.append(float(feature_dict[name]))

        X_np = np.array(x, dtype=float).reshape(1, -1)
        X_scaled = self.scaler_X.transform(X_np)
        y_pred = self.model.predict(X_scaled).flatten()[0]
        return float(y_pred)
