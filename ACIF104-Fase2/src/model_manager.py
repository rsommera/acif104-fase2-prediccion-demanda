
import os
import joblib
import yaml
import tensorflow as tf
from tensorflow import keras

class ModelManager:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.models_dir = self.config["paths"]["models"]
        self.default_model = self.config["models"]["default"]
        self.model_cache = {}
        self.scaler = None
        self.feature_names = None
        
        # Load shared resources
        self._load_shared_artifacts()

    def _load_shared_artifacts(self):
        scaler_path = os.path.join(self.models_dir, "scaler_X.pkl")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        
        feat_path = os.path.join(self.models_dir, "feature_names.txt")
        if os.path.exists(feat_path):
            with open(feat_path, "r") as f:
                self.feature_names = [line.strip() for line in f.readlines()]

    def list_models(self):
        # Return list of available model files in directory that match known extensions
        # Or just return what is in config if strict
        # return self.config["models"]["available"]
        # Better: check actual files
        files = os.listdir(self.models_dir)
        models = [f for f in files if f.endswith(".pkl") or f.endswith(".h5")]
        return models

    def load_model(self, model_name=None):
        if not model_name:
            model_name = self.default_model
        
        if model_name in self.model_cache:
            return self.model_cache[model_name]
        
        model_path = os.path.join(self.models_dir, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_name} not found.")
        
        print(f"Loading model: {model_name}")
        if model_name.endswith(".h5"):
            model = keras.models.load_model(model_path)
            # Keras models usually expect 3D input for LSTM/GRU or 2D for MLP? 
            # My current LSTM/GRU expect 3D (samples, window, features).
            # Need to handle input shape logic in Predictor, not just loader.
            self.model_cache[model_name] = ("dl", model)
        else:
            model = joblib.load(model_path)
            self.model_cache[model_name] = ("ml", model)
            
        return self.model_cache[model_name]

    def get_scaler(self):
        return self.scaler

    def get_feature_names(self):
        return self.feature_names
