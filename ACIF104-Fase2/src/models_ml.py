
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

def get_knn_model():
    return KNeighborsRegressor()

def get_svr_model():
    return SVR()

def get_xgboost_model():
    return XGBRegressor(objective='reg:squarederror')

def train_knn(X_train, y_train):
    """
    Entrena KNN buscando el mejor 'n_neighbors' (vecinos) aleatoriamente.
    """
    model = KNeighborsRegressor()
    # Rango de vecinos a explorar
    param_dist = {'n_neighbors': np.arange(3, 20)}
    
    # RandomizedSearchCV: Busca la mejor combinación sin probar TODAS (más rápido que GridSearch)
    search = RandomizedSearchCV(model, param_dist, n_iter=5, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
    search.fit(X_train, y_train)
    print(f"Mejores parámetros KNN: {search.best_params_}")
    return search.best_estimator_

def train_svr(X_train, y_train):
    """
    Entrena SVR (Support Vector Regression).
    Decisión Técnica: SVR escala cubicamente O(N^3), es muy lento con muchos datos.
    Si N > 5000, hacemos tuning sobre una sub-muestra para no bloquear el sistema.
    """
    if len(X_train) > 5000:
        indices = np.random.choice(len(X_train), 5000, replace=False)
        X_tune, y_tune = X_train[indices], y_train[indices]
    else:
        X_tune, y_tune = X_train, y_train

    model = SVR()
    # Hiperparámetros clave SVR:
    # C: Penalización de error (Regularización).
    # epsilon: Margen de tolerancia donde no se penaliza error.
    # kernel: Tipo de transformación (RBF es estándar para no linealidad).
    param_dist = {
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.5],
        'kernel': ['rbf', 'linear']
    }
    search = RandomizedSearchCV(model, param_dist, n_iter=5, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
    search.fit(X_tune, y_tune)
    
    # Re-entrenamos el mejor modelo con TODOS los datos
    best_model = SVR(**search.best_params_)
    best_model.fit(X_train, y_train)
    print(f"Mejores parámetros SVR: {search.best_params_}")
    return best_model

def train_xgboost(X_train, y_train):
    """
    Entrena XGBoost (Gradient Boosting optimizado).
    Ideal para competencias por su velocidad y rendimiento.
    """
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    
    # Tuning de learning rate (paso de aprendizaje) y profundidad del árbol
    param_dist = {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 200]
    }
    search = RandomizedSearchCV(model, param_dist, n_iter=5, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
    search.fit(X_train, y_train)
    print(f"Mejores parámetros XGBoost: {search.best_params_}")
    return search.best_estimator_
