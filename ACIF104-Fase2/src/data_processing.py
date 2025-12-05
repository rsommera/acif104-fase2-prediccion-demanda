
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neighbors import KNeighborsRegressor

def load_data(filepath: str, date_col: str = None):
    """
    Carga los datos desde un CSV, parsea fechas si se indica, y limpia nulos básicos en la variable objetivo.
    """
    df = pd.read_csv(filepath)
    
    # Limpiamos la variable objetivo 'Order_Demand'
    # Decisión Técnica: Se eliminan filas sin demanda porque no sirven para entrenamiento supervisado.
    if "Order_Demand" in df.columns:
        df["Order_Demand"] = pd.to_numeric(df["Order_Demand"], errors="coerce")
        df = df.dropna(subset=["Order_Demand"])
    
    # Procesamiento de fechas para Series Temporales
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        df = df.sort_values(date_col) # Ordenamos cronológicamente
        
    return df

def temporal_split(df, date_col="Date", test_size=0.2):
    """
    División temporal de los datos (Train/Test).
    
    Decisión Técnica (Criterio 2.3):
    En series de tiempo NO se puede usar random split (train_test_split aleatorio)
    porque filtraríamos información del futuro al pasado (data leakage).
    Se cortan los últimos datos (test_size) para validación.
    """
    if date_col in df.columns:
        df = df.sort_values(date_col)
    
    n = len(df)
    train_end = int(n * (1 - test_size))
    
    train = df.iloc[:train_end]
    test = df.iloc[train_end:]
    
    return train, test

def create_sliding_windows(X, y, window_size=10):
    """
    Genera ventanas deslizantes (Sliding Windows) para modelos secuenciales (LSTM/GRU).
    
    Decisión Técnica:
    Los modelos DL necesitan contexto histórico. Transformamos (N, features)
    a (N-window, window, features).
    """
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i : i + window_size])
        ys.append(y[i + window_size])
    return np.array(Xs), np.array(ys)

def handle_imbalance(X, y, method='none', bins=5, random_state=42):
    """
    Manejo de desbalance en Regresión mediante discretización.
    Métodos soportados: 'ros', 'rus', 'smote'.
    
    Decisión Técnica (Criterio 2.4 - Técnicas de Muestreo):
    Como es regresión, primero discretizamos la variable continua 'y' en bines
    para simular 'clases' y poder aplicar técnicas clásicas de re-muestreo.
    """
    if method == 'none':
        return X, y
    
    # 1. Discretizar target para formar "clases" temporales
    est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')
    y_reshaped = y.reshape(-1, 1)
    y_binned = est.fit_transform(y_reshaped).astype(int).ravel()
    
    # 2. Configurar el Sampler
    if method == 'ros':
        # Random OverSampling: Duplica muestras minoritarias.
        sampler = RandomOverSampler(random_state=random_state)
        # Truco: Muestreamos INDICES para recuperar luego los valores continuos reales de y
        indices = np.arange(len(X)).reshape(-1, 1)
        _, _ = sampler.fit_resample(indices, y_binned)
        res_indices = sampler.fit_resample(indices, y_binned)[0].ravel()
        
        return X[res_indices], y[res_indices]
        
    elif method == 'rus':
        # Random UnderSampling: Elimina muestras mayoritarias.
        sampler = RandomUnderSampler(random_state=random_state)
        indices = np.arange(len(X)).reshape(-1, 1)
        res_indices = sampler.fit_resample(indices, y_binned)[0].ravel()
        return X[res_indices], y[res_indices]
        
    elif method == 'smote':
        # SMOTE: Genera muestras SINTÉTICAS por interpolación.
        # Reto: SMOTE genera X sintético pero necesitamos un 'y' continuo correspondiente.
        
        # Validación: Necesitamos suficientes muestras por clase/bin
        min_class = np.min(np.bincount(y_binned))
        k = min(5, min_class - 1)
        if k < 1:
            print("Advertencia: No hay suficientes muestras para SMOTE. Retornando original.")
            return X, y
            
        sampler = SMOTE(random_state=random_state, k_neighbors=k)
        X_res, y_binned_res = sampler.fit_resample(X, y_binned)
        
        # Estrategia para obtener 'y' continuo de los nuevos X sintéticos:
        # Entrenamos un KNNRegressor con los datos originales para imputar el valor de y
        knn = KNeighborsRegressor(n_neighbors=3)
        knn.fit(X, y)
        
        # Predecimos y para todo el conjunto resampleado
        y_res = knn.predict(X_res)
        
        return X_res, y_res

    return X, y

