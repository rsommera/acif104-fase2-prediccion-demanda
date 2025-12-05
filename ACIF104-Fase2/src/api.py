
import os
import time
import pandas as pd
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict

from .model_manager import ModelManager
from .data_processing import create_sliding_windows

# Cargar Configuración
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

app = FastAPI(
    title=config['app']['title'],
    version=config['app']['version'],
    description=config['app']['description']
)

# CORS (Permitir acceso desde frontend local)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Archivos Estáticos (Frontend)
frontend_path = os.path.abspath("frontend")
app.mount("/static", StaticFiles(directory=frontend_path, html=True), name="static")

# Montar carpeta de gráficos (Plots)
plots_path = os.path.abspath(config['paths']['plots'])
if not os.path.exists(plots_path):
    os.makedirs(plots_path)
app.mount("/plots", StaticFiles(directory=plots_path), name="plots")

# Inicializar Gestor de Modelos
manager = ModelManager()

# Middleware de Logging (Registro de peticiones)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Registro simple en archivo
    log_entry = f"{time.ctime()} | {request.method} {request.url} | Status: {response.status_code} | Time: {process_time:.4f}s\n"
    with open(config['paths']['logs'], "a") as f:
        f.write(log_entry)
        
    return response

# Esquemas de Datos (Pydantic)
class PredictRequest(BaseModel):
    features: Dict[str, float]

class PredictResponse(BaseModel):
    prediction: float
    model_used: str
    latency_ms: float

# Endpoints

@app.get("/")
def root():
    """ Sirve el Dashboard principal. """
    return FileResponse(os.path.join(frontend_path, "index.html"))

@app.get("/health")
def health_check():
    """ Verificación de estado del sistema (Monitorización). """
    return {"status": "ok", "uptime": "running"}

@app.get("/models")
def list_models():
    """ Lista todos los modelos ML y DL disponibles en la carpeta /models. """
    return {"models": manager.list_models()}

@app.get("/metrics")
def get_metrics(type: str = "ml"): # type: ml, dl, clustering
    """ 
    Obtiene métricas históricas de rendimiento (CSV).
    Permite visualizar tablas comparativas en el frontend.
    """
    results_dir = "results"
    if type == "dl":
        path = os.path.join(results_dir, "dl_metrics.csv")
    elif type == "clustering":
        path = os.path.join(results_dir, "clustering_results.csv")
    else:
        path = os.path.join(results_dir, "ml_metrics.csv")
        
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df.to_dict(orient="records")
    return {"error": "Metrics not found"}

@app.get("/features")
def get_features():
    """ Retorna la lista de features numéricas esperadas por el modelo. """
    feats = manager.get_feature_names()
    if feats:
        return {"features": feats}
    return {"features": []}

@app.get("/explain")
def get_explanation():
    """ Retorna las rutas de los gráficos de explicabilidad (SHAP). """
    plots = os.listdir(plots_path)
    shap_plots = [f"plots/{p}" for p in plots if "shap" in p]
    return {"plots": shap_plots}

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, model: str = Query(None)):
    """ Endpoint principal de inferencia (Despliegue). """
    start = time.time()
    
    # 1. Cargar Modelo Dinámicamente
    try:
        model_type, loaded_model = manager.load_model(model)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Modelo no encontrado: {e}")
    
    # 2. Preprocesar Entrada
    feature_names = manager.get_feature_names()
    scaler = manager.get_scaler()
    
    if not feature_names:
        raise HTTPException(status_code=500, detail="Nombres de features no cargados.")
    
    # Validar y ordenar features
    try:
        input_data = [payload.features.get(f, 0) for f in feature_names] # 0 es default si falta
        input_array = np.array(input_data).reshape(1, -1)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Features inválidas: {e}")
    
    # Aplicar Escalado (si existe)
    if scaler:
        input_array = scaler.transform(input_array)
        
    # 3. Predicción
    try:
        if model_type == "dl":
            # Nota Técnica: Modelos DL esperan secuencia (1, window_size, features).
            # En esta API REST simple recibimos 1 punto.
            # Solución Demo: Replicamos el input para llenar la ventana.
            # En producción: El cliente debería enviar el historial completo o la API gestionar estado.
            window_size = 10
            seq_input = np.tile(input_array, (1, window_size, 1))
            pred = loaded_model.predict(seq_input)
            prediction = float(pred[0][0])
        else:
            pred = loaded_model.predict(input_array)
            prediction = float(pred[0])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fallo en predicción: {e}")
        
    latency = (time.time() - start) * 1000
    
    return PredictResponse(prediction=prediction, model_used=model or "default", latency_ms=latency)
