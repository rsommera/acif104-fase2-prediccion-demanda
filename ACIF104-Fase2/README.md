# Advanced Retail Demand Prediction System (Phase 2 & 3)

Este proyecto implementa un sistema avanzado de predicción de demanda utilizando Machine Learning (ML) y Deep Learning (DL), junto con análisis de Clustering y Explicabilidad (SHAP). Incluye una API robusta y un Dashboard interactivo.

## 📋 Características

- **Modelos ML**: KNN, SVR, XGBoost (con optimización de hiperparámetros).
- **Modelos DL**: LSTM, GRU, Transformer (para series temporales secuenciales).
- **Clustering**: K-Means, K-Medoids y Jerárquico (Analítica no supervisada).
- **Explicabilidad**: SHAP (Summary, Bar, Dependence plots).
- **Manejo de Desbalance**: SMOTE, Random OverSampling, Random UnderSampling.
- **Backend API**: FastAPI con carga dinámica de modelos, logging y métricas.
- **Frontend**: Dashboard con comparación de modelos, visualización de clusters y predicción en tiempo real.

## 🛠️ Requisitos Previos

- Python 3.9 o superior.
- pip actualizado.

## 🚀 Instalación

1.  **Clonar el repositorio o descargar el código.**

    > **Importante para Reproducibilidad**: Asegúrate de que el archivo de datos `Retail_Dataset2.csv` esté ubicado dentro de la carpeta `data/` en la raíz del proyecto. El sistema espera encontrarlo en `data/Retail_Dataset2.csv`.

2.  **Crear un entorno virtual (Recomendado):**
    ```bash
    python -m venv venv
    # En Windows:
    .\venv\Scripts\activate
    # En Mac/Linux:
    source venv/bin/activate
    ```

3.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
    *Esto instalará: `fastapi`, `uvicorn`, `pandas`, `numpy`, `scikit-learn`, `tensorflow`, `xgboost`, `shap`, `imbalanced-learn`, `scipy`, `scikit-learn-extra`, `pyyaml`.*

## ⚙️ Configuración

El archivo principal de configuración es `config.yaml`:
- **app**: Puerto y host del servidor.
- **paths**: Rutas a datos (`data/`), modelos (`models/`), logs (`logs/`) y gráficos (`plots/`).
- **models**: Modelos disponibles y por defecto.

## 🏃‍♂️ Ejecución del Sistema

### 1. Entrenar Modelos (Pipeline Completo)
Si deseas reentrenar todos los modelos (ML y DL) desde cero:
```bash
python src/train.py
```
*Nota: El entrenamiento de modelos secuenciales (DL) puede tomar varios minutos.*

### 2. Generar Análisis Adicionales
Para generar los gráficos de clustering y explicabilidad:
```bash
# Clustering (K-Means, PCA, etc.)
python src/clustering_particional.py

# Explicabilidad (SHAP plots para XGBoost)
python src/shap_explain.py
```

### 3. Iniciar la Aplicación (API + Dashboard)
Para iniciar el servidor web:
```bash
uvicorn src.api:app --reload
```
O si `uvicorn` no está en el PATH:
```bash
python -m uvicorn src.api:app --reload
```

- **Dashboard**: [http://localhost:8000](http://localhost:8000)
- **Documentación API (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

## 📂 Estructura del Proyecto

```
├── config.yaml               # Configuración global
├── data/                     # Dataset (Retail_Dataset2.csv)
├── frontend/                 # Archivos estáticos del Dashboard (HTML, JS, CSS)
├── logs/                     # Logs del sistema
├── models/                   # Modelos entrenados (.pkl, .h5) y escaladores
├── plots/                    # Gráficos generados (SHAP, Clustering, Convergencia)
├── results/                  # Tablas de métricas (CSV)
├── requirements.txt          # Dependencias
└── src/
    ├── api.py                # Servidor FastAPI
    ├── clustering_*.py       # Scripts de Clustering
    ├── data_processing.py    # Pipeline de datos y manejo de desbalance
    ├── generate_partial_plots.py # Utilidad para gráficos temporales
    ├── model_manager.py      # Gestor de carga dinámica de modelos
    ├── models_dl.py          # Definición de arquitecturas DL (LSTM, etc)
    ├── models_ml.py          # Definición de modelos ML (XGBoost, etc)
    ├── run_xgboost_final.py  # Script para modelo final de producción
    ├── shap_explain.py       # Generación de explicabilidad
    └── train.py              # Pipeline de entrenamiento completo
```


## 🏗️ Arquitectura del Sistema

El sistema sigue una arquitectura monolítica modular para facilitar el despliegue y la reproducibilidad:

1.  **Backend (API + Serving)**:
    *   **Tecnología**: FastAPI (Python).
    *   **Función**: Procesamiento de datos, inferencia de modelos ML/DL y **servidor de archivos estáticos**.
    *   **Ubicación**: `src/api.py`.
    *   **Ejecución**: El backend actúa como el punto de entrada único. Al iniciarse, expone la API REST en `/predict`, `/metrics`, etc., y sirve automáticamente el Frontend en la raíz `/`.

2.  **Frontend (Dashboard)**:
    *   **Tecnología**: Vanilla HTML5, CSS3, JavaScript (ES6+). No requiere compilación (build step).
    *   **Ubicación**: Carpeta `frontend/`.
    *   **Comunicación**: Realiza peticiones asíncronas (`fetch`) al backend utilizando rutas relativas.
    *   **Ejecución**: Se sirve pasivamente a través del Backend. No requiere un servidor web separado (como Nginx o Node.js) en este entorno de desarrollo/demo.

### Instrucciones Específicas de Ejecución por Componente

Aunque se inician juntos por conveniencia, los componentes están desacoplados a nivel de código:

*   **Para ejecutar el Backend (y el sistema completo):**
    ```bash
    uvicorn src.api:app --reload
    ```

