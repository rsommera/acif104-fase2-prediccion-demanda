# ACIF104 – Fase 2  
# Predicción de Demanda Retail (ML + DL)

Alumno: Ricardo Sommer  
Curso: ACIF104 – Aprendizaje Automático  
Institución: Universidad Andrés Bello  
Evaluación: Fase 2 – Proyecto Predictivo con ML/DL + API + Frontend

---

##  Descripción General del Proyecto
El objetivo de este proyecto es desarrollar un sistema completo de predicción de demanda (`Order_Demand`) en Retail utilizando técnicas de ML y DL, incluyendo EDA, modelamiento, comparaciones, explicabilidad SHAP, API en FastAPI y frontend.

---

##  Dataset
Se utiliza el archivo `Retail_Dataset2.csv` ubicado en `/data/`.


## 🤖 Técnicas Evaluadas
### ML:
- Regresión Lineal  
- Random Forest  
- Gradient Boosting  

### DL:
- MLP Simple  
- MLP Intermedio (Modelo Final)  
- MLP Complejo  

---

##  Modelo Seleccionado
**MLP Intermedio**  
- Capas: 64 → 32 → 1  
- Activación: ReLU  
- Optimizador: Adam  
- Pérdida: MSE  
- EarlyStopping activado

---

##  Backend – FastAPI
Endpoints:

| Método | Ruta       | Descripción |
|--------|------------|-------------|
| GET    | `/`        | Estado      |
| POST   | `/predict` | Predicción  |

Ejecutar:
```
python -m uvicorn src.api:app --reload
```

---

##  Frontend
Archivos en `/frontend`:
- index.html  
- app.js  
- styles.css  

La interfaz consume la API local.

---

## 🛠️ Instalación Local
### 1. Clonar repo
```
git clone https://github.com/tuusuario/acif104-fase2-prediccion-demanda.git
```

### 2. Crear entorno
Windows:
```
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar dependencias
```
pip install -r requirements.txt
```

### 4. Entrenar modelo
```
python src/train.py
```

### 5. Levantar API
```
python -m uvicorn src.api:app --reload
```

---

##  Estructura
```
acif104-fase2-prediccion-demanda/
│ data/
│ models/
│ notebooks/
│ src/
│ frontend/
│ images/
│ requirements.txt
│ README.md
```


##  Contacto
Alumno: Ricardo Sommer  
Universidad Andrés Bello
