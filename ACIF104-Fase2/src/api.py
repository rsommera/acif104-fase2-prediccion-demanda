
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .predict import DemandPredictor

app = FastAPI(
    title="API Predicción de Demanda Retail",
    version="1.0.0",
    description="API para predecir Order_Demand usando un modelo MLP entrenado.",
)

predictor = DemandPredictor()


class PredictRequest(BaseModel):
    features: dict  # {nombre_feature: valor}


class PredictResponse(BaseModel):
    prediction: float


@app.get("/")
def root():
    return {"message": "API de predicción de demanda funcionando."}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    try:
        y_pred = predictor.predict(payload.features)
        return PredictResponse(prediction=y_pred)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")
