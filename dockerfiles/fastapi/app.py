import pandas as pd
from fastapi import FastAPI, HTTPException

from src.schemas import (
    StudentFeatures,
    PredictionResponse,
    ModelInfoResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)
from src.model_loader import model_wrapper, MODEL_NAME, MODEL_ALIAS

app = FastAPI(title="Student Performance API")

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_wrapper.is_loaded}

@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    if not model_wrapper.is_loaded:
        raise HTTPException(status_code=503, detail="No hay champion registrado todavia. Corre el DAG de training primero.")
    return ModelInfoResponse(
        name=MODEL_NAME,
        version=model_wrapper.version,
        alias=MODEL_ALIAS,
        run_id=model_wrapper.run_id,
        f1_weighted=model_wrapper.f1_weighted,
        preprocessor_s3_path=model_wrapper.preprocessor_s3_path,
    )

@app.post("/predict", response_model=PredictionResponse)
def predict(features: StudentFeatures):
    try:
        df = pd.DataFrame([features.model_dump()])
        prediction = model_wrapper.predict(df)
        return PredictionResponse(
            final_grade=str(prediction[0]),
            model_version=model_wrapper.version,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(batch: BatchPredictionRequest):
    try:
        df = pd.DataFrame([s.model_dump() for s in batch.students])
        predictions = model_wrapper.predict(df)
        return BatchPredictionResponse(
            predictions=[str(p) for p in predictions],
            model_version=model_wrapper.version,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))