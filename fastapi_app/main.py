from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import glob

app = FastAPI(title="API con Selección Dinámica de Modelos")

MODEL_DIR = "/app/models"
active_model = None
active_model_name = "Ninguno"

def get_available_models():
    """Escanea la carpeta de modelos y devuelve una lista de nombres de archivo."""
    model_files = glob.glob(os.path.join(MODEL_DIR, "*.joblib"))
    return [os.path.basename(f) for f in model_files]

@app.get("/models", tags=["Gestión de Modelos"])
async def list_available_models():
    """Devuelve una lista de todos los modelos disponibles en el volumen."""
    models = get_available_models()
    return {
        "active_model": active_model_name,
        "available_models": models
    }

@app.post("/models/{model_name}/activate", tags=["Gestión de Modelos"])
async def activate_model(model_name: str):
    """Carga un modelo específico por su nombre de archivo para que sea el 'activo'."""
    global active_model, active_model_name

    model_path = os.path.join(MODEL_DIR, model_name)

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Modelo no encontrado.")

    try:
        active_model = joblib.load(model_path)
        active_model_name = model_name
        print(f"Modelo activado: {active_model_name}")
        return {"message": f"Modelo '{model_name}' activado exitosamente."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al cargar el modelo: {e}")

class PredictionRequest(BaseModel):
    features: list[float]

@app.post("/predict", tags=["Inferencia"])
async def predict(request: PredictionRequest):
    """Realiza una predicción usando el modelo actualmente activo."""
    if active_model is None:
        raise HTTPException(status_code=503, detail="Ningún modelo está activo. Por favor, active uno primero.")

    prediction = active_model.predict([request.features])
    return {
        "prediction": prediction.tolist(),
        "model_used": active_model_name
    }
