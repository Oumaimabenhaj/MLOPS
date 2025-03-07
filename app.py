from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel

# Charger le modèle
MODEL_PATH = "xgboost_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du modèle : {e}")

# Initialiser l'API
app = FastAPI()


# Définir le format des données en entrée
class ChurnInput(BaseModel):
    features: list


@app.post("/predict")
def predict(data: ChurnInput):
    """Prédiction du churn à partir des données fournies."""
    try:
        # Convertir les données en numpy array
        input_data = np.array(data.features).reshape(1, -1)

        # Faire la prédiction
        prediction = model.predict(input_data)

        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de prédiction : {str(e)}")


@app.get("/")
def home():
    """Endpoint de test pour vérifier que l'API fonctionne."""
    return {"message": "L'API FastAPI fonctionne !"}
