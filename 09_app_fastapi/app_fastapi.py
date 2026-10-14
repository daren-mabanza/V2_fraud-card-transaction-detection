import joblib
import pandas as pd
from fastapi import FastAPI
from FastAPITools.PydanticSchemas.Schemas import PredictionInput, ModelInfoOutput

app = FastAPI(
    title = "API - Detection des transactions frauduleuses",
    version = "1.0.0",
)


model = joblib.load("06_models/FullXGBoostModel_V1.joblib")
cost_optimizer = joblib.load("07_artifacts/FullCostOptimizer.joblib")


@app.get("/health")
def health():
    return {"status" : "ok"}


@app.post("/predict")
def predict(data: PredictionInput):
    
    validated_data = data.model_dump()
    
    observations = pd.DataFrame([validated_data])
    
    probabilite = model.predict_proba(observations)[:,1]
    prediction = (probabilite > cost_optimizer.get_best_threshold()).astype(int)
    
    probabilite = round(float(probabilite[0])*100,2)
    prediction = int(prediction[0])

    
    if prediction == 1:
        return {
            "message" : "La transaction a été refusé",
            "prédiction" : prediction,
            "probabilité" : probabilite,
        }
        
    else :
        return {
            "message" : "La transaction a été accepté",
            "prédiction" : prediction,
            "probabilité" : probabilite, 
        }
        
        
@app.get("/model-info", response_model=ModelInfoOutput)
def model_info():
    return {
        "model": "FullXGBoostModel",
        "version": "1.1",
        "seuil": round(cost_optimizer.get_best_threshold(),3),
        "variables": ["amt", "heure", "jour", "category", "state"],
        "model_path": "06_models/FullXGBoostModel_V1.joblib",
        "cost_optimizer_path": "07_artifacts/FullCostOptimizer.joblib"
        
        }