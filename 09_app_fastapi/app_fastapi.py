import joblib
import pandas as pd
from fastapi import FastAPI, Body

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
def predict(data = Body(...)):
    
    observations = pd.DataFrame([data])
    
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
        
        
@app.get("/model-info")
def model_info():
    return{"model" : "FullXGBoostModel"}