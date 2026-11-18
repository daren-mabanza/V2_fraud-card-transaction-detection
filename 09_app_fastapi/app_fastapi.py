import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from FastAPITools.FrontendData import demo_data, business_metrics
from FastAPITools.PydanticSchemas.Schemas import (
    BatchEntree, ObservationsSortie, BatchSortie, ShapValue, ShapInfo, ShapSortie,
)

app = FastAPI(
    title="API - Detection des transactions frauduleuses",
    version="1.0.0",
)

pipeline = joblib.load("06_models/FullXGBoostPipeline_V2.joblib")
preprocessing = pipeline[:-1]
model = pipeline[-1]
explainer = joblib.load("06_models/FullXGBoostExplainer_V2.joblib")


def observations_dataframe(batch: BatchEntree) -> pd.DataFrame:
    """Convertit le lot validé en DataFrame conforme aux entrées du pipeline.

    Conserve l'ordre des variables appris au fit. Les cinq colonnes de
    catégories restent de type objet ; les autres passent en float64.
    Les valeurs manquantes deviennent des NaN, même pour une seule ligne.
    """
    observations = pd.DataFrame(
        [observation.model_dump() for observation in batch.data],
        columns=pipeline.feature_names_in_,
    )
    categories = {"categorie_marchand", "pays_marchand", "pays_ip", "canal", "mode_saisie"}
    for column in observations:
        if column in categories:
            observations[column] = observations[column].astype(object).where(
                observations[column].notna(), np.nan
            )
        else:
            observations[column] = observations[column].astype("float64")
    return observations


@app.get("/health")
def health():
    """Renvoie le statut de disponibilité de l’API, sans lancer de prédiction."""
    return {"status": "ok"}


@app.get("/demo-data")
def get_demo_data():
    """Renvoie les paiements de démonstration depuis l'export Parquet.

    Raises:
        HTTPException: Erreur 503 si l'export manque ou ne respecte pas le format attendu.
    """
    try:
        return demo_data()
    except FileNotFoundError:
        raise HTTPException(503, "Les transactions de démonstration ne sont pas encore exportées. Lancez l’étape 5.")
    except (ValueError, KeyError):
        raise HTTPException(503, "Le fichier des transactions de démonstration est incomplet ou invalide.")


@app.get("/business-metrics")
def get_business_metrics():
    """Renvoie les métriques métier et leurs moyennes depuis les exports existants.

    Raises:
        HTTPException: Erreur 503 si les exports manquent ou sont incomplets ou invalides.
    """
    try:
        return business_metrics()
    except FileNotFoundError:
        raise HTTPException(503, "Les métriques métier ne sont pas encore exportées. Lancez l’étape 4.")
    except (ValueError, KeyError):
        raise HTTPException(503, "Les exports des métriques métier sont incomplets ou invalides.")


@app.post("/batch-predict")
def predict(batch: BatchEntree) -> BatchSortie:
    """Calcule les probabilités de fraude d'un lot de 1 à 10 observations validées.

    Les entrées contiennent les 71 variables déjà calculées. Renvoie un
    ``BatchSortie`` avec les probabilités en pourcentage et des identifiants
    positionnels commençant à 1, dans l'ordre du lot reçu.
    """
    observations = observations_dataframe(batch)
    probabilites = pipeline.predict_proba(observations)[:, 1] * 100

    return BatchSortie(
        model_version="Modèle XGBoost - 2.0",
        model_nb_probas=len(observations),
        model_probas=[
            ObservationsSortie(id_observation=index, model_proba=float(proba))
            for index, proba in enumerate(probabilites, start=1)
        ],
    )


@app.post("/shap-explanation")
def shap_explanation(batch: BatchEntree) -> ShapSortie:
    """Explique un lot validé avec l'explainer SHAP chargé au démarrage.

    Renvoie, pour chaque observation, la référence, la probabilité finale,
    les dix contributions de plus forte valeur absolue et leur reste signé
    sous « Autres variables ». Les contributions portent sur les variables
    après preprocessing et sont exprimées en points de pourcentage.
    Le modèle et le preprocessing ne sont pas réentraînés.
    """
    
    observations = observations_dataframe(batch)
    observations_processed = preprocessing.transform(observations)
    shap_values = explainer(observations_processed)
    probabilites = model.predict_proba(observations_processed)[:, 1] * 100
    explanations = []

    for index in range(len(observations)):
        contributions = np.asarray(shap_values.values[index], dtype=float) * 100
        valeurs = observations_processed.iloc[index]
        ordre = np.argsort(-np.abs(contributions), kind="stable")
        features = [
            ShapValue(
                name=str(observations_processed.columns[position]),
                value=None if pd.isna(valeurs.iloc[position]) else float(valeurs.iloc[position]),
                shap_value=float(contributions[position]),
            )
            for position in ordre[:10]
        ]
        features.append(ShapValue(
            name="Autres variables",
            value=None,
            shap_value=float(contributions[ordre[10:]].sum()),
        ))
        explanations.append(ShapInfo(
            id_observation=index + 1,
            base_value=float(shap_values.base_values[index]) * 100,
            prediction=float(probabilites[index]),
            features=features,
        ))

    return ShapSortie(
        model_version="Modèle XGBoost - 2.0",
        explanations=explanations,
    )
