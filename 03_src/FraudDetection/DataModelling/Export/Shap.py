import yaml
import shap
import joblib
import pandas as pd

from time import perf_counter

from Tools.DataManipulation import stratified_sample
from FraudDetection.DataModelling.ModellingParameters import features_selected
# ------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------


def export_shap():
    """Construit et exporte l'explainer SHAP du pipeline XGBoost V2 existant.

    Lit l'échantillon dans la configuration et tire 10 000 paiements du train,
    stratifiés sur ``fraude_confirmee``. Le train va du 31 décembre 2023
    au 1er janvier 2025 exclu, en UTC. Le preprocessing déjà ajusté transforme
    ces lignes ; le classifieur n'est pas réentraîné.

    Le TreeExplainer utilise cette référence pour expliquer les probabilités
    et remplace ``06_models/{echantillon}XGBoostExplainer_V2.joblib``.
    Lancer depuis la racine du projet, avec les données et le pipeline disponibles.
    """

    # 0. Import des configurations
    config = yaml.safe_load(open("11_configs/paths.yaml"))
    echantillon = config["echantillon"]["type"]

    print("step_6_export_shap - 0. Import des configurations - OK")

    # 1 - Import des artefacts et chargement des données
    pipeline = joblib.load(f"06_models/{echantillon}XGBoostPipeline_V2.joblib")

    data = pd.read_parquet(f"01_data/03_processed_data/{echantillon}PaiementsCarteForModel.parquet")
    df_copy = data.copy()

    print("step_6_export_shap - 1. Import des artefacts et chargement des données - OK")


    # 2 - Filtrage des données
    mask = df_copy["date_paiement_utc"].between(pd.Timestamp("2023-12-31", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC"), inclusive="left")

    df_copy = df_copy[mask]

    sample_data = stratified_sample(
        df = df_copy,
        columns = ["fraude_confirmee"],
        n = 10_000,
        random_state = 123
    )

    sample_train_data = sample_data[features_selected()+["fraude_confirmee"]]

    X_train = sample_train_data.drop(["fraude_confirmee"], axis=1)

    print("step_6_export_shap - 2. Filtrage des données - OK")


    # 3 - Transformation et entrainement des données
    preprocessing = pipeline[:-1]
    model = pipeline[-1]

    debut_fit = perf_counter()
    X_train_processed = preprocessing.transform(X_train)
    temps_fit = perf_counter() - debut_fit

    print(f"step_6_export_shap - 3. Transformation des données - Temps de transformation : {temps_fit:.3f} s - OK")


    # 4 - Création de l'explainer SHAP
    background = shap.sample(
    X_train_processed,
    10_000,
    random_state=42,
    )

    masker = shap.maskers.Independent(
        background,
        max_samples = 10_000,
    )

    explainer = shap.TreeExplainer(
        model = model,
        data = masker,
        model_output = "probability",
        feature_perturbation = "auto",
    )

    print("step_6_export_shap - 4. Création de l'explainer SHAP - OK")


    # 5 - Export de l'explainer SHAP
    joblib.dump(
        value = explainer,
        filename = f"06_models/{echantillon}XGBoostExplainer_V2.joblib",
        )

    print(f"step_6_export_shap - # 5 - Export de l'explainer SHAP - {echantillon}XGBoostExplainer_V2.joblib - OK")
