"""Entraînement et export du pipeline XGBoost V2 sur les paiements préparés."""

import yaml
import joblib

import pandas as pd

from time import perf_counter

from FraudDetection.DataModelling.ModellingParameters import (
    features_selected,
    one_hot_encoding,
    numeriques_sans_nan,
    custom_isolation_forest,
    xgboost,
    )

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector


# ------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------ 

def modelling():
    """Entraîne le pipeline XGBoost V2 et l'enregistre avec joblib.

    Lit l'échantillon dans ``11_configs/paths.yaml`` et charge le Parquet
    ``{echantillon}PaiementsCarteForModel.parquet``. L'entraînement utilise
    les paiements du 31 décembre 2023 au 1er janvier 2025 exclu, en UTC,
    avec les 71 variables sélectionnées et la cible ``fraude_confirmee``.

    Le pipeline combine encodage catégoriel, score d'anomalie et XGBoost.
    Il remplace ``06_models/{echantillon}XGBoostPipeline_V2.joblib`` et
    affiche la progression ainsi que la durée du fit complet.
    Lancer depuis la racine du projet ; aucun seuil n'est optimisé ici.

    Raises:
        FileNotFoundError: Si la configuration, les données ou le dossier de sortie manquent.
        KeyError: Si une clé de configuration ou une colonne requise manque.
        ValueError: Si les dates, les variables ou les données d'entraînement sont invalides.
    """
    
    # 0. Import des configurations
    config = yaml.safe_load(open("11_configs/paths.yaml"))
    echantillon = config["echantillon"]["type"]
    
    print("step_3_modelling - 0. Import des configurations - OK")
    
    
    
    # 1. Import des données 
    try:
        df = pd.read_parquet(f"01_data/03_processed_data/{echantillon}PaiementsCarteForModel.parquet")
        df_copy = df.copy()
        print("step_3_modelling - 1. Import des données - OK")
        
    except FileNotFoundError:
        raise FileNotFoundError("step_3_modelling - 1. Import des données - Le fichier est introuvable")
    
    
    
    # 2. Construction de la periode d'entrainement
        # Base d'entrainement : 2023-12-31 | 2025-01-01
        
    df_copy["date_paiement_utc"] = pd.to_datetime(df_copy["date_paiement_utc"], utc=True, errors="raise")

    df_train = df_copy[df_copy["date_paiement_utc"].between(pd.Timestamp("2023-12-31", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC"), inclusive="left")]

    print("step_3_modelling - 2. Construction de la periode d'entrainement - OK")

    # 3. Selection des variables d'interet

    X_train = df_train[features_selected()].copy()
    y_train = df_train["fraude_confirmee"].copy()
    
    print("step_3_modelling - 3. Selection des variables d'interet - OK")
    
    # 4. Controle minimal de la qualité des données 
        # TODO : Mettre en place une méthode de validation des données plus robuste avec Great Expectations
        
    if (len(list(X_train.columns)) == 71) & (sorted(features_selected()) == sorted(list(X_train.columns))):
        print("step_3_modelling - 4. Controle minimal de la qualité des données - 71 features : OK")
            
    else:
        raise ValueError("ERREUR : Les données ne correspondent pas à ce qui est attendu")
    
    
    # 5. Création du pipeline

    colonnes_categorielles = [
    "categorie_marchand", "pays_marchand", "pays_ip", "canal", "mode_saisie", "heure"
    ]

    ohe = ColumnTransformer(
    transformers = [
        ("one_hot_encoding", one_hot_encoding(), colonnes_categorielles),
        ],
            remainder = "passthrough",
            verbose_feature_names_out = False,
        
    ).set_output(transform="pandas")

    isolation_forest = ColumnTransformer(
    transformers = [
        ("isolation_forest", custom_isolation_forest(), numeriques_sans_nan),
        ("garder_autres_features", "passthrough", make_column_selector()),
        ],
            remainder = "passthrough",
            verbose_feature_names_out = False,
        
    ).set_output(transform="pandas")


    pipeline = Pipeline([
        ("ohe_cat", ohe),
        ("isolation_forest_num", isolation_forest),
        ("model", xgboost()),
    ])
    
    print("step_3_modelling - 5. Création du pipeline : OK")
    
    
    # 6. Entrainement du modèle

    debut_fit = perf_counter()
    pipeline.fit(X_train, y_train)
    temps_fit = perf_counter() - debut_fit

    print(
        "step_3_modelling - 6. Preprocessing et entrainement du modèle "
        f"(temps de fit : {temps_fit:.3f} s) : OK"
    )

    
    # 7. Export du modèle
    joblib.dump(
        value = pipeline,
        filename = f"06_models/{echantillon}XGBoostPipeline_V2.joblib",
    )

    print(f"step_3_modelling - 7. Export du modèle - Pipeline exporté - {echantillon}XGBoostPipeline_V2 : OK")
