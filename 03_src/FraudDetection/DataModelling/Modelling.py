"""Workflow de modélisation : préparation, entraînement, évaluation et export."""

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
    """Exécute le workflow de détection de fraude et exporte le pipeline entraîné.

    Lit le type d'échantillon dans ``11_configs/paths.yaml``, puis charge
    ``01_data/03_processed_data/{echantillon}DataForModel.parquet``.
    Les chemins sont relatifs au répertoire de travail : lancer cette
    fonction depuis la racine du projet.

    Le découpage chronologique réserve 2019 à l'entraînement, janvier à
    septembre 2020 à la validation et octobre à décembre 2020 au test.
    Les variables d'entrée sont amt, heure, jour, category et state ;
    is_fraud contient la cible binaire et trans_date_trans_time la date.

    Le preprocessing construit les niveaux de risque hors pli sur le
    train, les encode en indicatrices et ajoute un score d'anomalie.
    XGBoost apprend ensuite sur ces données préparées. Les temps du
    preprocessing et du fit du classifieur sont mesurés séparément.
    Le seuil minimisant le coût des erreurs est choisi sur la validation,
    puis les métriques sont affichées pour les trois jeux.

    Returns
    -------
    None
        Affiche l'avancement, les durées, le seuil et les métriques.
        Écrit le pipeline entraîné avec joblib dans
        ``06_models/{echantillon}XGBoostModel_V1.joblib``.

    Raises
    ------
    FileNotFoundError
        Si la configuration, les données ou le dossier d'export manquent.
    KeyError
        Si une clé de configuration ou une colonne requise manque.
    ValueError
        Si les dates, les données ou les contrôles de dimensions sont
        invalides. Le workflow attend 5 variables avant preprocessing
        et 18 après preprocessing.

    Notes
    -----
    Le fichier joblib existant au même emplacement est remplacé.
    Le seuil optimisé et les durées sont affichés mais ne sont ni
    retournés ni enregistrés dans ce fichier, qui contient le pipeline.
    Les décisions des métriques utilisent ``probabilité > seuil`` ;
    l'optimiseur de coût utilise ``probabilité >= seuil``.
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
