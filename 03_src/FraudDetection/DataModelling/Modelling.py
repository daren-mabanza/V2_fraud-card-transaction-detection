"""Workflow de modélisation : préparation, entraînement, évaluation et export."""

import yaml
import joblib

import numpy as np
import pandas as pd

from time import perf_counter

from FraudDetection.DataModelling.ModellingParameters import (
    risk_level_encoder, one_hot_encoder, custom_isolation_forest,
    xgboost, threshold_cost_optimizer
)

from FraudDetection.DataModelling.ModellingTools import (
    display_roc_auc_score, display_precision_score,
    display_recall_score, display_log_loss
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
        df = pd.read_parquet(f"01_data/03_processed_data/{echantillon}DataForModel.parquet")
        df_copy = df.copy()
        print("step_3_modelling - 1. Import des données - OK")
        
    except FileNotFoundError:
        raise FileNotFoundError("step_3_modelling - 1. Import des données - Le fichier est introuvable")
    
    
    
    # 2. Découpage chronologique et construction des échantillons (Train - Validation - Test)
        # Train : 2019
        # Validation : janvier–septembre 2020
        # Test : octobre–décembre 2020
        
    df_copy["trans_date_trans_time"] = pd.to_datetime(df_copy["trans_date_trans_time"], errors="raise")
    df_copy = df_copy.sort_values("trans_date_trans_time", kind="stable")
    
    masque_train = df_copy["trans_date_trans_time"].between("2019-01-01", "2020-01-01", inclusive="left")
    masque_validation = df_copy["trans_date_trans_time"].between("2020-01-01", "2020-10-01", inclusive="left")
    masque_test = df_copy["trans_date_trans_time"].between("2020-10-01", "2021-01-01", inclusive="left")
    
    df_train = df_copy.loc[masque_train].copy()
    df_validation = df_copy.loc[masque_validation].copy()
    df_test = df_copy.loc[masque_test].copy()
    
    colonnes_entree = ["amt", "heure", "jour", "category", "state"]

    X_train = df_train[colonnes_entree].copy()
    y_train = df_train["is_fraud"].copy()

    X_validation = df_validation[colonnes_entree].copy()
    y_validation = df_validation["is_fraud"].copy()

    X_test = df_test[colonnes_entree].copy()
    y_test = df_test["is_fraud"].copy()
    
    print("step_3_modelling - 3. Découpage chronologique et construction des échantillons - OK")
    
    
    
    # 4. Controle minimal de la qualité des données 
        # TODO : Mettre en place une méthode de validation des données plus robuste avec Great Expectations
        
    if (len(list(X_train.columns)) == 5) & (sorted(colonnes_entree) == sorted(list(X_train.columns))):
        print("step_3_modelling - 4. Controle minimal de la qualité des données - 5 features : OK")
            
    else:
        raise ValueError("ERREUR : Les données ne correspondent pas à ce qui est attendu")
    
    
    
    # 5. Création du pipeline    
    
    encoding = ColumnTransformer(
        transformers = [
            ("heure_jour_category_state", risk_level_encoder(), ["heure","jour","category","state"]),
            ],  
                remainder = "passthrough",
                verbose_feature_names_out = False,
        
    ).set_output(transform="pandas")


    ohe = ColumnTransformer(
        transformers = [
            ("niveaux_risque", one_hot_encoder(), make_column_selector(dtype_exclude=np.number)),
        ],  
            remainder = "passthrough",
            verbose_feature_names_out = False,
        
    ).set_output(transform="pandas")


    isolation_forest = ColumnTransformer(
    transformers = [
        ("isolation_forest", custom_isolation_forest(), make_column_selector(dtype_include=np.number)),
        ("garder_autres_features", "passthrough", make_column_selector(dtype_include=np.number)),
        ],
            remainder = "passthrough",
            verbose_feature_names_out = False,
        
    ).set_output(transform="pandas")
    
    print("step_3_modelling - 5. Création du pipeline : OK")
    
    
    
    # 6. Preprocesing des données d'entrainement
    
    preprocessing = Pipeline([
    ("encoding_niveaux_risque", encoding),
    
    ("ohe_niveaux_risque", ohe),
    
    ("ajout_feature_score_anomalie", isolation_forest),
    
    ])
    
    debut_preprocessing = perf_counter()
    X_train_preprocessing = preprocessing.fit_transform(X_train, y_train)
    temps_preprocessing = perf_counter() - debut_preprocessing

    print(
        "step_3_modelling - 6. Preprocessing des données d'entraînement "
        f"(temps de preprocessing : {temps_preprocessing:.3f} s) : OK"
    )
    
    
    
    # 7. Controle minimal de la qualité des données 
        # TODO : Mettre en place une méthode de validation des données plus robuste avec Great Expectations
        
    if (len(list(X_train_preprocessing.columns)) == 18):
        print("step_3_modelling - 7. Controle minimal de la qualité des données - 18 features : OK")
            
    else:
        raise ValueError("ERREUR : Les données ne correspondent pas à ce qui est attendu")
    
    
    
    # 8. Entrainement du modèle
    
    model = Pipeline([
    ("encoding_niveaux_risque", encoding),
    ("ohe_niveaux_risque", ohe),
    ("ajout_feature_score_anomalie", isolation_forest),
    ("model", xgboost()),
    ])
    
    # Les transformateurs du pipeline ont déjà été ajustés à l'étape 6.
    # Entraîner uniquement le classifieur pour mesurer son temps de fit seul.
    debut_fit = perf_counter()
    model.named_steps["model"].fit(X_train_preprocessing, y_train)
    temps_fit = perf_counter() - debut_fit

    print(
        "step_3_modelling - 8. Entraînement du modèle "
        f"(temps de fit : {temps_fit:.3f} s) : OK"
    )
    
    
    
    # 9. Recherche du seuil optimal sur l'échantillon de validation
    
    cost_optimizer = threshold_cost_optimizer()
    
    proba_validation = model.predict_proba(X_validation)[:,1]

    cost_optimizer.fit(y_validation, proba_validation)

    print(f"Seuil optimal : {cost_optimizer.get_best_threshold()}")
    print(f"Cout pour le seuil optimal : {cost_optimizer.get_best_cost()}€")
    
    print("step_3_modelling - 9. Recherche du seuil optimal sur l'échantillon de validation : OK")
    
    
    
    # 10. Calcul des probabilités et prédictions
    
    proba_train = model.predict_proba(X_train)[:,1]
    pred_train = (proba_train > cost_optimizer.get_best_threshold()).astype(int)

    pred_validation = (proba_validation > cost_optimizer.get_best_threshold()).astype(int)

    proba_test = model.predict_proba(X_test)[:,1]
    pred_test = (proba_test > cost_optimizer.get_best_threshold()).astype(int)
    
    print("step_3_modelling - 10. Calcul des probabilités et prédictions : OK")
    
    
    
    # 11. Affichage des métriques
    
    display_roc_auc_score(y_train,proba_train,
                          y_validation,proba_validation,
                          y_test,proba_test)
    
    display_precision_score(y_train,pred_train,
                            y_validation,pred_validation,
                            y_test,pred_test)
    
    display_recall_score(y_train,pred_train,
                        y_validation,pred_validation,
                        y_test,pred_test)
    
    display_log_loss(y_train,proba_train,
                    y_validation,proba_validation,
                    y_test,proba_test)
    
    print("step_3_modelling - Affichage des métriques : OK")
    
    
    
    # 12. Export des artefacts
    
    joblib.dump(
        value = model,
        filename = f"06_models/{echantillon}XGBoostModel_V1.joblib",
    )
    
    joblib.dump(
        value = cost_optimizer,
        filename = f"07_artifacts/{echantillon}CostOptimizer.joblib",
    )   
    
    print("step_3_modelling - Export du modèle : OK")

        