"""Fabriques des encodeurs, du classifieur et de l’optimiseur de seuil du projet."""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from FraudDetection.DataModelling.ModellingTools import CustomIsolationForest

# ------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------ 
def features_selected():

    features = [
    'montant_eur',
    'categorie_marchand',
    'pays_marchand',
    'pays_ip',
    'canal',
    'mode_saisie',
    'heure',
    'distance_domicile_marchand_km',
    'delai_precedent_carte_h',
    'nb_carte_1h_passe',
    'nb_carte_jour_passe',
    'nb_appareils_carte_passe',
    'anciennete_carte_j',
    'ecart_mediane_enseigne_eur',
    'ecart_mediane_carte_eur',

    # Parts des habitudes client, calculées uniquement sur les paiements passés.
    'part_client_heure_00_passe',
    'part_client_heure_01_passe',
    'part_client_heure_02_passe',
    'part_client_heure_03_passe',
    'part_client_heure_04_passe',
    'part_client_heure_05_passe',
    'part_client_heure_06_passe',
    'part_client_heure_07_passe',
    'part_client_heure_08_passe',
    'part_client_heure_09_passe',
    'part_client_heure_10_passe',
    'part_client_heure_11_passe',
    'part_client_heure_12_passe',
    'part_client_heure_13_passe',
    'part_client_heure_14_passe',
    'part_client_heure_15_passe',
    'part_client_heure_16_passe',
    'part_client_heure_17_passe',
    'part_client_heure_18_passe',
    'part_client_heure_19_passe',
    'part_client_heure_20_passe',
    'part_client_heure_21_passe',
    'part_client_heure_22_passe',
    'part_client_heure_23_passe',
    'part_client_categorie_marchand_alimentation_passe',
    'part_client_categorie_marchand_boulangerie_passe',
    'part_client_categorie_marchand_carburant_passe',
    'part_client_categorie_marchand_habillement_passe',
    'part_client_categorie_marchand_hebergement_passe',
    'part_client_categorie_marchand_loisirs_passe',
    'part_client_categorie_marchand_pharmacie_passe',
    'part_client_categorie_marchand_restauration_passe',
    'part_client_categorie_marchand_services_numeriques_passe',
    'part_client_categorie_marchand_transport_passe',
    'part_client_categorie_marchand_electronique_passe',
    'part_client_categorie_marchand_equipement_de_la_maison_passe',
    'part_client_pays_marchand_be_passe',
    'part_client_pays_marchand_ca_passe',
    'part_client_pays_marchand_ch_passe',
    'part_client_pays_marchand_de_passe',
    'part_client_pays_marchand_es_passe',
    'part_client_pays_marchand_fr_passe',
    'part_client_pays_marchand_gb_passe',
    'part_client_pays_marchand_it_passe',
    'part_client_pays_marchand_nl_passe',
    'part_client_pays_marchand_pt_passe',
    'part_client_pays_marchand_us_passe',
    'part_client_canal_application_passe',
    'part_client_canal_magasin_passe',
    'part_client_canal_site_web_passe',
    'part_client_canal_a_distance_passe',
    'part_client_mode_saisie_carte_enregistree_passe',
    'part_client_mode_saisie_portefeuille_mobile_passe',
    'part_client_mode_saisie_puce_passe',
    'part_client_mode_saisie_saisie_en_ligne_passe',
    'part_client_mode_saisie_sans_contact_passe',
    ]

    return features


def one_hot_encoding():

    custom_one_hot_encoding = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False

    )

    return custom_one_hot_encoding


def numeriques_sans_nan(X):
    """Colonnes numériques ou booléennes sans NaN, sélectionnées au fit seulement.

    Après le OHE, les indicatrices sont numériques et donc éligibles.
    ColumnTransformer conserve la sélection apprise pour les transformations
    suivantes ; validation et test ne servent pas à sélectionner les colonnes.
    """
    numeriques = X.select_dtypes(include=[np.number, "bool"])
    return numeriques.columns[~numeriques.isna().any()].tolist()


def custom_isolation_forest():
    """Crée le transformateur qui calcule le score d'anomalie des transactions.

    Returns
    -------
    CustomIsolationForest
        Instance non ajustée avec 300 arbres, max_samples="auto" et
        contamination=0.01. Les autres paramètres gardent leurs défauts.
        Après fit, transform produit uniquement score_anomalie, égal à
        l'opposé de decision_function : un score plus élevé indique
        une transaction plus atypique, sans être une probabilité de fraude.
    """
    
    custom_isolation_forest = CustomIsolationForest(
        n_estimators = 300,
        max_samples = "auto",
        contamination = 0.006,
    )
    
    return custom_isolation_forest



def xgboost():
    """Crée le classifieur XGBoost avec les hyperparamètres du projet.

    Returns
    -------
    XGBClassifier
        Nouvelle instance non entraînée. La configuration fixe notamment
        305 arbres, une profondeur maximale de 8, le taux d'apprentissage,
        les proportions d'échantillonnage et les pénalités de régularisation.
        Appeler fit sur les variables préparées avant toute prédiction.

    Notes
    -----
    Cette fonction ne réalise aucune recherche d'hyperparamètres ;
    elle instancie les valeurs déclarées dans son corps.
    """
    
    xgboost = XGBClassifier(
        colsample_bytree = 0.75,
        gamma = 0.105,
        learning_rate = 0.02,
        max_depth = 6,
        min_child_weight = 5,
        n_estimators = 135,
        reg_alpha = 0.2,
        reg_lambda = 1.5,
        subsample = 0.75,
        scale_pos_weight=3,
    )
    
    return xgboost
