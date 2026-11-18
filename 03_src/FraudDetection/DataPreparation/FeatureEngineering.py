import yaml
import duckdb
import pandas as pd
from FraudDetection.SQL.Query import query

# ------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------ 

def feature_engineering():
    """Calcule les variables métier et historiques avec DuckDB, puis les exporte.

    Charge ``{echantillon}PaiementsCarteForEda.parquet`` selon
    ``11_configs/paths.yaml`` et applique la requête SQL à ``df_copy``.
    Le résultat doit contenir 104 colonnes, dont ``fraude_confirmee``.
    Il remplace ``01_data/03_processed_data/{echantillon}PaiementsCarteForModel.parquet``.
    Les historiques excluent les paiements au même instant et les paiements futurs.
    Lancer depuis la racine du projet, avec le dossier de sortie existant.

    Raises:
        FileNotFoundError: Si la configuration, les données ou le dossier de sortie manquent.
        KeyError: Si la configuration ne contient pas les clés attendues.
        ValueError: Si le contrôle de structure du résultat échoue.
    """

    # 0. Import des configurations
    config = yaml.safe_load(open("11_configs/paths.yaml"))
    echantillon = config["echantillon"]["type"]
    
    print("step_2_feature_engineering - 0. Import des configurations - OK")
    
    
    # 1. Import des données 
    try:
        df = pd.read_parquet(f"01_data/02_intermediate_data/{echantillon}PaiementsCarteForEda.parquet")
        df_copy = df.copy()
        print("step_2_feature_engineering - 1. Import des données - OK")
        
    except FileNotFoundError:
        raise FileNotFoundError("step_2_feature_engineering - 1. Import des données - Le fichier est introuvable")
    
    
    # 2. Création des nouvelles colonnes issus du feature engineering

    df_copy = duckdb.sql(query()).to_df()

    print("step_2_feature_engineering - 2. Création des nouvelles colonnes issus du feature engineering - OK")
    
    
    # 3. Controle minimal de la qualité des données
        # TODO : Mettre en place une méthode de validation des données plus robuste avec Great Expectations
        
    if (len(list(df_copy.columns)) == 104) & ("fraude_confirmee" in list(df_copy.columns)):
        print("step_2_feature_engineering - 4. Controle minimal de la qualité des données - 104 colonnes | fraude_confirmee présent : OK")
            
    else:
        raise ValueError("ERREUR : Les données ne correspondent pas à ce qui est attendu")
    
    
    # 4. Export des données
    try:
        df_copy.to_parquet(f"01_data/03_processed_data/{echantillon}PaiementsCarteForModel.parquet")
        print("step_2_feature_engineering - 5. Export des données - OK")
        
    except FileNotFoundError:
        raise FileNotFoundError("step_2_feature_engineering - 4. Export des données - Le fichier est introuvable")


    # 5. DataFrame exporté

    print(f"step_2_feature_engineering - 5. DataFrame exporté - {echantillon}PaiementsCarteForModel.parquet")
