import yaml
import duckdb
import pandas as pd
from FraudDetection.SQL.Query import query

# ------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------ 

def feature_engineering():
    """Prépare les variables temporelles et exporte les transactions transformées.

    Lit ``11_configs/paths.yaml`` pour récupérer le type d'échantillon dans
    ``echantillon.type``, puis charge le fichier
    ``01_data/02_intermediate_data/{echantillon}DataForEda.parquet``.

    Conserve les colonnes utiles, crée ``jour`` et ``heure`` à partir de
    ``trans_date_trans_time``, puis ordonne les huit colonnes de sortie :
    ``id``, ``trans_date_trans_time``, ``jour``, ``heure``, ``state``,
    ``category``, ``amt`` et ``is_fraud``.

    ``jour`` est une variable catégorielle ordonnée de ``Monday`` à ``Sunday``.
    ``heure`` est une variable catégorielle contenant les heures observées,
    de 0 à 23. Une date manquante donne un jour et une heure manquants.

    Après un contrôle minimal du nombre de colonnes et de la présence de
    ``is_fraud``, écrit le résultat dans
    ``01_data/03_processed_data/{echantillon}DataForEda.parquet``.
    Les étapes sont signalées dans la sortie standard.

    Returns:
        None: Les données préparées sont enregistrées au format Parquet.

    Raises:
        FileNotFoundError: Si le fichier de configuration ou le fichier
            de données d'entrée est introuvable.
        yaml.YAMLError: Si la syntaxe du fichier YAML est invalide.
        KeyError: Si une clé de configuration ou une colonne requise manque.
        ValueError: Si une date ne peut pas être interprétée par pandas ou
            si le contrôle minimal de structure échoue.
        OSError: Si la lecture ou l'écriture d'un fichier échoue, notamment
            si le répertoire de sortie n'existe pas.

    Notes:
        La fonction ne prend aucun argument. Les chemins sont relatifs au
        répertoire de travail, qui doit être la racine du projet.
        Le répertoire de sortie doit déjà exister ; un fichier de sortie
        portant le même nom est remplacé lors de l'export.
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
