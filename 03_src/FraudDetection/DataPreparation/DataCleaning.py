import pandas as pd
from Tools.DataManipulation import convert_columns_type

# ------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------ 

def full_data_cleaning():
    """
    ATTENTION: 
        Version pour la base de données complète (Full)
        
    Actions: 
    Nettoie les données brutes et prépare le jeu de données pour l'EDA.

    La fonction lit ``FullData.parquet``, applique les transformations 
    et conversions de types nécessaires, contrôle la structure minimale
    du résultat, puis écrit les données nettoyées dans ``DataForEda.parquet``.

    Returns:
        None: Le résultat est enregistré dans le répertoire intermédiaire.

    Raises:
        KeyError: Si une colonne attendue est absente des données brutes.
        ValueError: Si le jeu de données final ne contient pas les 15 colonnes
            attendues ou si la colonne ``is_fraud`` est absente.
    """
        
    # 1. Import des données
        
    try:
        full_data = pd.read_parquet(f"01_data/01_raw_data/FullData.parquet")
        df_copy = full_data.copy()
        print("step_1_data_cleaning - 1. Import des données - OK")
            
    except FileNotFoundError:
        print("step_1_data_cleaning - 1. Import des données - Le fichier est introuvable")
        
    # 2. Traitemens spécifiques
        
        # column00
    df_copy["column00"] = 1
    df_copy["column00"] = df_copy["column00"].cumsum()
    df_copy = df_copy.rename(columns={"column00":"id"})
        
        # merchant
    df_copy["merchant"] = df_copy["merchant"].str.slice_replace(0,6,"")
        
    print("step_1_data_cleaning - 2. Traitemens spécifiques - OK")
        
    # 3. Suppression de colonnes
        
    df_copy = df_copy.drop(["cc_num","first","last","gender","street","zip","trans_num","unix_time"], axis = 1)
        
    print("step_1_data_cleaning - 3. Suppression de colonnes - OK")
        
    # 4. Conversion des colonnes

        # Variables numériques
    df_copy["is_fraud"] = df_copy["is_fraud"].astype("uint8")
    convert_columns_type(df_copy, ["id","city_pop"], "uint32")
    convert_columns_type(df_copy, ["amt","lat","long","merch_lat","merch_long"], "float32")
        
        # Variables date
    df_copy["dob"] = pd.to_datetime(df_copy["dob"])
    df_copy["trans_date_trans_time"] = pd.to_datetime(df_copy["trans_date_trans_time"])
        
        # Variables catégorielles
    convert_columns_type(df_copy, ["merchant","category","city","state","job"], "category")
        
    print("step_1_data_cleaning - 4. Conversion des colonnes - OK")
        
    # 5. Controle minimal de la qualité des données 
        
    if (len(list(df_copy.columns)) == 15) & ("is_fraud" in list(df_copy.columns)):
        print("step_1_data_cleaning - 5. Controle minimal de la qualité des données - 15 colonnes | is_fraud présent : OK")
            
    else:
        raise ValueError("ERREUR : Les données ne correspondent pas à ce qui est attendu")
        
    # 6. Export des données 
    
    try:
        df_copy.to_parquet(f"01_data/02_intermediate_data/FullDataForEda.parquet")
        print("step_1_data_cleaning - 5. Export des données - OK")
        
    except FileNotFoundError:
        print("step_1_data_cleaning - 5. Export des données - Le fichier est introuvable")
        
# ------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------         
        
def sample_data_cleaning():
    """
    ATTENTION: 
        Version pour la base de données échantillonée (Sample)
        
    Actions: 
    Nettoie les données brutes et prépare le jeu de données pour l'EDA.

    La fonction lit ``FullData.parquet``, applique les transformations 
    et conversions de types nécessaires, contrôle la structure minimale
    du résultat, puis écrit les données nettoyées dans ``DataForEda.parquet``.

    Returns:
        None: Le résultat est enregistré dans le répertoire intermédiaire.

    Raises:
        KeyError: Si une colonne attendue est absente des données brutes.
        ValueError: Si le jeu de données final ne contient pas les 15 colonnes
                    attendues ou si la colonne ``is_fraud`` est absente.
    """
        
    # 1. Import des données
        
    try:
        sample_data = pd.read_parquet(f"01_data/01_raw_data/SampleData.parquet")
        df_copy = sample_data.copy()
        print("step_1_data_cleaning - 1. Import des données - OK")
            
    except FileNotFoundError:
        print("step_1_data_cleaning - 1. Import des données - Le fichier est introuvable")
        
    # 2. Traitemens spécifiques
        
        # column00
    df_copy["column00"] = 1
    df_copy["column00"] = df_copy["column00"].cumsum()
    df_copy = df_copy.rename(columns={"column00":"id"})
        
        # merchant
    df_copy["merchant"] = df_copy["merchant"].str.slice_replace(0,6,"")
        
    print("step_1_data_cleaning - 2. Traitemens spécifiques - OK")
        
    # 3. Suppression de colonnes
        
    df_copy = df_copy.drop(["cc_num","first","last","gender","street","zip","trans_num","unix_time"], axis = 1)
        
    print("step_1_data_cleaning - 3. Suppression de colonnes - OK")
        
    # 4. Conversion des colonnes

        # Variables numériques
    df_copy["is_fraud"] = df_copy["is_fraud"].astype("uint8")
    convert_columns_type(df_copy, ["id","city_pop"], "uint32")
    convert_columns_type(df_copy, ["amt","lat","long","merch_lat","merch_long"], "float32")
        
        # Variables date
    df_copy["dob"] = pd.to_datetime(df_copy["dob"])
    df_copy["trans_date_trans_time"] = pd.to_datetime(df_copy["trans_date_trans_time"])
        
        # Variables catégorielles
    convert_columns_type(df_copy, ["merchant","category","city","state","job"], "category")
        
    print("step_1_data_cleaning - 4. Conversion des colonnes - OK")
        
    # 5. Controle minimal de la qualité des données 
        
    if (len(list(df_copy.columns)) == 15) & ("is_fraud" in list(df_copy.columns)):
        print("step_1_data_cleaning - 5. Controle minimal de la qualité des données - 15 colonnes | is_fraud présent : OK")
            
    else:
        raise ValueError("ERREUR : Les données ne correspondent pas à ce qui est attendu")
        
    # 6. Export des données 
    
    try:
        df_copy.to_parquet(f"01_data/02_intermediate_data/SampleDataForEda.parquet")
        print("step_1_data_cleaning - 5. Export des données - OK")
        
    except FileNotFoundError:
        print("step_1_data_cleaning - 5. Export des données - Le fichier est introuvable")