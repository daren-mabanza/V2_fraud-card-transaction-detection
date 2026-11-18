import yaml
import pandas as pd
from Tools.DataManipulation import convert_columns_type

# ------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------ 

def data_cleaning():
    """
    Prépare les données de paiements par carte pour l'EDA.

    Le préfixe de l'échantillon (``Full`` ou ``Sample``) est lu dans
    ``11_configs/paths.yaml``, via ``echantillon.type``. Les données sont
    chargées depuis ``01_data/01_raw_data/{echantillon}PaiementsCarte.parquet``
    puis copiées avant leur traitement.

    Les conversions concernent l'identifiant ``id_paiement`` en texte,
    les libellés répétés, les colonnes numériques et ``date_naissance``
    en datetime. Les codes postaux et INSEE sont traités comme des libellés
    pour conserver leurs zéros initiaux. Les horodatages UTC et les
    indicateurs booléens déjà typés sont conservés.

    Le contrôle minimal vérifie la présence de 35 colonnes et de la cible
    ``fraude_confirmee``. Le résultat est enregistré dans
    ``01_data/02_intermediate_data/{echantillon}PaiementsCarteForEda.parquet``.

    Returns:
        None: Le résultat est enregistré dans le répertoire intermédiaire.

    Raises:
        FileNotFoundError: Si le fichier de configuration ou les données brutes
            sont introuvables, ou si l'export échoue pour un chemin introuvable.
        KeyError: Si une colonne attendue est absente des données brutes.
            Également si une clé de configuration requise est absente.
        ValueError: Si une conversion échoue, si le jeu de données final
            ne contient pas les 35 colonnes attendues ou si la colonne
            ``fraude_confirmee`` est absente.
    """
    
    # 0. Import des configurations
    
    config = yaml.safe_load(open("11_configs/paths.yaml"))
    echantillon = config["echantillon"]["type"]
    
    print("step_1_data_cleaning - 0. Import des configurations - OK")
        
    # 1. Import des données
        
    try:
        full_data = pd.read_parquet(f"01_data/01_raw_data/{echantillon}PaiementsCarte.parquet")
        df_copy = full_data.copy()
        print("step_1_data_cleaning - 1. Import des données - OK")
            
    except FileNotFoundError:
        raise FileNotFoundError("step_1_data_cleaning - 1. Import des données - Le fichier est introuvable")
        
    # 2. Conversion des colonnes

        # id_paiement
    df_copy["id_paiement"] = df_copy["id_paiement"].astype("string")

        # Colonnes catégorielles
    colonnes_categorielles = [
    "id_client", "id_carte", "prenom", "nom", "adresse_domicile",
    "code_postal_domicile", "commune_domicile", "code_insee_domicile",
    "departement_domicile", "region_domicile", "id_marchand", "nom_marchand",
    "categorie_marchand", "id_point_de_vente", "ville_marchand",
    "pays_marchand", "devise", "canal", "mode_saisie", "id_appareil", "pays_ip",
    ]

    df_copy = convert_columns_type(df_copy, colonnes_categorielles, "category")

        # Entiers positifs
    entiers_positifs = [
        "temps_unix","population_commune"
    ]

    df_copy = convert_columns_type(df_copy, entiers_positifs, "uint32")

        # Nombres décimaux (float32)
    entiers_positifs = [
        "latitude_commune","longitude_commune","latitude_marchand","longitude_marchand"
    ]

    df_copy = convert_columns_type(df_copy, entiers_positifs, "float32")

        # Nombres décimaux (float64)
    nombres_decimaux = [
        "montant_origine","montant_eur"
    ]

    df_copy = convert_columns_type(df_copy, nombres_decimaux, "float64")

        # Date
    df_copy["date_naissance"] = pd.to_datetime(df_copy["date_naissance"], errors="raise")


    print("step_1_data_cleaning - 2. Conversion des colonnes - OK")
        
    # 3. Controle minimal de la qualité des données
        # TODO : Mettre en place une méthode de validation des données plus robuste avec Great Expectations
        
    if (len(list(df_copy.columns)) == 35) & ("fraude_confirmee" in list(df_copy.columns)):
        print("step_1_data_cleaning - 3. Controle minimal de la qualité des données - 35 colonnes | fraude_confirmee présent : OK")
            
    else:
        raise ValueError("ERREUR : Les données ne correspondent pas à ce qui est attendu")
        
    # 4. Export des données
    
    try:
        df_copy.to_parquet(f"01_data/02_intermediate_data/{echantillon}PaiementsCarteForEda.parquet")
        print("step_1_data_cleaning - 4. Export des données - OK")
        
    except FileNotFoundError:
        raise FileNotFoundError("step_1_data_cleaning - 4. Export des données - Le fichier est introuvable")

    # 5. Dataframe exporté
    print(f"step_1_data_cleaning - 5. Dataframe exporté - {echantillon}PaiementsCarteForEda.parquet")
