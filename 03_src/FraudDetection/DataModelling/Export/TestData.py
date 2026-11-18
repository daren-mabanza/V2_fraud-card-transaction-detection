import yaml

import joblib
import pandas as pd

# ------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------

def export_test_data():

    # 0. Import des configurations
    config = yaml.safe_load(open("11_configs/paths.yaml"))
    echantillon = config["echantillon"]["type"]

    print("step_5_export_test_data - 0. Import des configurations - OK")

    # 1. Import des données
    data = pd.read_parquet(f"01_data/03_processed_data/{echantillon}PaiementsCarteForModel.parquet")
    df_copy = data.copy()

    print("step_5_export_test_data - 1. Import des données - OK")


    # 2. Import des 10 transactions  (21 avril 2025)
    paiements = [
    "158977dfe57db3c3",
    "c8f35eef66312864",
    "7a1141566807179d",
    "4cd1f2013c126a23",
    "1e04daf101ad58b3",
    "92774f9b756f74c1",
    "786c6b1f8fa3ceb0",
    "c79c310746b6d77d",
    "55179fcfce4609c6",
    "d7e676e22638d09d",
    ]

    df_copy = df_copy.query("id_paiement in @paiements")

    print("step_5_export_test_data - 2. Import des 10 transactions  (21 avril 2025) - OK")


    # 3. Export des 10 transactions (21 avril 2025)
    df_copy.to_parquet("01_data/04_test_data/observations_2025_04_21.parquet")

    print("step_5_export_test_data - 3. Export des 10 transactions (21 avril 2025) - observations_2025_04_21.parquet - OK")
