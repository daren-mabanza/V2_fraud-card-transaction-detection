"""Lecture des exports existants pour l'interface, sans modèle ni écriture."""
from pathlib import Path
import json

import pandas as pd
import yaml

from FastAPITools.PydanticSchemas.Schemas import BatchEntree
from FraudDetection.DataModelling.ModellingParameters import features_selected

ROOT = Path(__file__).resolve().parents[2]
TEST_DATA = "01_data/04_test_data/observations_2025_04_21.parquet"
MONTHS_FR = ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin", "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"]
AVERAGE_METHOD = "Moyenne arithmétique des taux mensuels, chaque mois comptant autant."


def demo_data(root=ROOT):
    """Lit les dix paiements exportés ; seuls les champs utiles sont transmis."""
    root = Path(root)
    features = features_selected()
    data = pd.read_parquet(root / TEST_DATA, columns=["id_paiement", "date_paiement_utc", *features])
    data = data.sort_values("date_paiement_utc", kind="stable").reset_index(drop=True)
    dates = pd.to_datetime(data.date_paiement_utc, utc=True, errors="raise")
    days = dates.dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d")
    if len(data) != 10 or not data.id_paiement.is_unique or data.id_paiement.isna().any() or dates.isna().any() or days.nunique() != 1:
        raise ValueError("Le fichier doit contenir dix paiements distincts d'une même journée.")
    records = data[features].astype(object).where(data[features].notna(), None).to_dict("records")
    BatchEntree.model_validate({"data": records})
    return {
        "day": days.iloc[0], "timezone": "Europe/Paris", "source": TEST_DATA,
        "observations": [
            {"id": str(data.id_paiement.iloc[i]), "date_utc": dates.iloc[i].isoformat(), "features": values}
            for i, values in enumerate(records)
        ],
    }


def business_metrics(root=ROOT):
    """Lit les taux des CSV métier ; seule leur moyenne d'affichage est calculée."""
    root = Path(root)
    config = yaml.safe_load((root / "11_configs/paths.yaml").read_text())
    prefix = config["echantillon"]["type"]
    source = f"{prefix}_metriques_simulation_metier.csv"
    data = pd.read_csv(root / "07_artifacts" / source)
    manifest = json.loads((root / "07_artifacts" / f"{prefix}_parametres_export_metier.json").read_text())
    months = sorted(data.mois.unique())
    if not months:
        raise ValueError("L'export des métriques est vide.")
    periods = [
        {"key": month, "label": f"{MONTHS_FR[pd.Period(month, freq='M').month - 1]} {pd.Period(month, freq='M').year}"}
        for month in months
    ] + [{"key": "average", "label": "Moyenne de tous les mois"}]
    results = []
    columns = {
        "fraud_on_time": "fraudes_controlees_a_temps_pct",
        "relevant_cases": "dossiers_pertinents_pct",
        "recoverable_amount": "montant_potentiellement_recuperable_pct",
    }
    for period in periods:
        for capacity in [8, 10]:
            for method in ["ML", "Regle"]:
                selected = data.loc[data.capacite_jour.eq(capacity) & data.methode.eq(method)]
                if period["key"] != "average":
                    selected = selected.loc[selected.mois.eq(period["key"])]
                expected = len(months) if period["key"] == "average" else 1
                if len(selected) != expected or not selected.mois.is_unique or selected[list(columns.values())].isna().any().any():
                    raise ValueError("Les métriques doivent couvrir chaque mois, capacité et méthode.")
                results.append({
                    "period": period["key"], "capacity": capacity, "method": method,
                    **{key: float(selected[column].mean()) for key, column in columns.items()},
                    "cases": int(selected.n_revues.sum()),
                })
    return {"source": source, "exported_at": manifest["date_export_utc"], "periods": periods,
            "average_method": AVERAGE_METHOD, "results": results}
