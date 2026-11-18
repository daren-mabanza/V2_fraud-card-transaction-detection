"""Export des tableaux d'évaluation métier et de leur manifeste JSON.

Utiliser ``export_metrics`` depuis ce module ou lancer
``04_scripts/step_4_export_metrics.py``. L'import n'effectue aucun export.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from tempfile import NamedTemporaryFile

import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import yaml

from Tools.DataManipulation import (
    TZ_METIER,
    points_metier_transactions,
    resumer_metriques_metier,
    sessions_metier,
    simuler_file_metier,
)

RACINE_PROJET = Path(__file__).resolve().parents[4]
MOIS_PAR_DEFAUT = ("2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06")


def _chemin_projet(chemin: str | Path) -> Path:
    """Résout les chemins relatifs depuis la racine, indépendamment du cwd."""
    chemin = Path(chemin)
    return (chemin if chemin.is_absolute() else RACINE_PROJET / chemin).resolve()


def _sha256(chemin: Path) -> str:
    """Calcule et renvoie l’empreinte SHA-256 hexadécimale du fichier fourni."""
    with chemin.open("rb") as fichier:
        return hashlib.file_digest(fichier, "sha256").hexdigest()


def _ecrire_artefact(chemin: Path, contenu: pd.DataFrame | dict) -> None:
    """Remplace un fichier après écriture complète de son temporaire voisin."""
    temporaire = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=chemin.parent,
                                prefix=".export_metier_", suffix=".tmp", delete=False) as fichier:
            temporaire = Path(fichier.name)
            if isinstance(contenu, pd.DataFrame):
                contenu.to_csv(fichier, index=False)
            else:
                json.dump(contenu, fichier, ensure_ascii=False, indent=2, allow_nan=False)
                fichier.write("\n")
        temporaire.replace(chemin)
    finally:
        if temporaire is not None and temporaire.exists():
            temporaire.unlink()


def export_metrics(
    *,
    echantillon: str | None = None,
    source: str | Path | None = None,
    modele: str | Path | None = None,
    dossier_export: str | Path = "07_artifacts",
    mois: str | Sequence[str] | None = MOIS_PAR_DEFAUT,
    scenarios: Mapping[str, int] | None = None,
    heure_revue: int = 9,
    exclure_feries: bool = True,
    fin_train: str | pd.Timestamp = "2025-01-01",
    fuseau: str = TZ_METIER,
) -> dict[str, Path]:
    """Évalue le pipeline ajusté face aux règles métier et exporte les résultats.

    Simule une file indépendante par mois, méthode et capacité, sans
    réentraîner le modèle. Les labels servent à évaluer les dossiers après
    leur sélection ; ils ne participent pas à leur classement.

    Args:
        echantillon: ``Full`` ou ``Sample`` ; lu dans la configuration si absent.
        source: Parquet préparé ; par défaut, celui de l'échantillon choisi.
        modele: Pipeline joblib ajusté ; par défaut, le XGBoost V2 de l'échantillon.
        dossier_export: Dossier des résultats, créé si nécessaire.
        mois: Mois à évaluer, janvier à juin 2025 par défaut. ``None`` retient
            les trois mois précédant celui du dernier paiement.
        scenarios: Capacités quotidiennes nommées ; par défaut, 10 et 8 dossiers.
            Ces scénarios ne classent pas automatiquement la complexité des dossiers.
        heure_revue: Heure locale de traitement, entre 0 et 23.
        exclure_feries: Exclut les jours fériés métropolitains des sessions.
        fin_train: Fin exclue du train, à faire correspondre au modèle chargé.
            Les paiements antérieurs sont exclus ; une date sans fuseau est UTC.
        fuseau: Fuseau du calendrier et du score métier, Europe/Paris par défaut.

    Returns:
        Dictionnaire des chemins absolus de cinq CSV et d'un manifeste JSON.
        Les fichiers existants sont remplacés. Le manifeste conserve les
        paramètres et empreintes des sources ; les ratios indéfinis restent vides.

    Raises:
        FileNotFoundError: Si la configuration, la source ou le modèle manque.
        KeyError: Si une clé de configuration ou une colonne requise manque.
        TypeError: Si l'artefact n'est pas un pipeline XGBoost.
        ValueError: Si les paramètres, les données ou les résultats sont invalides.
        OSError: Si la création du dossier ou une écriture échoue.

    Notes:
        Les chemins relatifs sont résolus depuis la racine du projet.
        Le montant récupérable suppose une récupération intégrale des fraudes
        contrôlées avant confirmation ; il ne mesure pas une récupération réelle.
    """
    if echantillon is None:
        config = yaml.safe_load((RACINE_PROJET / "11_configs/paths.yaml").read_text())
        echantillon = config["echantillon"]["type"]
    if echantillon not in {"Full", "Sample"}:
        raise ValueError("echantillon doit valoir Full ou Sample")
    source_path = _chemin_projet(source if source is not None else
                                f"01_data/03_processed_data/{echantillon}PaiementsCarteForModel.parquet")
    modele_path = _chemin_projet(modele if modele is not None else
                                f"06_models/{echantillon}XGBoostPipeline_V2.joblib")
    destination = _chemin_projet(dossier_export)
    for chemin in [source_path, modele_path]:
        if not chemin.is_file():
            raise FileNotFoundError(f"Fichier requis introuvable : {chemin}")
    scenarios = dict(scenarios) if scenarios is not None else {"ordinaire_10": 10, "complexe_8": 8}
    if not scenarios or any(
        not isinstance(nom, str) or not nom or isinstance(cap, bool)
        or not isinstance(cap, Integral) or cap <= 0 for nom, cap in scenarios.items()
    ):
        raise ValueError("Fournir des scénarios nommés avec une capacité entière positive")
    scenarios = {nom: int(cap) for nom, cap in scenarios.items()}
    cutoff = pd.to_datetime(fin_train, utc=True, errors="raise")
    if pd.isna(cutoff):
        raise ValueError("fin_train doit être une date valide")

    dates = pd.to_datetime(pd.read_parquet(source_path, columns=["date_paiement_utc"])
                           .date_paiement_utc, utc=True, errors="raise")
    if dates.empty or dates.isna().any():
        raise ValueError("La source doit contenir des dates de paiement renseignées")
    if mois is None:
        dernier_mois = dates.max().tz_convert(fuseau).tz_localize(None).to_period("M")
        mois_retenus = list(pd.period_range(dernier_mois - 3, dernier_mois - 1, freq="M").astype(str))
    else:
        mois_retenus = [mois] if isinstance(mois, str) else list(mois)
        mois_retenus = sorted(str(pd.Period(m, freq="M")) for m in mois_retenus)
    if not mois_retenus or len(mois_retenus) != len(set(mois_retenus)):
        raise ValueError("Fournir au moins un mois, sans doublon")
    calendriers = {}
    for m in mois_retenus:
        debut = pd.Timestamp(m + "-01", tz=fuseau)
        if (debut + pd.DateOffset(months=1)).tz_convert("UTC") <= cutoff:
            raise ValueError(f"Le mois {m} est entièrement dans la période d'entraînement")
        calendriers[m] = sessions_metier(m, heure_revue, exclure_feries, fuseau=fuseau)

    pipeline = joblib.load(modele_path)
    if not hasattr(pipeline, "named_steps") or not type(pipeline.named_steps.get("model")).__module__.startswith("xgboost"):
        raise TypeError("L'artefact doit être un pipeline XGBoost ajusté")
    colonnes_modele = list(pipeline.feature_names_in_)
    if {"fraude_confirmee", "date_confirmation_fraude_utc"}.intersection(colonnes_modele):
        raise ValueError("La cible ou la confirmation ne peut pas faire partie des entrées du modèle")
    classes = list(pipeline.classes_)
    if len(classes) != 2 or set(classes) != {0, 1}:
        raise ValueError("Le modèle doit classifier les classes 0 et 1")
    indice_fraude = classes.index(1)
    champs = ["id_paiement", "id_client", "id_carte", "montant_eur", "fraude_confirmee",
              "date_paiement_utc", "date_confirmation_fraude_utc", "canal", "pays_marchand"]
    expositions, capacites, metriques = [], [], []
    ids_vus = set()
    print(f"Export métier — {echantillon} : {len(mois_retenus)} mois, modèle {modele_path.name}")
    for m in mois_retenus:
        debut = pd.Timestamp(m + "-01", tz=fuseau)
        fin = (debut + pd.DateOffset(months=1)).tz_convert("UTC")
        borne = max(debut.tz_convert("UTC"), cutoff)
        mensuel_source = pd.read_parquet(
            source_path, columns=list(dict.fromkeys(champs + colonnes_modele)),
            filters=[("date_paiement_utc", ">=", borne.to_pydatetime()),
                     ("date_paiement_utc", "<", fin.to_pydatetime())],
        )
        ids = mensuel_source.id_paiement
        if ids.isna().any() or ids.duplicated().any() or ids.isin(ids_vus).any():
            raise ValueError("Les identifiants de paiement doivent être renseignés et uniques")
        ids_vus.update(ids)
        proba = (pipeline.predict_proba(mensuel_source[colonnes_modele])[:, indice_fraude]
                 if len(mensuel_source) else np.empty(0))
        transactions = mensuel_source[champs].rename(columns={
            "montant_eur": "montant", "fraude_confirmee": "label", "date_paiement_utc": "date_paiement",
            "date_confirmation_fraude_utc": "date_confirmation",
        }).copy()
        transactions["date_paiement"] = pd.to_datetime(transactions.date_paiement, utc=True)
        transactions["date_confirmation"] = pd.to_datetime(transactions.date_confirmation, utc=True)
        transactions["proba"] = proba
        transactions["points"] = points_metier_transactions(mensuel_source, fuseau=fuseau).to_numpy()
        expositions.append({
            "mois": m, "n": len(transactions), "fraudes": int(transactions.label.sum()),
            "montant_total_eur": transactions.montant.sum(),
            "premier_paiement_utc": transactions.date_paiement.min(),
            "dernier_paiement_utc": transactions.date_paiement.max(),
        })
        capacites.append({"mois": m, "jours_ouvres": len(calendriers[m]), **{
            f"capacite_{cap}": cap * len(calendriers[m]) for cap in scenarios.values()
        }})
        for scenario, capacite in scenarios.items():
            for methode in ["ML", "Regle"]:
                resultat = simuler_file_metier(
                    transactions, m, methode, capacite, heure=heure_revue,
                    exclure_feries=exclure_feries, fuseau=fuseau,
                )
                ligne = dict(resultat["metriques"], scenario=scenario)
                if ligne["n_fraudes"] != (ligne["n_fraudes_a_temps"] + ligne["n_fraudes_controlees_non_a_temps"]
                                           + ligne["n_fraudes_non_traitees"]):
                    raise ValueError("Incohérence du décompte des fraudes")
                if ligne["n_revues"] > ligne["capacite_totale"]:
                    raise ValueError("Capacité de traitement dépassée")
                metriques.append(ligne)
        print(f"  {m} : {len(transactions):,} transactions évaluées")

    metriques = pd.DataFrame(metriques)
    metriques["utilisation_capacite_pct"] = 100 * metriques.n_revues / metriques.capacite_totale.replace(0, np.nan)
    tableaux = {
        "exposition_mensuelle_metier": pd.DataFrame(expositions),
        "capacites_mensuelles_metier": pd.DataFrame(capacites),
        "metriques_simulation_metier": metriques,
        **resumer_metriques_metier(metriques),
    }
    parametres = {
        "date_export_utc": datetime.now(timezone.utc).isoformat(),
        "echantillon": echantillon,
        "source": str(source_path), "source_sha256": _sha256(source_path),
        "modele": str(modele_path), "modele_sha256": _sha256(modele_path),
        "colonnes_modele": colonnes_modele, "mois": mois_retenus,
        "fin_train_exclue_utc": cutoff.isoformat(), "cutoff_declare_par_appelant": True,
        "fuseau": fuseau, "heure_revue": int(heure_revue), "exclure_feries": bool(exclure_feries),
        "scenarios": scenarios, "n_simulations": len(metriques),
        "n_transactions_evaluees": int(sum(ligne["n"] for ligne in expositions)),
        "seuil_eligibilite": None, "reouverture_cartes": True, "file_reinitialisee_chaque_mois": True,
        "tri_ml": ["somme_proba DESC", "nb_transactions DESC", "somme_montant DESC", "id_carte ASC"],
        "tri_regle": ["somme_points DESC", "nb_transactions DESC", "somme_montant DESC", "id_carte ASC"],
        "score_regle": "1 si montant>=100 + 1 si montant>=250 + 1 hors Magasin + 1 hors FR + 1 si 00:00<horaire<06:00",
        "condition_ok": "label=1 et date_traitement < date_confirmation",
        "recuperation": "potentielle, hypothèse 100% des montants contrôlés à temps",
        "pertinence_dossiers": "revues, réouvertures comprises ; cartes distinctes disponibles séparément",
        "csv": {"encodage": "utf-8", "separateur": ",", "index": False, "arrondi": False},
        "versions": {"pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
    }
    destination.mkdir(parents=True, exist_ok=True)
    chemins = {}
    for nom, tableau in tableaux.items():
        chemins[nom] = destination / f"{echantillon}_{nom}.csv"
        _ecrire_artefact(chemins[nom], tableau)
    chemins["parametres_export_metier"] = destination / f"{echantillon}_parametres_export_metier.json"
    parametres["artefacts"] = {nom: str(chemin) for nom, chemin in chemins.items()}
    _ecrire_artefact(chemins["parametres_export_metier"], parametres)
    print(f"Export terminé : {len(tableaux)} tableaux et 1 manifeste dans {destination}")

    print("step_4_export_metrics - 1. Export des métriques : OK")

    return chemins
