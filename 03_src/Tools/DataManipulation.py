from collections.abc import Sequence
from numbers import Integral, Real

import numpy as np
import pandas as pd

from datetime import timedelta
from dateutil.easter import easter

TZ_METIER = "Europe/Paris"


def convert_columns_type(
    df: pd.DataFrame,
    columns: str | Sequence[str],
    dtype: str | type,
) -> pd.DataFrame:
    """
    Convertit une ou plusieurs colonnes vers un type donné.

    La conversion modifie le DataFrame reçu et retourne ce même objet.
    Toutes les colonnes indiquées reçoivent le même type. Pour appliquer
    plusieurs types, appeler la fonction une fois par type.

    Args:
        df: DataFrame à modifier.
        columns: Nom d'une colonne ou séquence de noms de colonnes.
        dtype: Type accepté par ``Series.astype`` (par exemple ``"uint32"``,
            ``"float64"``, ``"category"`` ou ``"bool"``). ``"str"`` et
            ``"string"`` utilisent le type texte pandas. ``"datetime"``
            utilise ``pd.to_datetime(errors="raise")``.

    Returns:
        Le DataFrame fourni, avec les colonnes converties.

    Raises:
        TypeError: Si df n'est pas un DataFrame.
        ValueError: Si aucune colonne n'est fournie ou si une conversion échoue.
        KeyError: Si une colonne demandée est absente.

    Notes:
        Choisir un type compatible avec les valeurs et leur précision.
        En cas d'échec sur une colonne, les conversions des colonnes
        précédentes restent appliquées. Passer ``df.copy()`` pour préserver
        le DataFrame d'origine.

    Examples:
        >>> df = pd.DataFrame({"population_commune": [67, 2103778]})
        >>> df = convert_columns_type(df, "population_commune", "uint32")
        >>> str(df["population_commune"].dtype)
        'uint32'

        Conversion de plusieurs libellés du fichier de paiements :

        >>> df = pd.DataFrame({"canal": ["web", "web"], "devise": ["EUR", "EUR"]})
        >>> df = convert_columns_type(df, ["canal", "devise"], "category")
        >>> str(df["canal"].dtype)
        'category'
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df doit être un DataFrame pandas")

    if isinstance(columns, str):
        columns = [columns]
    else:
        columns = list(columns)

    if not columns:
        raise ValueError("Au moins une colonne doit être fournie")

    missing_columns = [column for column in columns if column not in df.columns]

    if missing_columns:
        raise KeyError(f"Colonnes absentes : {missing_columns}")

    target_dtype = "string" if dtype in {"str", "string"} else dtype

    for column in columns:
        try:
            if dtype == "datetime":
                df[column] = pd.to_datetime(df[column], errors="raise")
            else:
                df[column] = df[column].astype(target_dtype)

        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Impossible de convertir la colonne '{column}' "
                f"du type '{df[column].dtype}' vers '{dtype}'"
            ) from error

    return df


def stratified_sample(
    df: pd.DataFrame,
    columns: str | Sequence[str],
    *,
    n: int | None = None,
    frac: float | None = None,
    random_state: int | None = 42,
) -> pd.DataFrame:
    """Tire un échantillon proportionnel, sans remise, par strate.

    Une strate correspond à une valeur de ``columns`` ou, avec plusieurs
    colonnes, à une combinaison de leurs valeurs. Les valeurs manquantes
    forment aussi des strates et ne sont pas exclues du tirage.

    Args:
        df: DataFrame source, qui n'est pas modifié.
        columns: Colonne ou séquence de colonnes définissant les strates.
        n: Nombre exact de lignes à tirer, entre 0 et ``len(df)`` inclus.
            Fournir exactement un des paramètres ``n`` et ``frac``.
        frac: Proportion à tirer, entre 0 et 1 inclus. La taille finale
            vaut ``round(len(df) * frac)`` (arrondi Python au pair en cas
            d'égalité). Ne pas fournir en même temps que ``n``.
        random_state: Graine du générateur aléatoire. La même graine et le
            même DataFrame dans le même ordre reproduisent le tirage.
            Utiliser ``None`` pour un tirage non déterministe.

    Returns:
        Un nouveau DataFrame contenant le nombre demandé de lignes, dans
        un ordre aléatoire, avec les colonnes, types et index d'origine.
        Un index source non unique peut donc rester non unique.

    Raises:
        TypeError: Si ``df`` n'est pas un DataFrame, si ``columns`` n'est
            pas un nom ou une séquence de noms, ou si ``n`` / ``frac``
            n'a pas le type numérique attendu.
        KeyError: Si une colonne de stratification est absente.
        ValueError: Si aucune colonne n'est fournie, si les noms sont
            répétés ou ambigus, si aucun ou les deux paramètres de taille
            sont fournis, ou si la taille demandée est hors limites.

    Notes:
        Les effectifs théoriques sont arrondis à l'entier inférieur, puis
        les lignes restantes sont attribuées aux plus grands restes.
        L'écart à l'effectif proportionnel théorique est inférieur à une
        ligne par strate. Les ex æquo suivent l'ordre des groupes pandas.
        Une strate rare peut être absente d'un petit échantillon : sa
        présence n'est pas garantie. Ce tirage conserve les proportions ;
        il ne rééquilibre pas les classes de fraude.

    Examples:
        >>> df = pd.DataFrame({"fraude_confirmee": [False] * 90 + [True] * 10})
        >>> sample = stratified_sample(df, "fraude_confirmee", n=20)
        >>> sample["fraude_confirmee"].value_counts().to_dict()
        {False: 18, True: 2}

        Créer un échantillon de 10 % des paiements en conservant la
        distribution conjointe de la fraude et du canal :

        .. code-block:: python

            full_data = pd.read_parquet(
                "01_data/01_raw_data/FullPaiementsCarte.parquet"
            )
            sample_data = stratified_sample(
                full_data,
                ["fraude_confirmee", "canal"],
                frac=0.10,
                random_state=42,
            )
            sample_data.to_parquet(
                "01_data/01_raw_data/SamplePaiementsCarte.parquet",
                index=False,
            )
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df doit être un DataFrame pandas")
    if isinstance(columns, str):
        columns = [columns]
    elif isinstance(columns, Sequence) and all(isinstance(c, str) for c in columns):
        columns = list(columns)
    else:
        raise TypeError("columns doit être un nom ou une séquence de noms de colonnes")
    if not columns:
        raise ValueError("Au moins une colonne de stratification doit être fournie")
    if len(set(columns)) != len(columns):
        raise ValueError("Les colonnes de stratification doivent être distinctes")
    missing_columns = [c for c in columns if c not in df.columns]
    if missing_columns:
        raise KeyError(f"Colonnes absentes : {missing_columns}")
    if any((df.columns == c).sum() != 1 for c in columns):
        raise ValueError("Les noms des colonnes de stratification doivent être uniques dans df")
    if (n is None) == (frac is None):
        raise ValueError("Fournir exactement un des paramètres n et frac")
    if n is not None:
        if isinstance(n, bool) or not isinstance(n, Integral):
            raise TypeError("n doit être un entier")
        if not 0 <= n <= len(df):
            raise ValueError("n doit être compris entre 0 et le nombre de lignes")
    else:
        if isinstance(frac, bool) or not isinstance(frac, Real):
            raise TypeError("frac doit être un nombre réel")
        if not 0 <= frac <= 1:
            raise ValueError("frac doit être compris entre 0 et 1")
        n = round(len(df) * frac)

    if n == 0:
        return df.iloc[:0].copy()

    groups = list(df.groupby(columns, observed=True, dropna=False, sort=False).indices.values())
    sizes = np.array([len(positions) for positions in groups], dtype=np.int64)
    # Calcul entier pour éviter les erreurs d'arrondi des quotas.
    allocations = sizes * n // len(df)
    remainders = sizes * n % len(df)
    remaining = int(n - allocations.sum())
    order = np.argsort(-remainders, kind="stable")
    allocations[order[:remaining]] += 1

    rng = np.random.default_rng(random_state)
    selected = np.concatenate([
        rng.choice(positions, size=int(count), replace=False)
        for positions, count in zip(groups, allocations)
        if count > 0
    ])
    rng.shuffle(selected)
    return df.iloc[selected].copy()



def feries_metropole_metier(annee):
    """Renvoie un ensemble de dates des jours fériés nationaux pour l'année donnée.

    Inclut les huit fêtes fixes et les trois fêtes liées à Pâques.
    Le calendrier ne comprend pas les particularités régionales.
    """
    fixes = [(1, 1), (5, 1), (5, 8), (7, 14), (8, 15), (11, 1), (11, 11), (12, 25)]
    jours = {pd.Timestamp(year=annee, month=m, day=j).date() for m, j in fixes}
    paques = easter(annee)
    jours.update(paques + timedelta(days=d) for d in [1, 39, 50])
    return jours


def sessions_metier(mois, heure=9, exclure_feries=True, *, fuseau=TZ_METIER):
    """Renvoie les sessions du mois, du lundi au vendredi, à l'heure locale donnée.

    Args:
        mois: Mois calendaire, par exemple ``"2025-04"``.
        heure: Heure locale de revue, de 0 à 23.
        exclure_feries: Exclure les jours fériés du calendrier métropolitain.
        fuseau: Fuseau de la session, Europe/Paris par défaut.

    Returns:
        DatetimeIndex avec fuseau, y compris si aucun jour n'est retenu.
    """
    if isinstance(heure, bool) or not isinstance(heure, Integral) or not 0 <= heure <= 23:
        raise ValueError("heure doit être un entier de 0 à 23")
    debut = pd.Period(mois, freq="M").start_time.tz_localize(fuseau)
    fin = debut + pd.DateOffset(months=1)
    feries = feries_metropole_metier(debut.year) if exclure_feries else set()
    return pd.DatetimeIndex([
        jour + pd.DateOffset(hours=int(heure))
        for jour in pd.date_range(debut, fin, freq="D", inclusive="left")
        if jour.weekday() < 5 and jour.date() not in feries
    ], tz=fuseau)


def points_metier_transactions(d, *, fuseau=TZ_METIER):
    """Calcule de 0 à 5 points par transaction, sans modifier le DataFrame.

    Utilise montant_eur, canal, pays_marchand et date_paiement_utc.
    Les seuils >=100 et >=250 EUR sont cumulatifs. Hors Magasin, hors FR
    et un horaire strictement entre 00:00 et 06:00 locales ajoutent un point.
    Un champ manquant ne satisfait pas son critère. La cible n'est pas utilisée.
    Renvoie une Series int8 avec le même index que l'entrée.
    """
    local = pd.to_datetime(d["date_paiement_utc"], utc=True).dt.tz_convert(fuseau)
    # Horloge locale, pas durée écoulée depuis minuit (important aux changements d'heure).
    secondes = local.dt.hour*3600 + local.dt.minute*60 + local.dt.second + local.dt.microsecond/1e6
    return (
        d.montant_eur.ge(100).fillna(False).astype("int8")
        + d.montant_eur.ge(250).fillna(False).astype("int8")
        + (d.canal.notna() & d.canal.ne("Magasin")).astype("int8")
        + (d.pays_marchand.notna() & d.pays_marchand.ne("FR")).astype("int8")
        + (secondes.gt(0) & secondes.lt(6*3600)).astype("int8")
    )


def simuler_file_metier(transactions, mois, methode, capacite, heure=9, exclure_feries=True, *, fuseau=TZ_METIER):
    """Simule une file mensuelle, sans regarder les transactions futures.

    Entrée : id_paiement, id_client, id_carte, montant, proba, points, label,
    date_paiement et date_confirmation. Dates interprétées comme instants UTC.
    Toutes les transactions du mois sont conservées dans la sortie, traitées ou non.
    Une revue traite toutes les transactions déjà arrivées de la carte retenue.
    Seuls scores, effectifs et montants des transactions en attente décident du rang.

    Retourne : détail transactions, journal des revues, suivi quotidien, métriques.
    Les métriques de pertinence distinguent revues (réouvertures comprises) et
    cartes distinctes. Une confirmation exactement à l'heure du traitement est trop tard.

    ``fuseau`` détermine les bornes du mois et les sessions ; les instants de
    paiement, traitement et confirmation sont conservés en UTC. Aucune dépendance
    aux variables du notebook. Les dates sont requises au format datetime ou ISO.
    Un mois vide est accepté ; un ratio sans dénominateur vaut NaN.
    """
    if methode not in ["ML", "Regle"] or isinstance(capacite, bool) or not isinstance(capacite, Integral) or capacite < 1:
        raise ValueError("Méthode attendue : ML/Regle ; capacité entière strictement positive")
    requis = {"id_paiement", "id_client", "id_carte", "montant", "proba", "points", "label",
              "date_paiement", "date_confirmation"}
    if manquantes := sorted(requis.difference(transactions.columns)):
        raise KeyError(f"Colonnes de simulation absentes : {manquantes}")
    mois = str(pd.Period(mois, freq="M"))
    capacite = int(capacite)
    d = transactions.copy().reset_index(drop=True)
    if d.id_paiement.isna().any() or d.id_paiement.duplicated().any() or d.id_carte.isna().any():
        raise ValueError("Identifiants paiement uniques et cartes renseignées requis")
    d["date_paiement"] = pd.to_datetime(d.date_paiement, utc=True, format="mixed")
    d["date_confirmation"] = pd.to_datetime(d.date_confirmation, utc=True, format="mixed")
    debut = pd.Timestamp(str(mois) + "-01", tz=fuseau)
    fin = debut + pd.DateOffset(months=1)
    if not d.date_paiement.between(debut, fin, inclusive="left").all():
        raise ValueError("Fournir uniquement les transactions du mois étudié, dates valides")
    if not np.isin(d.label, [0, 1]).all():
        raise ValueError("Les labels doivent être binaires")
    if not (np.isfinite(d[["montant", "proba", "points"]]).all().all()
            and d.proba.between(0, 1).all() and d.points.between(0, 5).all()):
        raise ValueError("Montants, probabilités ou points invalides")
    d["id_carte"] = d.id_carte.astype("string")
    d["date_traitement"] = pd.Series(pd.NaT, index=d.index, dtype="datetime64[us, UTC]")
    d["id_dossier"] = pd.Series(pd.NA, index=d.index, dtype="Int64")
    journaux, suivis = [], []
    prochain_id = 1
    for session in sessions_metier(mois, heure, exclure_feries, fuseau=fuseau):
        instant = session.tz_convert("UTC")
        attente = d.date_traitement.isna() & d.date_paiement.le(instant)
        # Aucun label ni date de confirmation ne participe à cette agrégation.
        file = d.loc[attente].groupby("id_carte", observed=True).agg(
            somme_proba=("proba", "sum"), somme_points=("points", "sum"),
            nb_transactions=("id_paiement", "size"), somme_montant=("montant", "sum"),
        ).reset_index()
        priorite = "somme_proba" if methode == "ML" else "somme_points"
        selection = file.sort_values(
            [priorite, "nb_transactions", "somme_montant", "id_carte"],
            ascending=[False, False, False, True], kind="stable",
        ).head(capacite).copy()
        selection["id_dossier"] = np.arange(prochain_id, prochain_id+len(selection))
        prochain_id += len(selection)
        selection["date_traitement"] = instant
        selection["rang_journalier"] = np.arange(1, len(selection)+1)
        if len(selection):
            ids = selection.set_index("id_carte").id_dossier
            masque = attente & d.id_carte.isin(ids.index)
            d.loc[masque, "id_dossier"] = d.loc[masque, "id_carte"].map(ids).to_numpy()
            d.loc[masque, "date_traitement"] = instant
            journaux.append(selection)
        suivis.append({
            "session_paris": session, "capacite": capacite,
            "transactions_en_attente_avant": int(attente.sum()),
            "dossiers_en_attente_avant": len(file), "revues_effectuees": len(selection),
            "places_inutilisees": capacite-len(selection),
            "transactions_en_attente_apres": int((d.date_traitement.isna() & d.date_paiement.le(instant)).sum()),
        })
    d["ok"] = (d.label.eq(1) & d.date_traitement.notna()
               & d.date_traitement.lt(d.date_confirmation)).astype("int8")
    d["montant_potentiel"] = d.montant.where(d.ok.eq(1), 0)
    colonnes_journal = ["id_carte", "somme_proba", "somme_points", "nb_transactions", "somme_montant",
                        "id_dossier", "date_traitement", "rang_journalier"]
    journal = pd.concat(journaux, ignore_index=True) if journaux else pd.DataFrame(columns=colonnes_journal)
    controles = d.loc[d.date_traitement.notna()]
    if len(journal):
        bilan = controles.groupby("id_dossier").agg(
            fraudes=("label", "sum"), fraudes_a_temps=("ok", "sum"), montant_potentiel=("montant_potentiel", "sum"),
        )
        journal = journal.merge(bilan, left_on="id_dossier", right_index=True, validate="one_to_one")
        d = d.merge(journal[["id_dossier", "somme_proba", "somme_points", "nb_transactions", "somme_montant", "rang_journalier"]],
                    on="id_dossier", how="left", validate="many_to_one")
    else:
        for c in ["fraudes", "fraudes_a_temps", "montant_potentiel"]:
            journal[c] = pd.Series(dtype="float64")
        for c in ["somme_proba", "somme_points", "nb_transactions", "somme_montant", "rang_journalier"]:
            d[c] = np.nan
    div = lambda a, b: 100*a/b if b else np.nan
    n_fraudes = int(d.label.sum())
    montant_fraude = float(d.loc[d.label.eq(1), "montant"].sum())
    cartes_controlees = controles.groupby("id_carte", observed=True).label.max()
    n_jours = len(suivis)
    metriques = {
        "mois": str(mois), "methode": methode, "capacite_jour": capacite,
        "jours_ouvres": n_jours, "capacite_totale": n_jours*capacite,
        "n_transactions": len(d), "n_fraudes": n_fraudes,
        "n_transactions_controlees": len(controles), "n_revues": len(journal),
        "n_cartes_distinctes_controlees": len(cartes_controlees),
        "n_revues_pertinentes": int(journal.fraudes.gt(0).sum()),
        "n_revues_pertinentes_a_temps": int(journal.fraudes_a_temps.gt(0).sum()),
        "n_cartes_pertinentes": int(cartes_controlees.sum()),
        "n_fraudes_a_temps": int(d.ok.sum()),
        "n_fraudes_controlees_non_a_temps": int((d.label.eq(1) & d.date_traitement.notna() & d.ok.eq(0)).sum()),
        "n_fraudes_non_traitees": int((d.label.eq(1) & d.date_traitement.isna()).sum()),
        "n_transactions_non_traitees": int(d.date_traitement.isna().sum()),
        "montant_fraude_total_eur": montant_fraude,
        "montant_potentiellement_recuperable_eur": float(d.montant_potentiel.sum()),
        "fraudes_controlees_a_temps_pct": div(d.ok.sum(), n_fraudes),
        "dossiers_pertinents_pct": div(journal.fraudes.gt(0).sum(), len(journal)),
        "dossiers_pertinents_a_temps_pct": div(journal.fraudes_a_temps.gt(0).sum(), len(journal)),
        "cartes_distinctes_pertinentes_pct": div(cartes_controlees.sum(), len(cartes_controlees)),
        "montant_potentiellement_recuperable_pct": div(d.montant_potentiel.sum(), montant_fraude),
    }
    return {"transactions": d, "dossiers": journal, "quotidien": pd.DataFrame(suivis), "metriques": metriques}


def resumer_metriques_metier(metriques: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compare ML/règle par mois et calcule la synthèse pondérée de la période.

    Args:
        metriques: Une ligne par (mois, scénario, méthode, capacité), avec les
            métriques retournées par simuler_file_metier et la colonne scenario.
            ML et Regle doivent être présents pour chaque mois/scénario/capacité.

    Returns:
        Deux DataFrames sans index métier implicite : ``ecarts_ml_regle_metier``
        (ML moins règle, en points de pourcentage) et ``synthese_periode_metier``
        (sommes des effectifs/montants, taux recalculés sur ces sommes).
        Les cartes distinctes ne sont pas sommées entre mois : une carte peut
        apparaître dans plusieurs mois. Un ratio sans dénominateur reste NaN.

    Raises:
        KeyError: Si une colonne nécessaire est absente.
        ValueError: Si l'entrée est vide, dupliquée ou si une méthode manque.
    """
    kpi = ["fraudes_controlees_a_temps_pct", "dossiers_pertinents_pct",
           "montant_potentiellement_recuperable_pct"]
    sommes = ["n_transactions", "n_fraudes", "n_revues", "n_revues_pertinentes",
              "n_fraudes_a_temps", "montant_fraude_total_eur", "montant_potentiellement_recuperable_eur",
              "capacite_totale", "n_transactions_non_traitees"]
    index = ["mois", "scenario", "capacite_jour"]
    if manquantes := sorted(set(index + ["methode"] + kpi + sommes).difference(metriques.columns)):
        raise KeyError(f"Colonnes de métriques absentes : {manquantes}")
    if metriques.empty or metriques[index + ["methode"]].isna().any().any():
        raise ValueError("Métriques non vides et clés renseignées requises")
    if metriques.duplicated(index + ["methode"]).any():
        raise ValueError("Une seule ligne est attendue par mois/scénario/méthode/capacité")
    methodes = metriques.groupby(index, observed=True).methode.agg(set)
    if not methodes.map(lambda m: m == {"ML", "Regle"}).all():
        raise ValueError("ML et Regle sont requis pour chaque mois/scénario/capacité")

    comparaison = metriques.pivot(index=index, columns="methode", values=kpi)
    ecarts = pd.DataFrame({
        k + "_ecart_points": comparaison[k]["ML"] - comparaison[k]["Regle"] for k in kpi
    }).reset_index()
    synthese = metriques.groupby(["scenario", "methode"], observed=True)[sommes].sum()
    for k, num, den in [
        (kpi[0], "n_fraudes_a_temps", "n_fraudes"),
        (kpi[1], "n_revues_pertinentes", "n_revues"),
        (kpi[2], "montant_potentiellement_recuperable_eur", "montant_fraude_total_eur"),
    ]:
        synthese[k] = 100 * synthese[num] / synthese[den].replace(0, np.nan)
    return {"ecarts_ml_regle_metier": ecarts, "synthese_periode_metier": synthese.reset_index()}
