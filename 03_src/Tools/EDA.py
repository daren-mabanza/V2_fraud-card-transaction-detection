import json
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

def plot_densites(df, colonnes, ncols=3, color="steelblue",
                   fill=True, bw_adjust=1.0,
                   figsize_par_graphique=(5.5, 4.5),
                   titre_general="Densité des variables", hue=None, palette="Set2"):
    """
    Trace une ou plusieurs courbes de densité (KDE pure, sans histogramme)
    en grille, avec nombre de graphiques par ligne configurable.

    Parameters
    ----------
    df : pd.DataFrame
    colonnes : str ou list[str]
        Colonne unique ou liste de colonnes numériques à tracer.
    ncols : int
        Nombre de graphiques par ligne (3 par défaut).
    color : str
        Couleur de la courbe/remplissage.
    fill : bool
        Si True, remplit l'aire sous la courbe de densité.
    bw_adjust : float
        Facteur d'ajustement de la largeur de bande du KDE
        (>1 = plus lisse, <1 = plus sensible aux détails locaux).
    figsize_par_graphique : tuple
        Taille (largeur, hauteur) allouée à CHAQUE sous-graphique.
    titre_general : str
        Titre global de la figure.
    hue : str ou None
        Colonne binaire ou à plusieurs modalités pour comparer les groupes.
        None (défaut) : une courbe globale. Les groupes manquants sont exclus.
    palette : str ou list
        Palette Seaborn pour les groupes, avec une couleur stable par modalité.

    Returns
    -------
    fig : matplotlib.figure.Figure
    resume : pd.DataFrame
        Statistiques par variable, et par groupe si hue est renseigné.

    Notes
    -----
    Chaque courbe de groupe est normalisée séparément (aire égale à 1),
    pour comparer les distributions même si les effectifs diffèrent.
    Les valeurs manquantes ou infinies sont exclues. Une KDE nécessite au
    moins deux valeurs distinctes ; sinon le groupe est résumé mais non tracé.
    """
    if isinstance(colonnes, str):
        colonnes = [colonnes]

    if len(colonnes) == 0:
        raise ValueError("Sélectionner au moins une colonne à tracer.")
    if ncols < 1 or bw_adjust <= 0:
        raise ValueError("ncols et bw_adjust doivent être strictement positifs.")
    for col in colonnes:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"La variable {col!r} doit être numérique.")

    groupes = [None] if hue is None else df[hue].dropna().unique().tolist()
    couleurs = [color] if hue is None else sns.color_palette(palette, n_colors=len(groupes))

    n = len(colonnes)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))

    largeur_fig = figsize_par_graphique[0] * ncols
    hauteur_fig = figsize_par_graphique[1] * nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(largeur_fig, hauteur_fig))
    axes = np.array(axes).reshape(-1)   # Fonctionne même si nrows=ncols=1

    resultats = []

    for i, col in enumerate(colonnes):
        ax = axes[i]
        effectif = 0
        non_traces = []
        for groupe, couleur in zip(groupes, couleurs):
            serie = df[col] if hue is None else df.loc[df[hue] == groupe, col]
            x = serie.dropna().to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            effectif += len(x)
            ligne = {"variable": col}
            if hue is not None:
                ligne["groupe"] = groupe
            ligne.update({
                "n": len(x),
                "moyenne": round(np.mean(x), 4) if len(x) else np.nan,
                "mediane": round(np.median(x), 4) if len(x) else np.nan,
                "ecart_type": round(np.std(x, ddof=1), 4) if len(x) > 1 else np.nan,
                "min": round(np.min(x), 4) if len(x) else np.nan,
                "max": round(np.max(x), 4) if len(x) else np.nan,
            })
            resultats.append(ligne)

            if len(x) < 2 or np.ptp(x) == 0:
                non_traces.append(str(groupe) if hue is not None else col)
                continue

            kde_est = stats.gaussian_kde(x)
            kde_est.set_bandwidth(bw_method=kde_est.factor * bw_adjust)
            grille = np.linspace(x.min(), x.max(), 400)
            densite = kde_est(grille)
            ax.plot(grille, densite, color=couleur, linewidth=2.0,
                    label=str(groupe) if hue is not None else None)
            if fill:
                ax.fill_between(grille, densite, alpha=0.35, color=couleur)

        if hue is not None and ax.lines:
            ax.legend(title=hue)
        if non_traces or not groupes:
            message = "KDE indisponible : " + ", ".join(non_traces) if groupes else "Aucun groupe renseigné"
            ax.text(0.02, 0.98, message, transform=ax.transAxes, va="top", fontsize=8)
        ax.set(title=f"{col} (n={effectif})", xlabel=col, ylabel="Densité")
        ax.grid(alpha=0.25)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(titre_general, y=1.02, fontsize=13)
    fig.tight_layout()

    resume = pd.DataFrame(resultats)
    return fig, resume
# ------------------------------------------------------

def resume_boxplot(df, colonnes, iqr_factor=1.5):
    """
    Calcule les statistiques classiques du boxplot (méthode de Tukey) pour
    une ou plusieurs colonnes numériques.

    Parameters
    ----------
    df : pd.DataFrame
    colonnes : str ou list[str]
        Colonne unique ou liste de colonnes numériques à résumer.
    iqr_factor : float
        Facteur multiplicatif de l'IQR pour les bornes (1.5 = standard Tukey,
        3.0 = bornes plus larges, souvent utilisées pour les outliers "extrêmes").

    Returns
    -------
    pd.DataFrame
        Colonnes : variable, n, borne_basse, Q1, mediane, moyenne, Q3, borne_haute,
        IQR, nb_outliers_bas, nb_outliers_haut.
    """
    if isinstance(colonnes, str):
        colonnes = [colonnes]

    resultats = []

    for col in colonnes:
        x = df[col].dropna().values

        q1 = np.percentile(x, 25)
        mediane = np.percentile(x, 50)
        q3 = np.percentile(x, 75)
        iqr = q3 - q1

        borne_basse = q1 - iqr_factor * iqr
        borne_haute = q3 + iqr_factor * iqr

        moyenne = np.mean(x)

        nb_outliers_bas = int(np.sum(x < borne_basse))
        nb_outliers_haut = int(np.sum(x > borne_haute))

        resultats.append({
            "variable": col,
            "n": len(x),
            "borne_basse": round(borne_basse, 4),
            "Q1": round(q1, 4),
            "mediane": round(mediane, 4),
            "moyenne": round(moyenne, 4),
            "Q3": round(q3, 4),
            "borne_haute": round(borne_haute, 4),
            "IQR": round(iqr, 4),
            "nb_outliers_bas": nb_outliers_bas,
            "nb_outliers_haut": nb_outliers_haut
        })

    return pd.DataFrame(resultats)
# ------------------------------------------------------

def plot_countplots(df, colonnes, ncols=3, order_by_freq=True,
                     stat="count", hue=None, palette="Set2", color="steelblue",
                     figsize_par_graphique=(5.5, 4.5),
                     titre_general="Répartition des variables catégorielles",
                     top_n=None, horizontal=False):
    """Trace un ou plusieurs countplots en grille, avec nombre de graphiques
    par ligne configurable.

    Parameters
    ----------
    df : pd.DataFrame
    colonnes : str ou list[str]
        Colonne unique ou liste de colonnes catégorielles à tracer.
    ncols : int
        Nombre de graphiques par ligne (3 par défaut).
    order_by_freq : bool
        Si True, trie les catégories par effectif décroissant.
    stat : str
        "count" (effectifs bruts) ou "percent" (pourcentages).
    hue : str ou None
        Variable catégorielle secondaire pour comparer des groupes.
    palette : str
        Palette utilisée si hue est défini.
    color : str
        Couleur unique des barres si hue est None.
    figsize_par_graphique : tuple
        Taille (largeur, hauteur) allouée à CHAQUE sous-graphique.
    titre_general : str
        Titre global de la figure.
    top_n : int ou None
        Avec order_by_freq=True, limite l'affichage aux N catégories les plus fréquentes.
    horizontal : bool
        Si True, barres horizontales (utile si labels longs ou nombreuses catégories).

    Returns
    -------
    fig : matplotlib.figure.Figure
    resume : pd.DataFrame
        Nombre de catégories, valeurs manquantes et catégorie dominante par variable.
    """
    if isinstance(colonnes, str):
        colonnes = [colonnes]

    n = len(colonnes)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))

    largeur_fig = figsize_par_graphique[0] * ncols
    hauteur_fig = figsize_par_graphique[1] * nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(largeur_fig, hauteur_fig))
    axes = np.array(axes).reshape(-1)   # Fonctionne même si nrows=ncols=1

    resultats = []

    for i, col in enumerate(colonnes):
        ax = axes[i]

        ordre = df[col].value_counts().index if order_by_freq else None
        if top_n is not None and ordre is not None:
            ordre = ordre[:top_n]

        kwargs = dict(data=df, order=ordre, stat=stat, ax=ax)
        if horizontal:
            kwargs["y"] = col
        else:
            kwargs["x"] = col

        if hue is not None:
            kwargs["hue"] = hue
            kwargs["palette"] = palette
        else:
            kwargs["color"] = color

        sns.countplot(**kwargs)

        ax.set_title(f"{col} ({df[col].nunique()} catégories)")
        if not horizontal:
            ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y" if not horizontal else "x", alpha=0.25)

        resultats.append({
            "variable": col,
            "n": df[col].notna().sum(),
            "nb_categories": df[col].nunique(),
            "nb_manquants": df[col].isna().sum(),
            "categorie_dominante": df[col].value_counts().idxmax(),
            "freq_dominante": df[col].value_counts(normalize=True).max().round(4)
        })

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(titre_general, y=1.02, fontsize=13)
    fig.tight_layout()

    resume = pd.DataFrame(resultats)
    return fig, resume
# ------------------------------------------------------

def heatmap_correlation_triangulaire(df, variables=None, methode="pearson",
                                       triangle="lower", colorscale="RdBu",
                                       decimales=2, titre=None):
    """Crée une heatmap Seaborn et retourne (figure Matplotlib, corrélations).

    variables=None sélectionne toutes les colonnes numériques.
    methode : "pearson", "spearman" ou "kendall".
    triangle : "lower" ou "upper", diagonale incluse.
    colorscale : nom d'une palette Matplotlib ("RdBu", "viridis", etc.).
    decimales : nombre de décimales dans les annotations.
    titre : titre personnalisé, facultatif.
    La taille de la figure, les polices et la rotation des libellés s'adaptent
    au nombre de variables et à la longueur de leurs noms.

    Exemple dans un notebook :
        import matplotlib.pyplot as plt
        fig, corr = heatmap_correlation_triangulaire(df)
        plt.show()
    """
    donnees = df.select_dtypes(include=np.number) if variables is None else df[variables]
    if donnees.shape[1] == 0:
        raise ValueError("Sélectionner au moins une variable numérique.")
    non_numeriques = [col for col in donnees if not pd.api.types.is_numeric_dtype(donnees[col])]
    if non_numeriques:
        raise TypeError(f"Ces variables ne sont pas numériques : {non_numeriques}")
    if methode not in ("pearson", "spearman", "kendall"):
        raise ValueError("methode doit être 'pearson', 'spearman' ou 'kendall'.")
    if triangle not in ("lower", "upper"):
        raise ValueError("triangle doit être 'lower' ou 'upper'.")

    corr = donnees.corr(method=methode)
    masque = np.ones(corr.shape, dtype=bool)
    masque = np.triu(masque, k=1) if triangle == "lower" else np.tril(masque, k=-1)

    n = len(corr)
    longueur_max = max(len(str(col)) for col in corr.columns)
    taille_labels = float(np.clip(12 - 0.12 * n, 8, 11))
    taille_annotations = float(np.clip(12 - 0.16 * n, 7, 11))
    rotation = 90 if n > 20 else (60 if n > 10 or longueur_max > 18 else 30)
    # Agrandir les cellules évite de rendre les textes minuscules.
    # Prévoir aussi de la place pour les noms des nouvelles features.
    cote_matrice = max(3.5, n * max(0.38, (decimales + 3) * taille_annotations / 100))
    marge_labels = longueur_max * taille_labels / 110
    fig, ax = plt.subplots(figsize=(
        max(8, cote_matrice + marge_labels + 1.5),
        max(6, cote_matrice + marge_labels + 1),
    ))
    sns.heatmap(corr, mask=masque, annot=True, fmt=f".{decimales}f",
                annot_kws={"fontsize": taille_annotations},
                xticklabels=True, yticklabels=True,
                cmap=colorscale, vmin=-1, vmax=1, center=0,
                square=True, linewidths=0.5, cbar_kws={"label": "Corr."}, ax=ax)
    ax.set_title(titre or f"Matrice de corrélation ({methode.capitalize()})",
                 fontsize=taille_labels + 3, pad=16)
    ax.tick_params(axis="both", labelsize=taille_labels)
    plt.setp(ax.get_xticklabels(), rotation=rotation,
             ha="center" if rotation == 90 else "right", va="top",
             rotation_mode="default")
    plt.setp(ax.get_yticklabels(), rotation=0)
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=taille_labels)
    colorbar.set_label("Corr.", fontsize=taille_labels)
    fig.tight_layout()
    return fig, corr
# ------------------------------------------------------

def test_mann_whitney_multi(df, var_numeriques=None, var_binaires=None,
                              alpha=0.05, alternative="two-sided",
                              method="auto"):
    """
    Enchaîne le test de Mann-Whitney (U de Mann-Whitney / Wilcoxon rank-sum)
    dans DEUX modes d'usage, selon ce qui est fourni :

    Mode 1 - "1 binaire vs plusieurs numériques" :
        var_binaires = "nom_unique_de_colonne", var_numeriques = [liste]
        -> teste chaque variable numérique entre les 2 groupes de la binaire.

    Mode 2 - "1 numérique vs plusieurs binaires" :
        var_numeriques = "nom_unique_de_colonne", var_binaires = [liste]
        -> teste la variable numérique entre les 2 groupes de chaque binaire.

    H0 : les deux groupes proviennent de la même distribution (pas de décalage
         de position/médiane systématique).
    H1 : les distributions diffèrent (test non paramétrique, alternative au t-test
         quand normalité non vérifiée ou petits échantillons / présence d'outliers).

    Parameters
    ----------
    df : pd.DataFrame
    var_numeriques : str ou list[str]
        - Mode 1 : liste de variables numériques à tester.
        - Mode 2 : nom unique de la variable numérique à tester.
    var_binaires : str ou list[str]
        - Mode 1 : nom unique de la variable binaire (groupe).
        - Mode 2 : liste de variables binaires à tester (chacune = groupe).
    alpha : float
        Seuil de significativité.
    alternative : str
        "two-sided", "greater" ou "less".
    method : str
        "auto", "exact" ou "asymptotic" (passé à scipy.stats.mannwhitneyu).

    Returns
    -------
    pd.DataFrame
        Colonnes : variable_numerique, variable_binaire, groupe1, groupe2,
        n1, n2, mediane1, mediane2, statistique_U, p_value, seuil, significatif,
        r, sens_effet.

    Notes
    -----
    Les groupes sont triés par leur libellé converti en chaîne (0 puis 1 pour
    une variable 0/1). r = Z / sqrt(n1 + n2), où Z est calculé à partir de U1,
    avec correction des ex aequo, sans correction de continuité. Ce calcul
    asymptotique de r est indépendant de la méthode choisie pour la p-value.
    r > 0 ("positif") : le groupe 1 tend à dépasser le groupe 2 ;
    r < 0 ("negatif") : tendance inverse ; r = 0 : "nul".
    Le sens porte sur les rangs, pas uniquement sur la différence de médianes.
    Si toutes les valeurs sont identiques, Z et r sont indéfinis : r = NaN
    et sens_effet = "indetermine". Même résultat pour une paire non testable.
    Les p-values ne sont ni arrondies ni corrigées pour les tests multiples.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha doit être strictement compris entre 0 et 1.")

    mode1 = isinstance(var_numeriques, (list, tuple)) and isinstance(var_binaires, str)
    mode2 = isinstance(var_binaires, (list, tuple)) and isinstance(var_numeriques, str)

    if not (mode1 or mode2):
        raise ValueError(
            "Utilisation ambiguë. Fournir soit :\n"
            "  - var_numeriques=[liste] et var_binaires='colonne' (Mode 1)\n"
            "  - var_numeriques='colonne' et var_binaires=[liste] (Mode 2)"
        )

    if mode1:
        paires = [(col_num, var_binaires) for col_num in var_numeriques]
    else:
        paires = [(var_numeriques, col_bin) for col_bin in var_binaires]

    resultats = []

    for col_num, col_bin in paires:
        sous_df = df[[col_num, col_bin]].dropna()
        modalites = sous_df[col_bin].unique()

        if len(modalites) != 2:
            resultats.append({
                "variable_numerique": col_num, "variable_binaire": col_bin,
                "groupe1": None, "groupe2": None, "n1": np.nan, "n2": np.nan,
                "mediane1": np.nan, "mediane2": np.nan,
                "statistique_U": np.nan, "p_value": np.nan,
                "seuil": alpha, "significatif": None,
                "r": np.nan, "sens_effet": "indetermine"
            })
            continue

        g1_label, g2_label = sorted(modalites, key=str)
        x1 = sous_df.loc[sous_df[col_bin] == g1_label, col_num].values
        x2 = sous_df.loc[sous_df[col_bin] == g2_label, col_num].values

        if len(x1) < 1 or len(x2) < 1:
            resultats.append({
                "variable_numerique": col_num, "variable_binaire": col_bin,
                "groupe1": g1_label, "groupe2": g2_label, "n1": len(x1), "n2": len(x2),
                "mediane1": np.nan, "mediane2": np.nan,
                "statistique_U": np.nan, "p_value": np.nan,
                "seuil": alpha, "significatif": None,
                "r": np.nan, "sens_effet": "indetermine"
            })
            continue

        stat, p_value = stats.mannwhitneyu(x1, x2, alternative=alternative, method=method)
        significatif = p_value < alpha

        # Standardisation de U1 avec variance corrigée pour les ex aequo.
        n1, n2 = len(x1), len(x2)
        total = n1 + n2
        _, effectifs = np.unique(np.concatenate([x1, x2]), return_counts=True)
        effectifs = effectifs.astype(float)  # Évite le débordement du cube entier.
        correction = np.sum(effectifs**3 - effectifs)
        variance_u = n1 * n2 / 12 * (
            total + 1 - correction / (total * (total - 1))
        )
        if variance_u > 0:
            z = (float(stat) - n1 * n2 / 2) / np.sqrt(variance_u)
            r = float(z / np.sqrt(total))
            sens_effet = "positif" if r > 0 else "negatif" if r < 0 else "nul"
        else:
            r, sens_effet = np.nan, "indetermine"

        resultats.append({
            "variable_numerique": col_num,
            "variable_binaire": col_bin,
            "groupe1": g1_label, "groupe2": g2_label,
            "n1": len(x1), "n2": len(x2),
            "mediane1": round(float(np.median(x1)), 4),
            "mediane2": round(float(np.median(x2)), 4),
            "statistique_U": round(float(stat), 4),
            "p_value": float(p_value),
            "seuil": alpha,
            "significatif": significatif,
            "r": r,
            "sens_effet": sens_effet
        })

    colonnes = ["variable_numerique", "variable_binaire", "groupe1", "groupe2",
                "n1", "n2", "mediane1", "mediane2", "statistique_U", "p_value",
                "seuil", "significatif", "r", "sens_effet"]
    resume = pd.DataFrame(resultats, columns=colonnes)
    return resume.reset_index(drop=True)
# ------------------------------------------------------

def analyse_contingence_multi(df, var_lignes=None, var_colonnes=None, alpha=0.05,
                                normalize=False, correction=True):
    """
    Enchaîne table de contingence + test du Khi-2 d'indépendance + V de Cramér
    sur UNE OU PLUSIEURS paires de variables catégorielles, dans DEUX modes :

    Mode 1 - "1 variable fixe vs plusieurs autres" :
        var_lignes = "nom_unique", var_colonnes = [liste]
    Mode 2 - "plusieurs variables vs 1 variable fixe" :
        var_lignes = [liste], var_colonnes = "nom_unique"

    H0 : les deux variables catégorielles sont indépendantes.
    H1 : il existe une association entre les deux variables.

    Le V de Cramér mesure la FORCE de l'association (0=aucune, 1=parfaite),
    indépendamment de la taille d'échantillon.

    Parameters
    ----------
    df : pd.DataFrame
    var_lignes : str ou list[str]
    var_colonnes : str ou list[str]
    alpha : float
    normalize : bool ou str
        False (effectifs bruts), "index", "columns" ou "all".
    correction : bool
        Correction de continuité de Yates.

    Returns
    -------
    tuple (resume, tables)
    """
    mode1 = isinstance(var_lignes, str) and isinstance(var_colonnes, (list, tuple))
    mode2 = isinstance(var_colonnes, str) and isinstance(var_lignes, (list, tuple))

    if not (mode1 or mode2):
        raise ValueError(
            "Fournir soit :\n"
            "  - var_lignes='colonne', var_colonnes=[liste] (Mode 1)\n"
            "  - var_lignes=[liste], var_colonnes='colonne' (Mode 2)"
        )

    paires = [(var_lignes, c) for c in var_colonnes] if mode1 else [(l, var_colonnes) for l in var_lignes]

    resultats = []
    tables = {}

    for v1, v2 in paires:
        sous_df = df[[v1, v2]].dropna()
        table_brute = pd.crosstab(sous_df[v1], sous_df[v2])

        chi2, p_value, dof, _ = stats.chi2_contingency(table_brute, correction=correction)

        n = table_brute.values.sum()
        r, c = table_brute.shape
        v_cramer = np.sqrt((chi2 / n) / (min(r - 1, c - 1))) if min(r, c) > 1 else np.nan

        resultats.append({
            "var1": v1, "var2": v2,
            "chi2": round(float(chi2), 4),
            "dof": int(dof),
            "p_value": round(float(p_value), 4),
            "seuil": alpha,
            "significatif": p_value < alpha,
            "V_cramer": round(float(v_cramer), 4),
            "n_observations": int(n)
        })

        table_affichee = pd.crosstab(sous_df[v1], sous_df[v2], normalize=normalize, margins=True)
        tables[(v1, v2)] = table_affichee.round(4) if normalize else table_affichee

    resume = pd.DataFrame(resultats)
    return resume, tables
