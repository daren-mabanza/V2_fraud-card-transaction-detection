"""Transformateurs sklearn, optimisation du seuil et affichage des métriques binaires."""

import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.metrics import recall_score, precision_score,log_loss,roc_auc_score


# ------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------ 

class RiskLevelEncoder(TransformerMixin, BaseEstimator):
    """Remplace chaque colonne reçue par son niveau de risque de fraude.

    ``fit(X, y)`` mémorise les taux de fraude par modalité et un discrétiseur
    quantile par variable. Chaque modalité observée a le même poids dans le
    calcul des quantiles. ``transform(X)`` réutilise ces statistiques, sans
    cible et sans réajustement sur les nouvelles transactions.

    ``fit_transform(X, y)`` produit les niveaux d'entraînement hors pli : les
    taux ET les seuils utilisés pour chaque pli sont appris sur les autres
    plis. Il ajuste ensuite les statistiques finales sur tout l'entraînement
    pour les futurs appels à ``transform``. Ces deux transformations de
    l'entraînement ne sont donc pas équivalentes.

    Parameters
    ----------
    cv : int, default=5
        Nombre de plis KFold mélangés pour l'encodage d'entraînement. Ce
        découpage est indépendant de y, mais ne respecte pas la chronologie.
    random_state : int, default=42
        Graine du mélange reproductible des plis.

    Attributes
    ----------
    feature_names_in_ : numpy.ndarray
        Noms des colonnes reçues pendant fit, dans leur ordre d'entrée.
    n_features_in_ : int
        Nombre de colonnes reçues, égal au nombre de colonnes produites.
    rates_ : dict of pandas.Series
        Taux de fraude par modalité, appris sur l'ensemble fourni à fit.
    binners_ : dict
        KBinsDiscretizer ajusté par variable, ou None si tous les taux sont
        identiques ou qu'aucune modalité non manquante n'est disponible.
    global_mean_ : float
        Taux global de fraude utilisé pour les modalités inconnues/manquantes.
    n_bins_ : dict of int
        Nombre d'intervalles conservés pour chaque variable.

    Notes
    -----
    X doit être un DataFrame sans colonne is_fraud ; y contient des 0 et 1.
    TOUTES les colonnes reçues sont traitées comme des variables catégorielles,
    y compris celles codées par des nombres. La classe ne sélectionne aucun
    nom ou type prédéfini : sélectionner les colonnes en amont, par exemple
    dans un ColumnTransformer. Ne pas transmettre un montant continu si
    l'objectif est simplement de le conserver.

    La sortie contient uniquement une colonne niveau_risque_<nom> par colonne
    d'entrée, dans le même ordre et avec le même index. Elle n'inclut ni les
    colonnes d'origine ni des statistiques intermédiaires. Les niveaux sont
    des catégories ordonnées : tres_faible, faible, moderee, elevee.
    Par exemple, X[["merchant", "job"]] donne niveau_risque_merchant et
    niveau_risque_job. Pour conserver les autres variables, sélectionner ces
    deux colonnes dans un ColumnTransformer avec remainder="passthrough".

    Les ex aequo peuvent réduire le nombre d'intervalles. S'il en reste deux
    ou trois, leurs étiquettes sont réparties dans l'ordre entre tres_faible
    et elevee (deux : 0/3 ; trois : 0/2/3). Si tous les taux sont identiques,
    le niveau reste manquant : aucun risque relatif ne peut être distingué.
    Les quatre catégories de sortie restent déclarées dans tous les cas.

    Les plis internes mélangés ne simulent pas une disponibilité historique
    des labels. Pour une évaluation temporelle, réserver au minimum une
    période de test ultérieure et interpréter l'encodage interne en conséquence.
    """

    LEVELS = ("tres_faible", "faible", "moderee", "elevee")

    def __init__(self, cv=5, random_state=42):
        """Mémorise cv et random_state, sans apprendre de statistiques de fraude."""
        self.cv = cv
        self.random_state = random_state

    def __sklearn_tags__(self):
        """Déclare les capacités de l'encodeur auprès de scikit-learn.

        Returns
        -------
        Tags
            Indique que la cible est requise et que les variables
            catégorielles ainsi que les valeurs manquantes sont acceptées.
        """
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        tags.input_tags.allow_nan = True
        tags.input_tags.categorical = True
        return tags

    def _check_input(self, X, *, reset):
        """Valide le DataFrame et mémorise ou vérifie son schéma de colonnes.

        Parameters
        ----------
        X : pandas.DataFrame
            Données avec au moins une colonne, des noms textuels uniques
            et aucune colonne is_fraud.
        reset : bool
            Si True, mémorise les noms et le nombre de colonnes ; sinon,
            vérifie leur correspondance avec le schéma appris.

        Returns
        -------
        pandas.DataFrame
            X lui-même, sans copie ni modification de ses valeurs.

        Raises
        ------
        TypeError
            Si X n'est pas un DataFrame.
        ValueError
            Si les colonnes sont invalides ou ne respectent pas le schéma.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X doit être un DataFrame pandas.")
        if "is_fraud" in X.columns:
            raise ValueError("Retirer is_fraud de X et passer la cible séparément dans y.")
        if not X.columns.is_unique or not all(isinstance(c, str) for c in X.columns):
            raise ValueError("Les noms des colonnes de X doivent être uniques et textuels.")
        if X.shape[1] == 0:
            raise ValueError("X doit contenir au moins une colonne à encoder.")
        validate_data(self, X, reset=reset, skip_check_array=True)
        return X

    @staticmethod
    def _check_target(X, y):
        """Vérifie la cible binaire et la convertit en tableau de flottants.

        Parameters
        ----------
        X : pandas.DataFrame
            Variables servant de référence pour la longueur et l'index.
        y : array-like of shape (n_samples,)
            Une valeur 0 ou 1 par ligne de X, sans valeur manquante.
            Si y est une Series, son index doit être identique à celui de X.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Cible convertie en float, dans l'ordre fourni.

        Raises
        ------
        ValueError
            Si y est absent, vide, mal aligné ou ne contient pas une cible
            binaire valide par ligne de X.
        """
        if y is None:
            raise ValueError("y est nécessaire pour apprendre les taux de fraude.")
        if isinstance(y, pd.Series) and not y.index.equals(X.index):
            raise ValueError("Les index de X et y doivent être alignés dans le même ordre.")
        values = np.asarray(y)
        if (values.ndim != 1 or len(values) != len(X) or len(values) == 0
                or pd.isna(values).any() or not np.isin(values, [0, 1]).all()):
            raise ValueError("y doit contenir un 0 ou un 1 par ligne de X, sans valeur manquante.")
        return values.astype(float)

    def fit(self, X, y):
        """Apprend les taux de fraude et les seuils de risque sur toutes les lignes.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            Colonnes à traiter comme catégorielles, sans is_fraud.
        y : array-like of shape (n_samples,)
            Cible binaire 0/1 dans le même ordre que X. Pour une Series,
            l'index doit également correspondre à celui de X.

        Returns
        -------
        self : RiskLevelEncoder
            Encodeur ajusté avec le taux global, les taux par modalité
            et les discrétiseurs quantiles propres à chaque colonne.

        Notes
        -----
        Cet appel utilise toutes les cibles fournies. Pour construire les
        variables d'entraînement hors pli, utiliser fit_transform.
        Chaque modalité observée a le même poids dans les quantiles.
        Un nouvel appel remplace les statistiques précédemment apprises.
        """
        self._check_input(X, reset=True)
        target = self._check_target(X, y)
        self.global_mean_ = float(target.mean())
        self.rates_ = {}
        self.binners_ = {}
        self.n_bins_ = {}
        self.bin_label_indices_ = {}

        for variable in self.feature_names_in_:
            # Index positionnel : pas de réalignement involontaire de y.
            rates = pd.Series(target).groupby(
                X[variable].reset_index(drop=True), observed=True, dropna=True
            ).mean()
            self.rates_[variable] = rates
            if rates.nunique() < 2:
                self.binners_[variable] = None
                self.n_bins_[variable] = int(not rates.empty)
                self.bin_label_indices_[variable] = np.array([], dtype=int)
                continue

            binner = KBinsDiscretizer(
                n_bins=4, encode="ordinal", strategy="quantile",
                quantile_method="averaged_inverted_cdf", subsample=None,
            )
            with warnings.catch_warnings():
                # Les intervalles fusionnés sont gérés et exposés dans n_bins_.
                warnings.filterwarnings("ignore", message="Bins whose width.*", category=UserWarning)
                binner.fit(rates.to_numpy().reshape(-1, 1))
            n_bins = int(binner.n_bins_[0])
            self.binners_[variable] = binner if n_bins > 1 else None
            self.n_bins_[variable] = n_bins
            self.bin_label_indices_[variable] = np.rint(np.linspace(0, 3, n_bins)).astype(int)
        return self

    def _output(self, X, labels):
        """Construit le DataFrame des niveaux de risque avec l'index d'entrée.

        Parameters
        ----------
        X : pandas.DataFrame
            Données dont l'index doit être conservé.
        labels : numpy.ndarray of shape (n_samples, n_features)
            Étiquettes de risque par ligne et colonne ; None indique
            l'absence d'un niveau discriminant.

        Returns
        -------
        pandas.DataFrame
            Une catégorie ordonnée niveau_risque_<nom> par colonne apprise,
            dans l'ordre de feature_names_in_. Aucune colonne d'origine
            ni statistique intermédiaire n'est ajoutée.
        """
        result = pd.DataFrame(index=X.index)
        for j, variable in enumerate(self.feature_names_in_):
            result[f"niveau_risque_{variable}"] = pd.Categorical(
                labels[:, j], categories=self.LEVELS, ordered=True
            )
        return result

    def transform(self, X):
        """Applique les niveaux de risque appris, sans réentraîner l'encodeur.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            Données avec les mêmes noms de colonnes, dans le même ordre
            qu'au fit. La cible n'est pas nécessaire.

        Returns
        -------
        pandas.DataFrame of shape (n_samples, n_features)
            Colonnes niveau_risque_<nom> avec l'index d'entrée. Les niveaux
            sont des catégories ordonnées déclarées dans LEVELS.

        Raises
        ------
        NotFittedError
            Si l'encodeur n'a pas encore été ajusté.
        TypeError
            Si X n'est pas un DataFrame.
        ValueError
            Si les colonnes ne correspondent pas au schéma appris.

        Notes
        -----
        Les modalités inconnues ou manquantes utilisent le taux global
        d'entraînement. Si aucun découpage discriminant n'a pu être appris
        pour une variable, son niveau reste manquant pour toutes les lignes.
        """
        return self._transform_levels(X)

    def _transform_levels(self, X):
        """Calcule les niveaux à partir des statistiques mémorisées.

        Implémentation interne de transform : vérifie le schéma de X,
        associe les taux appris aux modalités, puis applique les seuils.
        Renvoie un DataFrame de catégories ordonnées avec l'index de X.
        """
        check_is_fitted(self, ["rates_", "binners_", "global_mean_"])
        self._check_input(X, reset=False)
        labels = np.full((len(X), len(self.feature_names_in_)), None, dtype=object)
        for j, variable in enumerate(self.feature_names_in_):
            binner = self.binners_[variable]
            if binner is None or not len(X):
                continue
            rates = X[variable].astype(object).map(self.rates_[variable])
            rates = rates.astype(float).fillna(self.global_mean_)
            codes = binner.transform(rates.to_numpy().reshape(-1, 1)).ravel().astype(int)
            indices = self.bin_label_indices_[variable][codes]
            labels[:, j] = np.asarray(self.LEVELS, dtype=object)[indices]
        return self._output(X, labels)

    def fit_transform(self, X, y=None, **fit_params):
        """Encode l'entraînement hors pli, puis apprend les statistiques finales.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            Variables catégorielles d'entraînement, sans colonne is_fraud.
        y : array-like of shape (n_samples,), default=None
            Cible binaire requise. None est présent dans la signature pour
            l'interface sklearn mais provoque une erreur à l'exécution.
        **fit_params : dict
            Aucun paramètre supplémentaire n'est pris en charge.

        Returns
        -------
        pandas.DataFrame of shape (n_samples, n_features)
            Niveaux de risque calculés hors pli, avec l'index de X.
            L'instance conserve ensuite les statistiques apprises sur tout X
            pour les futurs appels à transform.

        Raises
        ------
        TypeError
            Si X n'est pas un DataFrame ou si fit_params n'est pas vide.
        ValueError
            Si X ou y est invalide, ou si cv n'est pas un entier Python
            compris entre 2 et le nombre de lignes.

        Notes
        -----
        Les taux et les quantiles de chaque pli sont appris uniquement
        sur les autres plis. Le KFold mélangé ne respecte pas la chronologie.
        Le résultat diffère donc de fit(X, y).transform(X).
        """
        # sklearn enveloppe cette méthode et transform pour set_output.
        # Garder le calcul dans des méthodes privées évite les références aux
        # imports du module dans ces points d'entrée modifiés par autoreload.
        return self._fit_transform_oof(X, y, **fit_params)

    def _fit_transform_oof(self, X, y=None, **fit_params):
        """Réalise le calcul hors pli et l'ajustement final de l'encodeur.

        Implémentation interne de fit_transform : valide X, y et cv,
        ajuste un clone sur chaque ensemble d'apprentissage, puis replace
        les niveaux du pli laissé de côté à leurs positions d'origine.
        Renvoie les catégories hors pli après avoir ajusté self sur tout X.
        Tout paramètre fourni dans fit_params est refusé.
        """
        if fit_params:
            raise TypeError("Aucun paramètre supplémentaire de fit n'est pris en charge.")
        self._check_input(X, reset=True)
        target = self._check_target(X, y)
        if isinstance(self.cv, bool) or not isinstance(self.cv, int) or not 2 <= self.cv <= len(X):
            raise ValueError("cv doit être un entier compris entre 2 et le nombre de lignes.")
        splitter = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        labels = np.full((len(X), len(self.feature_names_in_)), None, dtype=object)
        columns = [f"niveau_risque_{v}" for v in self.feature_names_in_]
        for train, valid in splitter.split(X):
            fold_encoder = clone(self).fit(X.iloc[train], target[train])
            labels[valid] = fold_encoder.transform(X.iloc[valid])[columns].to_numpy()
        self.fit(X, y)
        return self._output(X, labels)

    def get_feature_names_out(self, input_features=None):
        """Renvoie les noms des colonnes produites par le transformateur.

        Parameters
        ----------
        input_features : array-like of str, default=None
            Noms d'entrée facultatifs. S'ils sont fournis, leur nombre et,
            lorsque disponibles, leurs valeurs doivent correspondre au fit.

        Returns
        -------
        numpy.ndarray of shape (n_features,)
            Noms niveau_risque_<nom> dans l’ordre des colonnes apprises.

        Raises
        ------
        NotFittedError
            Si le transformateur n'a pas encore été ajusté.
        ValueError
            Si input_features ne correspond pas au schéma appris.
        """
        check_is_fitted(self, ["rates_", "feature_names_in_"])
        if input_features is not None and not np.array_equal(input_features, self.feature_names_in_):
            raise ValueError("input_features doit correspondre aux colonnes vues pendant fit.")
        return np.asarray([f"niveau_risque_{v}" for v in self.feature_names_in_], dtype=object)




class CustomIsolationForest(TransformerMixin, BaseEstimator):
    """Transforme les variables numériques en une colonne de score d'anomalie.

    Le modèle est entraîné uniquement dans ``fit``. ``transform`` réutilise
    ce modèle et renvoie ``-model_.decision_function(X)`` : un score plus
    élevé indique une observation plus atypique. Il ne s'agit pas d'une
    probabilité de fraude.

    Parameters
    ----------
    n_estimators : int, default=300
        Nombre d'arbres de l'Isolation Forest.
    max_samples : "auto", int or float, default="auto"
        Nombre ou proportion d'observations utilisés pour construire un arbre.
    contamination : "auto" or float, default=0.07
        Proportion d'anomalies attendue, utilisée pour définir le seuil du
        score. Une valeur numérique doit appartenir à l'intervalle ]0, 0.5].
    max_features : int or float, default=1.0
        Nombre ou proportion de variables utilisés pour construire un arbre.
    bootstrap : bool, default=False
        Active l'échantillonnage avec remise pour les arbres.
    n_jobs : int or None, default=-1
        Nombre de tâches parallèles pour l'entraînement ; -1 utilise tous
        les processeurs disponibles.
    random_state : int, RandomState or None, default=42
        Contrôle l'aléatoire de l'entraînement.

    Attributes
    ----------
    model_ : sklearn.ensemble.IsolationForest
        Modèle ajusté sur les données transmises à fit.
    n_features_in_ : int
        Nombre de variables d'entrée appris pendant fit.
    feature_names_in_ : numpy.ndarray
        Noms d'entrée lorsque X possède des noms de colonnes textuels.

    Notes
    -----
    La sortie contient uniquement le score, de forme (n_lignes, 1).
    ``set_output(transform="pandas")`` fournit un DataFrame avec la colonne
    ``score_anomalie`` et conserve l'index d'entrée. Dans un ColumnTransformer,
    utiliser une branche passthrough supplémentaire pour conserver aussi
    les variables numériques d'origine, par exemple amt.

    La classe s'utilise directement dans Pipeline, sans FunctionTransformer.
    Les paramètres sont accessibles via get_params/set_params et peuvent
    être optimisés avec GridSearchCV dans un pipeline comportant un modèle.
    """

    def __init__(self, n_estimators=300, max_samples="auto", contamination=0.07,
                 max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42):
        """Mémorise les hyperparamètres de la forêt, sans entraîner de modèle.

        Les paramètres et leurs valeurs par défaut sont décrits dans la
        docstring de la classe. model_ sera créé lors de l'appel à fit.
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state

    def __sklearn_tags__(self):
        """Déclare la prise en charge des matrices creuses auprès de sklearn.

        Returns
        -------
        Tags
            Capacités du transformateur avec l'entrée sparse activée.
        """
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags

    def fit(self, X, y=None, sample_weight=None):
        """Entraîne la forêt sur les variables numériques fournies.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Variables numériques finies ; un DataFrame est accepté.
            Les valeurs sont converties en float32 et les entrées creuses
            en format CSC pour l'apprentissage.
        y : object, default=None
            Ignoré ; accepté pour l'intégration dans un pipeline sklearn.
        sample_weight : array-like of shape (n_samples,), default=None
            Poids des observations transmis à IsolationForest.fit.

        Returns
        -------
        self : CustomIsolationForest
            Transformateur ajusté, dont model_ contient la forêt entraînée.

        Raises
        ------
        ValueError
            Si les données, les poids ou les hyperparamètres sont invalides.

        Notes
        -----
        Un nouvel appel remplace la forêt précédemment apprise.
        """
        X_checked = validate_data(
            self, X, reset=True, dtype=np.float32, accept_sparse="csc"
        )
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        ).fit(X_checked, sample_weight=sample_weight)
        return self

    def transform(self, X):
        """Calcule l'opposé de decision_function sans réentraîner la forêt.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Variables numériques finies correspondant au schéma du fit.
            Les noms des colonnes d'un DataFrame doivent être identiques
            et dans le même ordre. Les valeurs sont converties en float32.

        Returns
        -------
        numpy.ndarray or pandas.DataFrame of shape (n_samples, 1)
            Score d'anomalie : plus il est élevé, plus la ligne est atypique.
            Avec set_output(transform="pandas"), la colonne est nommée
            score_anomalie et l'index de X est conservé. Les variables
            d'entrée ne sont pas incluses dans la sortie.

        Raises
        ------
        NotFittedError
            Si fit n'a pas encore été appelé avec succès.
        ValueError
            Si les valeurs ou le schéma des variables sont invalides.
        """
        check_is_fitted(self, "model_")
        X_checked = validate_data(
            self, X, reset=False, dtype=np.float32, accept_sparse="csr"
        )
        return (-self.model_.decision_function(X_checked)).reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        """Renvoie les noms des colonnes produites par le transformateur.

        Parameters
        ----------
        input_features : array-like of str, default=None
            Noms d'entrée facultatifs. S'ils sont fournis, leur nombre et,
            lorsque disponibles, leurs valeurs doivent correspondre au fit.

        Returns
        -------
        numpy.ndarray of shape (1,)
            Tableau contenant uniquement le nom score_anomalie.

        Raises
        ------
        NotFittedError
            Si le transformateur n'a pas encore été ajusté.
        ValueError
            Si input_features ne correspond pas au schéma appris.
        """
        check_is_fitted(self, "model_")
        if input_features is not None:
            names = np.asarray(input_features, dtype=object)
            if names.ndim != 1 or len(names) != self.n_features_in_:
                raise ValueError("input_features doit contenir un nom par variable d'entrée.")
            if hasattr(self, "feature_names_in_") and not np.array_equal(names, self.feature_names_in_):
                raise ValueError("input_features doit correspondre aux colonnes vues pendant fit.")
        return np.asarray(["score_anomalie"], dtype=object)




class ThresholdCostOptimizer:
    """Choisit le seuil de classification qui minimise le coût des erreurs.

    Pour chaque seuil t, une transaction est prédite frauduleuse si sa
    probabilité de fraude est supérieure ou égale à t. Le coût total vaut
    ``cost_fp * FP + cost_fn * FN`` : FP désigne les transactions légitimes
    signalées à tort et FN les fraudes non détectées. Les bonnes prédictions
    ont un coût nul.

    Cette classe ne crée ni n'entraîne un modèle prédictif. Elle reçoit les
    probabilités d'un modèle déjà entraîné. Ajuster le seuil sur le jeu de
    validation, puis conserver ce seuil pour évaluer le jeu de test.

    Parameters
    ----------
    cost_fp : float, default=20
        Coût unitaire, fini et positif ou nul, d'un faux positif.
    cost_fn : float, default=100
        Coût unitaire, fini et positif ou nul, d'un faux négatif, exprimé
        dans la même unité que cost_fp.
    n_thresholds : int, default=200
        Nombre de seuils régulièrement espacés entre 0 et 1, bornes incluses.
        Doit être supérieur ou égal à 2. Le minimum est recherché sur cette
        grille ; il n'est pas nécessairement le minimum sur tous les seuils
        possibles. En cas d'égalité de coût, le plus petit seuil est retenu.

    Attributes
    ----------
    seuils_ : ndarray of shape (n_thresholds,) or None
        Seuils évalués, dans l'ordre croissant ; None avant fit.
    costs_ : ndarray of shape (n_thresholds,) or None
        Coûts totaux associés aux seuils ; None avant fit.
    best_threshold_ : float or None
        Seuil qui minimise le coût sur la grille ; None avant fit.
    best_cost_ : float or None
        Coût total au seuil retenu ; None avant fit.

    Notes
    -----
    Les probabilités de chaque classe réelle sont triées une seule fois.
    Une recherche dichotomique calcule ensuite FP et FN pour chaque seuil,
    sans construire une matrice de confusion à chaque itération ni une
    matrice observations × seuils. Pour n observations et m seuils, le calcul
    coûte O(n log n + m log n) en temps et O(n + m) en mémoire.

    Les scores bruts d'IsolationForest ne sont pas des probabilités et ne
    conviennent pas à cette grille [0, 1].

    Examples
    --------
    >>> optimizer = ThresholdCostOptimizer(cost_fp=20, cost_fn=125)
    >>> optimizer.fit([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
    ... # doctest: +ELLIPSIS
    <...ThresholdCostOptimizer object at ...>
    >>> optimizer.get_best_cost()
    0.0
    """

    def __init__(self, cost_fp=20, cost_fn=100, n_thresholds=200):
        """Mémorise les coûts et la taille de grille, sans rechercher de seuil.

        Les paramètres sont décrits dans la docstring de la classe.
        Les attributs de résultats restent à None jusqu'à un fit réussi ;
        la validation des paramètres est effectuée lors de cet appel.
        """
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn
        self.n_thresholds = n_thresholds
        self.seuils_ = None
        self.costs_ = None
        self.best_threshold_ = None
        self.best_cost_ = None

    def fit(self, y_true, y_proba):
        """Évalue les seuils et mémorise celui qui minimise le coût total.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            Étiquettes réelles : 0 pour une transaction légitime, 1 pour une
            fraude. Ne pas passer la matrice de variables X à cet argument.
        y_proba : array-like of shape (n_samples,)
            Probabilités de la classe 1, finies et comprises dans [0, 1],
            dans le même ordre que y_true. Avec un classifieur sklearn dont
            classes_ vaut [0, 1], utiliser predict_proba(X_validation)[:, 1].
            Si les deux entrées sont des Series pandas, leurs index doivent
            être identiques, dans le même ordre.

        Returns
        -------
        self : ThresholdCostOptimizer
            Objet ajusté, permettant le chaînage des appels.

        Raises
        ------
        ValueError
            Si les paramètres sont invalides, les entrées sont vides, ne sont
            pas des vecteurs de même longueur ou contiennent des valeurs
            invalides. Un jeu contenant une seule classe est accepté.
        """
        for name, value in (("cost_fp", self.cost_fp), ("cost_fn", self.cost_fn)):
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, float, np.integer, np.floating))
                    or not np.isfinite(value) or value < 0):
                raise ValueError(f"{name} doit être un nombre fini positif ou nul.")
        if (isinstance(self.n_thresholds, (bool, np.bool_))
                or not isinstance(self.n_thresholds, (int, np.integer))
                or self.n_thresholds < 2):
            raise ValueError("n_thresholds doit être un entier supérieur ou égal à 2.")
        if (isinstance(y_true, pd.Series) and isinstance(y_proba, pd.Series)
                and not y_true.index.equals(y_proba.index)):
            raise ValueError("Les index de y_true et y_proba doivent être alignés.")

        labels = np.asarray(y_true)
        if (labels.ndim != 1 or labels.size == 0 or pd.isna(labels).any()
                or not np.isin(labels, [0, 1]).all()):
            raise ValueError(
                "y_true doit être un vecteur non vide d'étiquettes 0/1, et non X."
            )
        try:
            probabilities = np.asarray(y_proba, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("y_proba doit contenir des probabilités numériques.") from exc
        if probabilities.ndim != 1 or probabilities.size != labels.size:
            raise ValueError("y_proba doit être un vecteur de même longueur que y_true.")
        if (not np.isfinite(probabilities).all()
                or ((probabilities < 0) | (probabilities > 1)).any()):
            raise ValueError("y_proba doit contenir des probabilités finies entre 0 et 1.")

        seuils = np.linspace(0.0, 1.0, self.n_thresholds)
        negatives = np.sort(probabilities[labels == 0])
        positives = np.sort(probabilities[labels == 1])
        # side='left' compte strictement p < seuil : p == seuil prédit 1.
        fp = negatives.size - np.searchsorted(negatives, seuils, side="left")
        fn = np.searchsorted(positives, seuils, side="left")
        costs = float(self.cost_fp) * fp + float(self.cost_fn) * fn
        idx_best = int(costs.argmin())

        self.seuils_ = seuils
        self.costs_ = costs
        self.best_threshold_ = float(seuils[idx_best])
        self.best_cost_ = float(costs[idx_best])
        return self

    def _check_fitted(self):
        """Vérifie que les seuils et les coûts ont été calculés.

        Raises
        ------
        RuntimeError
            Si aucun appel à fit n'a encore produit les résultats attendus.
        """
        if self.seuils_ is None or self.costs_ is None:
            raise RuntimeError("Appelle d'abord .fit(y_true, y_proba).")

    def plot_cost_curve(self):
        """Affiche la courbe des coûts et signale le meilleur seuil.

        Les coûts sont exprimés dans l'unité de cost_fp et cost_fn. Affiche
        également le seuil et son coût dans la sortie texte.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Axes de la figure, permettant d'adapter le graphique si besoin.

        Raises
        ------
        RuntimeError
            Si fit n'a pas encore été appelé avec succès.
        """
        self._check_fitted()
        best_t = self.best_threshold_
        best_cost = self.best_cost_
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.seuils_, self.costs_, label="Coût total")
        ax.axvline(best_t, color="red", linestyle="--",
                   label=f"Seuil optimal = {best_t:.3f}")
        ax.scatter([best_t], [best_cost], color="red")
        ax.annotate(
            f"t={best_t:.3f}\ncoût={best_cost:.2f}",
            xy=(best_t, best_cost), xytext=(0, 15), textcoords="offset points",
            ha="center", arrowprops={"arrowstyle": "->", "color": "red"},
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "red", "alpha": 0.8},
        )
        ax.set(xlabel="Seuil de décision", ylabel="Coût total (FP/FN)",
               title="Coût en fonction du seuil de décision")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()
        print(f"Meilleur seuil : {best_t:.3f} | Coût minimal : {best_cost:.2f}")
        return ax

    def get_best_threshold(self):
        """Renvoie le résultat mémorisé lors du dernier fit réussi.

        Returns
        -------
        float
            Seuil de décision retenu sur la grille entre 0 et 1.

        Raises
        ------
        RuntimeError
            Si fit n'a pas encore été appelé avec succès.
        """
        self._check_fitted()
        return self.best_threshold_

    def get_best_cost(self):
        """Renvoie le résultat mémorisé lors du dernier fit réussi.

        Returns
        -------
        float
            Coût total minimal, dans l’unité de cost_fp et cost_fn.

        Raises
        ------
        RuntimeError
            Si fit n'a pas encore été appelé avec succès.
        """
        self._check_fitted()
        return self.best_cost_




def display_roc_auc_score(y_train,proba_train,
                          y_validation,proba_validation,
                          y_test,proba_test):
    """Affiche les scores de ROC AUC sur les jeux train, validation et test.

    Parameters
    ----------
    y_train : array-like of shape (n_train,)
        Vraies classes 0/1 du jeu d’entraînement.
    proba_train : array-like of shape (n_train,)
        Probabilités de fraude entre 0 et 1, alignées sur y_train.
    y_validation : array-like of shape (n_validation,)
        Vraies classes 0/1 du jeu de validation.
    proba_validation : array-like of shape (n_validation,)
        Probabilités de fraude entre 0 et 1, alignées sur y_validation.
    y_test : array-like of shape (n_test,)
        Vraies classes 0/1 du jeu de test.
    proba_test : array-like of shape (n_test,)
        Probabilités de fraude entre 0 et 1, alignées sur y_test.

    Returns
    -------
    None
        Affiche trois scores dans la sortie standard, puis un séparateur.

    Notes
    -----
    Mesure la capacité à classer les fraudes devant les transactions légitimes.
    Un score plus élevé est préférable ; 0.5 correspond au classement aléatoire.
    Chaque jeu doit contenir les deux classes pour une AUC interprétable.
    Les scores sont affichés tels quels, sans conversion en pourcentage.
    La fonction ne réentraîne pas le modèle et ne choisit aucun seuil.
    """
    
    print("ROC AUC (train) :", roc_auc_score(y_train, proba_train))
    print("ROC AUC (valid) :", roc_auc_score(y_validation, proba_validation))
    print("ROC AUC (test) :",  roc_auc_score(y_test, proba_test))
    print("---")
    
    

def display_precision_score(y_train,pred_train,
                            y_validation,pred_validation,
                            y_test,pred_test):
    """Affiche les scores de précision sur les jeux train, validation et test.

    Parameters
    ----------
    y_train : array-like of shape (n_train,)
        Vraies classes 0/1 du jeu d’entraînement.
    pred_train : array-like of shape (n_train,)
        Classes 0/1 prédites après application du seuil, alignées sur y_train.
    y_validation : array-like of shape (n_validation,)
        Vraies classes 0/1 du jeu de validation.
    pred_validation : array-like of shape (n_validation,)
        Classes 0/1 prédites après application du seuil, alignées sur y_validation.
    y_test : array-like of shape (n_test,)
        Vraies classes 0/1 du jeu de test.
    pred_test : array-like of shape (n_test,)
        Classes 0/1 prédites après application du seuil, alignées sur y_test.

    Returns
    -------
    None
        Affiche trois scores dans la sortie standard, puis un séparateur.

    Notes
    -----
    Calcule VP / (VP + FP), la proportion de fraudes parmi les alertes.
    Un score plus élevé est préférable. Sans prédiction positive, sklearn
    renvoie 0 et émet un avertissement avec son réglage par défaut.
    Les scores sont affichés tels quels, sans conversion en pourcentage.
    La fonction ne réentraîne pas le modèle et ne choisit aucun seuil.
    """
    
    print("Précision (train) :", precision_score(y_train, pred_train))
    print("Précision (valid) :", precision_score(y_validation, pred_validation))
    print("Précision (test)  :",  precision_score(y_test, pred_test))
    print("---")
    


def display_recall_score(y_train,pred_train,
                        y_validation,pred_validation,
                        y_test,pred_test):
    """Affiche les scores de rappel sur les jeux train, validation et test.

    Parameters
    ----------
    y_train : array-like of shape (n_train,)
        Vraies classes 0/1 du jeu d’entraînement.
    pred_train : array-like of shape (n_train,)
        Classes 0/1 prédites après application du seuil, alignées sur y_train.
    y_validation : array-like of shape (n_validation,)
        Vraies classes 0/1 du jeu de validation.
    pred_validation : array-like of shape (n_validation,)
        Classes 0/1 prédites après application du seuil, alignées sur y_validation.
    y_test : array-like of shape (n_test,)
        Vraies classes 0/1 du jeu de test.
    pred_test : array-like of shape (n_test,)
        Classes 0/1 prédites après application du seuil, alignées sur y_test.

    Returns
    -------
    None
        Affiche trois scores dans la sortie standard, puis un séparateur.

    Notes
    -----
    Calcule VP / (VP + FN), la proportion de fraudes effectivement détectées.
    Un score plus élevé est préférable. Sans fraude réelle, sklearn renvoie
    0 et émet un avertissement avec son réglage par défaut.
    Les scores sont affichés tels quels, sans conversion en pourcentage.
    La fonction ne réentraîne pas le modèle et ne choisit aucun seuil.
    """
    
    print("Rappel (train) :", recall_score(y_train, pred_train))
    print("Rappel (valid) :", recall_score(y_validation, pred_validation))
    print("Rappel (test)  :",  recall_score(y_test, pred_test))
    print("---")
    
    
    
def display_log_loss(y_train,proba_train,
                    y_validation,proba_validation,
                    y_test,proba_test):
    """Affiche les scores de log loss sur les jeux train, validation et test.

    Parameters
    ----------
    y_train : array-like of shape (n_train,)
        Vraies classes 0/1 du jeu d’entraînement.
    proba_train : numpy.ndarray or pandas.Series of shape (n_train,)
        Probabilités de fraude entre 0 et 1, alignées sur y_train.
    y_validation : array-like of shape (n_validation,)
        Vraies classes 0/1 du jeu de validation.
    proba_validation : numpy.ndarray or pandas.Series of shape (n_validation,)
        Probabilités de fraude entre 0 et 1, alignées sur y_validation.
    y_test : array-like of shape (n_test,)
        Vraies classes 0/1 du jeu de test.
    proba_test : numpy.ndarray or pandas.Series of shape (n_test,)
        Probabilités de fraude entre 0 et 1, alignées sur y_test.

    Returns
    -------
    None
        Affiche trois scores dans la sortie standard, puis un séparateur.

    Notes
    -----
    Mesure la perte logarithmique moyenne ; un score plus faible est préférable.
    Reconstruit les probabilités [1 - p, p] pour les classes [0, 1].
    Chaque jeu doit contenir les deux classes car labels n’est pas spécifié.
    Les scores sont affichés tels quels, sans conversion en pourcentage.
    La fonction ne réentraîne pas le modèle et ne choisit aucun seuil.
    """
    
    print("Log Loss (train) :", log_loss(y_train, np.vstack([1-proba_train, proba_train]).T))
    print("Log Loss (valid) :", log_loss(y_validation, np.vstack([1-proba_validation, proba_validation]).T))
    print("Log Loss (test)  :",  log_loss(y_test, np.vstack([1-proba_test, proba_test]).T))
    print("---")