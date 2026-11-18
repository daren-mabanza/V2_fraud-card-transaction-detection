"""Transformateurs sklearn, optimisation du seuil et affichage des métriques binaires."""

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
