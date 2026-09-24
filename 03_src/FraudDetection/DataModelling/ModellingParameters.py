"""Fabriques des encodeurs, du classifieur et de l’optimiseur de seuil du projet."""

from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from FraudDetection.DataModelling.ModellingTools import RiskLevelEncoder, CustomIsolationForest, ThresholdCostOptimizer

# ------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------ 

def risk_level_encoder():
    """Crée un encodeur de risque non ajusté avec cinq plis internes.

    Returns
    -------
    RiskLevelEncoder
        Nouvelle instance avec cv=5 et les autres paramètres par défaut.
        Chaque colonne reçue est remplacée par niveau_risque_<nom>.
        fit_transform utilise un encodage hors pli sur l'entraînement ;
        transform réutilise les statistiques apprises, sans cible.
    """
    
    risk_level_encoder = RiskLevelEncoder(cv = 5)
    
    return risk_level_encoder



def one_hot_encoder():
    """Crée l'encodage indicateur des quatre variables de niveau de risque.

    Returns
    -------
    OneHotEncoder
        Encodeur non ajusté pour quatre colonnes, chacune ayant les
        catégories tres_faible, faible, moderee et elevee, dans cet ordre.
        Aucune catégorie n'est supprimée : la sortie dense comprend
        16 indicatrices. Une catégorie inconnue provoque une erreur.

    Notes
    -----
    Le nombre de colonnes attendu est fixé à quatre par categories.
    Adapter cette configuration si le nombre de variables de risque change.
    """
    
    niveaux_risque = ["tres_faible", "faible", "moderee", "elevee"]
    
    one_hot_encoder = OneHotEncoder(
        categories = [niveaux_risque] * 4,
        drop = None,
        sparse_output = False,
        handle_unknown = "error",
    )
    
    return one_hot_encoder



def custom_isolation_forest():
    """Crée le transformateur qui calcule le score d'anomalie des transactions.

    Returns
    -------
    CustomIsolationForest
        Instance non ajustée avec 300 arbres, max_samples="auto" et
        contamination=0.01. Les autres paramètres gardent leurs défauts.
        Après fit, transform produit uniquement score_anomalie, égal à
        l'opposé de decision_function : un score plus élevé indique
        une transaction plus atypique, sans être une probabilité de fraude.
    """
    
    custom_isolation_forest = CustomIsolationForest(
        n_estimators = 300,
        max_samples = "auto",
        contamination = 0.006,
    )
    
    return custom_isolation_forest



def xgboost():
    """Crée le classifieur XGBoost avec les hyperparamètres du projet.

    Returns
    -------
    XGBClassifier
        Nouvelle instance non entraînée. La configuration fixe notamment
        305 arbres, une profondeur maximale de 8, le taux d'apprentissage,
        les proportions d'échantillonnage et les pénalités de régularisation.
        Appeler fit sur les variables préparées avant toute prédiction.

    Notes
    -----
    Cette fonction ne réalise aucune recherche d'hyperparamètres ;
    elle instancie les valeurs déclarées dans son corps.
    """
    
    xgboost = XGBClassifier(
        colsample_bytree = 0.8736932106048627,
        gamma = 0.010299214204601972,
        learning_rate = 0.014413697528610409,
        max_depth = 8,
        min_child_weight = 4,
        n_estimators = 305,
        reg_alpha = 0.014899847475658236,
        reg_lambda = 0.20399223998555607,
        subsample = 0.902144564127061,
    )
    
    return xgboost



def threshold_cost_optimizer():
    """Crée l'optimiseur du seuil de décision selon le coût des erreurs.

    Returns
    -------
    ThresholdCostOptimizer
        Instance non ajustée avec un coût de 20 par faux positif, de 125
        par faux négatif et 200 seuils régulièrement espacés entre 0 et 1.
        Le coût total est donc 20 * FP + 125 * FN, dans l'unité choisie.

    Notes
    -----
    Passer les étiquettes et les probabilités de fraude de la validation
    à fit, puis conserver le seuil choisi pour l'évaluation sur le test.
    En cas d'égalité de coût, le plus petit seuil de la grille est retenu.
    """
    
    cost_optimizer = ThresholdCostOptimizer(
        cost_fp = 20,
        cost_fn = 125,
        n_thresholds = 200,
    )
    
    return cost_optimizer