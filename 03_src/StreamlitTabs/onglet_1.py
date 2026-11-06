"""Présentation du projet, des transformations et de la décision de fraude."""

from collections.abc import Callable
from html import escape
import json
import math
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from google.auth.exceptions import GoogleAuthError
from google.auth.transport import requests
from google.oauth2 import id_token

import streamlit as st


URL_API = "https://api-fraud-detection-967845542272.europe-west1.run.app"


class ModelInfoError(RuntimeError):
    """Informations du modèle indisponibles ou non conformes."""


@st.cache_data(ttl=60, show_spinner=False)
def obtenir_informations_modele(api_url: str) -> dict:
    """Lit GET /model-info et garde une réponse valide en cache pendant 60 s.

    Renvoie model, version, seuil (entre 0 et 1) et variables. Le cache est
    distinct par URL de service. Les erreurs lèvent ModelInfoError et ne sont
    pas mises en cache ; aucun modèle n'est chargé dans l'interface.
    """
    try:
        token = id_token.fetch_id_token(requests.Request(), URL_API)
    except GoogleAuthError as exc:
        raise ModelInfoError(
            "Impossible d’obtenir le token d’authentification. Vérifiez les identifiants Google du service."
        ) from exc

    try:
        parsed = urlsplit(api_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError
        request = Request(
            api_url.rstrip("/") + "/model-info",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"}, method="GET",
        )
    except (TypeError, ValueError) as exc:
        raise ModelInfoError("L’adresse du service d’informations est invalide.") from exc

    try:
        with urlopen(request, timeout=5) as response:
            result = json.load(response)
    except HTTPError as exc:
        if exc.code == 401:
            message = "Authentification refusée (401) : le token est absent, invalide ou expiré."
        elif exc.code == 403:
            message = "Accès interdit (403) : vérifiez le token et les autorisations du compte de service pour appeler l’API."
        elif exc.code == 404:
            message = "La route d’informations du modèle est introuvable (404). Vérifiez l’adresse du service et la présence de /model-info."
        elif exc.code == 429:
            message = "Trop de requêtes vers le service d’informations (429). Patientez avant d’actualiser."
        elif 500 <= exc.code < 600:
            message = f"Le service d’informations a rencontré une erreur serveur ({exc.code}). Réessayez dans quelques instants."
        else:
            message = f"Le service ne peut pas fournir les informations du modèle (erreur HTTP {exc.code})."
        exc.close()
        raise ModelInfoError(message) from exc
    except TimeoutError as exc:
        raise ModelInfoError("Le service d’informations n’a pas répondu dans le délai de 5 secondes. Réessayez.") from exc
    except (URLError, OSError) as exc:
        if isinstance(exc, URLError) and isinstance(exc.reason, TimeoutError):
            message = "Le service d’informations n’a pas répondu dans le délai de 5 secondes. Réessayez."
        else:
            message = "Le service d’informations est injoignable. Vérifiez la connexion et que le service est démarré."
        raise ModelInfoError(message) from exc
    except (ValueError, UnicodeError) as exc:
        raise ModelInfoError("Le service a renvoyé des informations illisibles.") from exc

    if not isinstance(result, dict):
        raise ModelInfoError("Les informations reçues sont incomplètes ou invalides.")
    threshold = result.get("seuil")
    variables = result.get("variables")
    if (not isinstance(result.get("model"), str) or not result["model"].strip()
            or not isinstance(result.get("version"), str) or not result["version"].strip()
            or isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or not 0 < threshold < 1
            or not isinstance(variables, list)
            or not all(isinstance(name, str) and name.strip() for name in variables)):
        raise ModelInfoError("Les informations reçues sont incomplètes ou invalides.")
    return {key: result[key] for key in ("model", "version", "seuil", "variables")}


def afficher_informations_modele(api_url: str):
    """Affiche les métadonnées du service sans bloquer la présentation en cas d'erreur."""
    with st.container(border=True, key="model_info"):
        title, refresh = st.columns([3, 1], vertical_alignment="center")
        with title:
            st.subheader("Informations sur le modèle")
        with refresh:
            if st.button("Actualiser", icon=":material/refresh:", key="actualiser_modele", width="stretch"):
                obtenir_informations_modele.clear(api_url)
        try:
            info = obtenir_informations_modele(api_url)
        except ModelInfoError as exc:
            st.info(f"{exc} Vous pouvez toujours tester une transaction ou actualiser cet encadré.")
            return

        threshold = f"{100 * info['seuil']:.2f}".replace(".", ",")
        labels = {"amt": "Montant", "heure": "Heure", "jour": "Jour", "category": "Catégorie", "state": "État"}
        variables = "".join(
            f'<span class="model-variable">{escape(labels.get(name, name))} '
            f'<code>{escape(name)}</code></span>' for name in info["variables"]
        ) or '<span class="model-info-label">Aucune variable indiquée par le service.</span>'
        st.html(f'''
            <dl class="model-info-grid">
                <div><dt>Modèle</dt><dd>{escape(info['model'])}</dd></div>
                <div><dt>Version</dt><dd>{escape(info['version'])}</dd></div>
                <div><dt>Seuil de décision</dt><dd>{threshold} %</dd></div>
            </dl>
            <div class="model-inputs">
                <span class="model-info-label">Variables d’entrée</span>
                <div class="model-variables">{variables}</div>
            </div>
        ''')
        st.caption("Seuil communiqué par le service, affiché en pourcentage. La décision reste calculée par l’API.")


def afficher_onglet(ouvrir_outil: Callable[[], None], api_url: str):
    """Affiche les explications et un accès au formulaire de transaction.

    Parameters
    ----------
    ouvrir_outil : callable
        Callback de navigation vers le deuxième onglet.
    api_url : str
        URL de base du service fournissant les informations via GET /model-info.
    """
    st.html('''
        <div class="hero">
            <span class="eyebrow">PRÉSENTATION DU PROJET</span>
            <h1>Comprendre la décision.</h1>
            <p>Cinq informations sur une transaction, une estimation du risque de fraude.
            Découvrez comment le modèle passe des données à la décision.</p>
        </div>
        <div class="overview-strip">
            <div><strong>5</strong><span>informations à renseigner</span></div>
            <div><strong>18</strong><span>variables après transformation</span></div>
            <div><strong>1</strong><span>décision expliquée</span></div>
        </div>
    ''')

    with st.container(border=True):
        st.subheader("À quoi sert cette application ?")
        st.write(
            "Simulez une transaction bancaire américaine et observez si le modèle "
            "la considère comme frauduleuse. Vous obtenez une décision d’acceptation "
            "ou de refus et sa probabilité de fraude estimée."
        )
        st.button(
            "Tester une transaction", type="primary", icon=":material/arrow_forward:",
            on_click=ouvrir_outil, key="ouvrir_simulateur",
        )

    afficher_informations_modele(api_url)

    st.subheader("Du formulaire à la décision")
    steps = [
        ("01", "Estimer les niveaux de risque",
         "Pour l’heure, le jour, la catégorie et l’État, le modèle réutilise les taux "
         "de fraude appris sur l’entraînement. Ces taux sont répartis par quantiles "
         "en quatre niveaux : très faible, faible, modéré et élevé."),
        ("02", "Préparer les variables",
         "Les quatre niveaux de risque sont encodés en 16 indicatrices numériques "
         ". Le montant est également conservé."),
        ("03", "Mesurer le caractère atypique",
         "L'algorithme Isolsation Forest apprend les profils des transactions d’entraînement. "
         "Il ajoute un score d’anomalie aux 17 variables précédentes. "
         "Plus ce score est élevé, plus la transaction est atypique. Ce score n’est pas une probabilité."),
        ("04", "Calculer la probabilité",
         "Le classifieur XGBoost utilise ces 18 variables pour estimer une probabilité "
         "de fraude. Le service de prédiction la compare ensuite au seuil choisi pour rendre sa décision."),
    ]
    for row in (steps[:2], steps[2:]):
        for column, (number, title, description) in zip(st.columns(2, gap="medium"), row):
            with column:
                st.html(f'''
                    <article class="step-card">
                        <span class="step-number">{number}</span>
                        <h3>{title}</h3><p>{description}</p>
                    </article>
                ''')
    st.caption(
        "Les niveaux d’entraînement sont construits hors pli pour que la cible d’une "
        "transaction ne serve pas à son propre encodage. Lors d’un test dans cette "
        "application, les transformations apprises sont réutilisées sans réentraînement."
    )

    st.caption("Une probabilité est une estimation du modèle, pas une certitude sur la transaction.")
