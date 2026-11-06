"""Formulaire de simulation et client HTTP de l'API de prédiction.

L'interface ne charge aucun modèle : la décision et le pourcentage affichés
proviennent de POST /predict. Les libellés français sont uniquement visuels.
"""

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

JOURS = {
    "Monday": "Lundi", "Tuesday": "Mardi", "Wednesday": "Mercredi",
    "Thursday": "Jeudi", "Friday": "Vendredi", "Saturday": "Samedi", "Sunday": "Dimanche",
}

CATEGORIES = {
    "grocery_pos": "Courses en magasin", "grocery_net": "Courses en ligne",
    "food_dining": "Restaurants et repas", "gas_transport": "Carburant et transport",
    "shopping_pos": "Achats en magasin", "shopping_net": "Achats en ligne",
    "entertainment": "Loisirs et divertissement", "health_fitness": "Santé et sport",
    "home": "Maison", "kids_pets": "Enfants et animaux", "personal_care": "Soins personnels",
    "travel": "Voyages", "misc_pos": "Autres achats en magasin", "misc_net": "Autres achats en ligne",
}

ETATS = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "Californie", "NC": "Caroline du Nord", "SC": "Caroline du Sud",
    "CO": "Colorado", "CT": "Connecticut", "ND": "Dakota du Nord", "SD": "Dakota du Sud",
    "DE": "Delaware", "DC": "District de Columbia", "FL": "Floride", "GA": "Géorgie",
    "HI": "Hawaï", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiane", "ME": "Maine",
    "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska",
    "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NY": "New York",
    "NM": "Nouveau-Mexique", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvanie", "RI": "Rhode Island", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginie", "WV": "Virginie-Occidentale",
    "WA": "Washington", "WI": "Wisconsin", "WY": "Wyoming",
}

DEFAULT_TRANSACTION = {"amt": 52.80, "heure": 18, "jour": "Monday", "category": "grocery_pos", "state": "CA"}



class PredictionError(RuntimeError):
    """Erreur du service de prédiction présentable à l'utilisateur."""


def demander_prediction(transaction: dict, api_url: str) -> dict:
    """Envoie une transaction à l'API et valide la décision reçue.

    Parameters
    ----------
    transaction : dict
        Objet contenant amt, heure, jour, category et state ; ce n'est pas une liste.
    api_url : str
        URL de base du service, par exemple http://127.0.0.1:8000.

    Returns
    -------
    dict
        Clés prediction (0 ou 1) et probabilite (pourcentage entre 0 et 100).
        L'API renvoie déjà un pourcentage : aucune multiplication par 100.

    Raises
    ------
    PredictionError
        Si le token ne peut pas être obtenu, si le service est indisponible,
        refuse la requête, dépasse le délai de 12 secondes ou renvoie une
        réponse non conforme.
    """
    try:
        token = id_token.fetch_id_token(requests.Request(), URL_API)
    except GoogleAuthError as exc:
        raise PredictionError(
            "Impossible d’obtenir le token d’authentification. Vérifiez les identifiants Google du service."
        ) from exc

    try:
        parsed = urlsplit(api_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError
        request = Request(
            api_url.rstrip("/") + "/predict",
            data=json.dumps(transaction, allow_nan=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
    except (ValueError, TypeError) as exc:
        raise PredictionError("La requête ne peut pas être préparée. Vérifiez les valeurs et la configuration du service.") from exc
    
    
    try:
        with urlopen(request, timeout=12) as response:
            result = json.load(response)
    except HTTPError as exc:
        if exc.code == 401:
            message = "Authentification refusée (401) : le token est absent, invalide ou expiré."
        elif exc.code == 403:
            message = "Accès interdit (403) : vérifiez le token et les autorisations du compte de service pour appeler l’API."
        elif exc.code == 422:
            message = "Le service n’a pas pu valider cette transaction. Vérifiez les champs puis réessayez."
        else:
            message = "Le service de prédiction a rencontré une erreur. Réessayez dans quelques instants."
        exc.close()
        raise PredictionError(message) from exc
    except (TimeoutError, URLError, OSError) as exc:
        raise PredictionError("Le service de prédiction est injoignable ou met trop de temps à répondre. Vérifiez qu’il est démarré, puis réessayez.") from exc
    except (ValueError, UnicodeError) as exc:
        raise PredictionError("Le service a renvoyé une réponse illisible. Aucune décision ne peut être affichée.") from exc


    prediction = result.get("prédiction") if isinstance(result, dict) else None
    probability = result.get("probabilité") if isinstance(result, dict) else None
    
    if (type(prediction) is not int or prediction not in (0, 1)
            or isinstance(probability, bool) or not isinstance(probability, (int, float))
            or not math.isfinite(probability) or not 0 <= probability <= 100):
        raise PredictionError("La réponse du service est incomplète ou invalide. Aucune décision ne peut être affichée.")
    
    
    return {"prediction": prediction, "probabilite": float(probability)}




def afficher_resultat(resultat: dict):
    """Affiche la décision et le récapitulatif de la dernière transaction testée."""
    accepted = resultat["prediction"] == 0
    style = "accepted" if accepted else "refused"
    title = "La transaction est acceptée" if accepted else "La transaction est refusée"
    symbol = "✓" if accepted else "×"
    probability = f"{resultat['probabilite']:.2f}".replace(".", ",")
    transaction = resultat["transaction"]
    amount = f"{transaction['amt']:,.2f}".replace(",", " ").replace(".", ",")
    recap = escape(
        f"{amount} USD · {JOURS[transaction['jour']]} à {transaction['heure']:02d} h · "
        f"{CATEGORIES[transaction['category']]} · {ETATS[transaction['state']]}"
    )
    st.html(f'''
        <section class="result-card {style}" role="status" aria-live="polite">
            <span class="eyebrow">DERNIÈRE TRANSACTION TESTÉE</span>
            <div class="result-body">
                <div class="result-decision"><span class="result-symbol" aria-hidden="true">{symbol}</span>
                    <h2>{title}</h2></div>
                <div class="result-probability"><span>Probabilité de fraude</span>
                    <strong>{probability}<small> %</small></strong></div>
            </div>
            <p class="result-recap">{recap}</p>
        </section>
    ''')




def afficher_onglet(api_url: str):
    """Affiche le formulaire de transaction et son résultat.

    La route de prédiction est appelée uniquement à la soumission du formulaire.
    Conserve la dernière transaction soumise et sa réponse dans la session.
    En cas d'échec d'une nouvelle analyse, l'ancien résultat est effacé pour
    qu'il ne puisse pas être confondu avec une réponse à la nouvelle demande.
    """
    st.html('''
        <div class="hero">
            <span class="eyebrow">UTILISATION DE L’OUTIL</span>
            <h1>Testez une transaction.</h1>
            <p>Créez votre transaction et découvrez si le modèle la considérerait comme
            frauduleuse. Le résultat affiche sa décision et la probabilité de fraude estimée.</p>
        </div>
    ''')

    draft = st.session_state.get("derniere_transaction", DEFAULT_TRANSACTION)
    with st.form("transaction_form", border=True):
        st.subheader("Les détails de votre transaction")
        st.caption("Les cinq champs sont nécessaires à l’analyse. Le montant est exprimé en dollars américains.")
        left, right = st.columns(2, gap="large")
        with left:
            amount = st.number_input(
                "Montant (USD)", min_value=0.1, value=float(draft["amt"]), step=1.0,
                format="%.2f", key="transaction_amt", help="Montant libre, positif ou nul, sans plafond imposé.",
            )
            day = st.selectbox(
                "Jour", options=list(JOURS), index=list(JOURS).index(draft["jour"]),
                format_func=JOURS.__getitem__, key="transaction_jour",
            )
            state = st.selectbox(
                "État américain", options=list(ETATS), index=list(ETATS).index(draft["state"]),
                format_func=lambda code: f"{ETATS[code]} ({code})", key="transaction_state",
            )
        with right:
            hour = st.select_slider(
                "Heure", options=list(range(24)), value=draft["heure"],
                format_func=lambda value: f"{value:02d} h", key="transaction_heure",
            )
            category = st.selectbox(
                "Catégorie d’achat", options=list(CATEGORIES),
                index=list(CATEGORIES).index(draft["category"]),
                format_func=CATEGORIES.__getitem__, key="transaction_category",
            )
            st.caption("Le jour, l’heure et le type d’achat aident à situer cette transaction dans son contexte.")
        st.write("")
        submitted = st.form_submit_button(
            "Tester la transaction", type="primary", icon=":material/arrow_forward:",
            width="stretch", key="tester_transaction",
        )

    if submitted:
        st.session_state.pop("resultat_transaction", None)
        st.session_state.pop("erreur_prediction", None)
        if amount is None or not math.isfinite(amount) or amount < 0:
            st.session_state["erreur_prediction"] = "Saisissez un montant valide, positif ou nul."
        else:
            transaction = {"amt": float(amount), "heure": int(hour), "jour": day, "category": category, "state": state}
            st.session_state["derniere_transaction"] = transaction
            try:
                with st.spinner("Analyse de votre transaction…"):
                    prediction = demander_prediction(transaction, api_url)
                st.session_state["resultat_transaction"] = {**prediction, "transaction": transaction.copy()}
            except PredictionError as exc:
                st.session_state["erreur_prediction"] = str(exc)

    if st.session_state.get("erreur_prediction"):
        st.error(st.session_state["erreur_prediction"], icon=":material/error:")
    elif st.session_state.get("resultat_transaction"):
        afficher_resultat(st.session_state["resultat_transaction"])
    else:
        st.html('''
            <div class="empty-result"><strong>Votre résultat apparaîtra ici.</strong>
                <p>Renseignez une transaction puis lancez son analyse.</p></div>
        ''')
