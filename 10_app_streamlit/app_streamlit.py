"""Point d'entrée Streamlit : thème, navigation et affichage des deux onglets.

Depuis la racine du projet :
    streamlit run 10_app_streamlit/app_streamlit.py

FRAUD_API_URL permet de remplacer http://127.0.0.1:8080 pour les deux routes :
GET /model-info (informations du modèle) et POST /predict (simulation).
"""

import os
from pathlib import Path
import sys

import streamlit as st

# Résolution indépendante du dossier depuis lequel le script est exécuté.
APP_DIR = Path(__file__).resolve().parent
SOURCE_DIR = APP_DIR.parent / "03_src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from StreamlitTabs import onglet_1, onglet_2


def ouvrir_outil():
    """Sélectionne le formulaire depuis le bouton de la présentation."""
    st.session_state["navigation"] = "Utilisation de l’outil"



def main():
    """Configure l'application et affiche la page choisie dans la sidebar."""
    st.set_page_config(
        page_title="FraudLens",
        page_icon="👁️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.html(f"<style>{(APP_DIR / 'style.css').read_text(encoding='utf-8')}</style>")

    with st.sidebar:
        st.html('''
            <div class="brand">
                <span class="brand-mark" aria-hidden="true">👁️</span>
                <div><strong>FraudLens</strong><span>Détection de fraude</span></div>
            </div>
            <p class="nav-label">VOTRE ESPACE</p>
        ''')
        page = st.radio(
            "Navigation",
            ["Présentation du projet", "Utilisation de l’outil"],
            key="navigation",
            label_visibility="collapsed",
        )
        st.divider()
        st.html('''
            <div class="sidebar-note">
                <strong>Comprendre. Tester. Explorer.</strong>
                <p>Une démonstration de détection de fraude par apprentissage automatique.</p>
                <span>Aucune transaction réelle n’est exécutée.</span>
            </div>
        ''')

    # Les deux onglets utilisent le même service pour les informations et la prédiction.
    api_url = os.environ.get("FRAUD_API_URL", "http://127.0.0.1:8080")
    if page == "Présentation du projet":
        onglet_1.afficher_onglet(ouvrir_outil, api_url)
    else:
        onglet_2.afficher_onglet(api_url)



if __name__ == "__main__":
    main()
