"""Présentation du projet, des transformations et de la décision de fraude."""

from collections.abc import Callable

import streamlit as st



def afficher_onglet(ouvrir_outil: Callable[[], None]):
    """Affiche les explications et un accès au formulaire de transaction.

    Parameters
    ----------
    ouvrir_outil : callable
        Callback de navigation vers le deuxième onglet.
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

    st.subheader("Les cinq informations d’entrée")
    st.table([
        {"Variable": "Montant · amt", "Description": "Montant de la transaction, en dollars américains (USD)."},
        {"Variable": "Heure · heure", "Description": "Heure de la transaction, de 0 à 23."},
        {"Variable": "Jour · jour", "Description": "Jour de la semaine ; le libellé français est traduit pour le modèle."},
        {"Variable": "Catégorie · category", "Description": "Type d’achat : courses, restaurant, transport, achats en ligne…"},
        {"Variable": "État · state", "Description": "État américain de la transaction, transmis sous son code à deux lettres."},
    ])

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

    st.subheader("Pourquoi ce seuil de décision ?")
    with st.container(border=True):
        st.write(
            "Le seuil ne vaut pas automatiquement 50 %. Il est choisi sur le jeu de "
            "validation en comparant 200 seuils entre 0 et 1 pour minimiser le coût des erreurs."
        )
        left, right = st.columns(2)
        with left:
            st.metric("Fausse alerte", "20$", help="Coût unitaire d’un faux positif.")
            st.caption("Une transaction légitime est signalée comme frauduleuse.")
        with right:
            st.metric("Fraude manquée", "125$", help="Coût unitaire d’un faux négatif.")
            st.caption("Une transaction frauduleuse n’est pas détectée.")
        st.html('''
            <div class="cost-formula">Coût total = 20 × fausses alertes + 125 × fraudes manquées</div>
        ''')
        st.caption(
            "Coûts exprimés dans une même unité, selon la configuration de cette version du projet. "
            "Le seuil enregistré par le service de prédiction fait foi pour chaque simulation."
        )
        st.write(
            "Pour séparer apprentissage et évaluation : train en 2019, validation de janvier "
            "à septembre 2020, puis test d’octobre à décembre 2020."
        )

    st.subheader("Comment lire les résultats ?")
    st.table([
        {"Sortie": "Probabilité de fraude", "Lecture": "Estimation affichée de 0 à 100 %. Un pourcentage plus élevé indique un risque estimé plus élevé."},
        {"Sortie": "Prédiction · 0 ou 1", "Lecture": "0 : transaction acceptée. 1 : transaction refusée. La décision dépend du seuil du service, pas d’un seuil fixé à 50 %."},
        {"Sortie": "Message", "Lecture": "La décision est mise en évidence en vert clair ou en rouge clair."},
    ])
    st.caption("Une probabilité est une estimation du modèle, pas une certitude sur la transaction.")
