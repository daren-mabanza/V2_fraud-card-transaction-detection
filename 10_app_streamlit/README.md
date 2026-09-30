# Interface Streamlit — Vigie

Depuis la racine du projet, lancer deux terminaux avec le `.venv` existant.

API :

```bash
.venv/bin/uvicorn app_fastapi:app --app-dir 09_app_fastapi --host 127.0.0.1 --port 8000
```

Interface :

```bash
.venv/bin/streamlit run 10_app_streamlit/app_streamlit.py
```

Ouvrir `http://localhost:8501`. Le thème du projet est dans `.streamlit/config.toml` ;
les styles complémentaires sont dans `10_app_streamlit/style.css`.

Pour changer l'adresse du service :

```bash
FRAUD_API_URL=http://127.0.0.1:8000 .venv/bin/streamlit run 10_app_streamlit/app_streamlit.py
```

`FRAUD_API_URL` désigne l'URL de base, sans `/predict`. Avec un futur déploiement,
`127.0.0.1` désignera la machine exécutant Streamlit.

- `app_streamlit.py` configure la page et la navigation.
- `StreamlitTabs/onglet_1.py` explique le projet et la politique de coût actuelle.
- `StreamlitTabs/onglet_2.py` affiche le formulaire et appelle l'API, avec un délai de 12 secondes.
- Aucun modèle n'est chargé dans Streamlit ; seule l'API réalise l'inférence.

Les libellés des menus sont français ; les codes transmis restent ceux de
l'entraînement. La réponse attendue contient `prédiction` (0 ou 1) et
`probabilité` (pourcentage de 0 à 100, déjà converti par l'API).
La décision affichée provient de `prédiction`, sans recalculer de seuil côté interface.
Si le modèle, ses modalités ou ses coûts changent, actualiser les explications et menus correspondants.

Vérifications sans serveur ni réentraînement :

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_streamlit_app.py'
```
