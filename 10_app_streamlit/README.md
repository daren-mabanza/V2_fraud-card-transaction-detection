# Interface Streamlit — FraudLens

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
Le rechargement à l'enregistrement est activé par `server.runOnSave`.
Après une modification de cette configuration, redémarrer Streamlit une fois.

En développement, afficher également les outils de rechargement et de cache :

```bash
.venv/bin/streamlit run 10_app_streamlit/app_streamlit.py --client.toolbarMode developer
```

Pour changer l'adresse du service :

```bash
FRAUD_API_URL=http://127.0.0.1:8000 .venv/bin/streamlit run 10_app_streamlit/app_streamlit.py
```

`FRAUD_API_URL` désigne l'URL de base, sans `/predict` ni `/model-info`. Avec un futur déploiement,
`127.0.0.1` désignera la machine exécutant Streamlit.

- `app_streamlit.py` configure la page et la navigation.
- `StreamlitTabs/onglet_1.py` explique le projet et affiche les informations du modèle.
- `StreamlitTabs/onglet_2.py` affiche le formulaire et le résultat de la prédiction.
- `GET /model-info` utilise un délai de 5 secondes et un cache de 60 secondes par URL.
- `POST /predict` est appelé uniquement à la soumission, avec un délai de 12 secondes.
- Aucun modèle n'est chargé dans Streamlit ; seule l'API réalise l'inférence.

Les libellés des menus sont français ; les codes transmis restent ceux de
l'entraînement. La réponse attendue contient `prédiction` (0 ou 1) et
`probabilité` (pourcentage de 0 à 100, déjà converti par l'API).
La décision affichée provient de `prédiction`, sans recalculer de seuil côté interface.
Si le modèle, ses modalités ou ses coûts changent, actualiser les explications et menus correspondants.

Dans « Présentation du projet », l'encadré « Informations sur le modèle » apparaît
entre « À quoi sert cette application ? » et « Du formulaire à la décision ». Il affiche `model`, `version`, `seuil` et
`variables`, issus de la réponse validée par `ModelInfoOutput`.
Le seuil de cette route est entre 0 et 1 et est converti en pourcentage pour
l'affichage, contrairement à `probabilité` qui est déjà un pourcentage.

Le bouton **Actualiser** force une nouvelle lecture des informations sans attendre
l'expiration du cache. Un échec de cette lecture affiche un message et laisse le
formulaire disponible.

Pour vérifier le service depuis un terminal :

```bash
curl http://127.0.0.1:8000/model-info
```

Si l'encadré n'apparaît pas, vérifier que le fichier lancé est bien
`10_app_streamlit/app_streamlit.py`, redémarrer Streamlit puis ouvrir
« Présentation du projet ». Si l'encadré apparaît mais indique une indisponibilité,
vérifier que l'API tourne à l'adresse `FRAUD_API_URL`, puis cliquer sur **Actualiser**.
