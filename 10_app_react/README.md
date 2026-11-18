# FraudLens — interface React

Deux onglets, conservés dans `03_src/ReactTabs`, assemblés par `10_app_react/app_react.tsx` : présentation du projet et démonstration sur dix paiements sélectionnés selon la méthode du notebook 04. Le style suit la spécification PDF. Aucun réentraînement n’est effectué.

## Lancement local, sans token

Prérequis : Node.js **24 LTS**, npm et l’environnement Python du projet.

Depuis la racine, lancer FastAPI dans un terminal :

```bash
uv run uvicorn 09_app_fastapi.app_fastapi:app --host 127.0.0.1 --port 8080
```

Dans un second terminal :

```bash
cd 10_app_react
npm ci
npm run dev
```

Ouvrir <http://localhost:5173>. Au chargement, React récupère les exports existants via FastAPI. L’onglet Démonstration envoie les dix observations à `/batch-predict` uniquement au clic, puis une transaction à `/shap-explanation` lorsque « Expliquer » est sélectionné.

L’URL FastAPI est configurable dans `10_app_react/.env` (copier `.env.example`) ou par variable d’environnement :

```bash
FRAUD_API_URL=http://127.0.0.1:8000 npm run dev
```

## Variante avec token GCP

Même interface, même lot, même code React. Le serveur Node obtient un **ID token** pour l’audience de l’API et l’ajoute à l’appel FastAPI. Le navigateur reçoit seulement les résultats. Aucun token ni fichier d’identifiants n’est inclus dans le bundle React.

```bash
FRAUD_API_URL=https://mon-api-xxxxx-ew.a.run.app npm run dev:gcp
```

`google-auth-library` utilise les identifiants du service. Sur Cloud Run, attacher au frontend un compte de service ayant le rôle `roles/run.invoker` sur l’API. `FRAUD_API_AUDIENCE` permet de préciser l’audience si l’adresse d’appel diffère de l’URL du service. En local, fournir des identifiants compatibles avec la génération d’ID tokens ; les identifiants ADC d’un utilisateur ne suffisent pas toujours à cet usage.

## Construction et exécution de production

```bash
npm run build
PORT=5173 npm start
# ou
PORT=5173 npm run start:gcp
```

Le serveur sert les fichiers React et relaie `/api/health`, `/api/demo-data`, `/api/business-metrics`, `/api/batch-predict` et `/api/shap-explanation` vers FastAPI. Les deux variantes utilisent ce relais de même origine : aucune configuration CORS n’est nécessaire pour ce parcours.

## Docker : une image, deux modes

Depuis la racine du dépôt :

```bash
docker build -f 10_app_react/Dockerfile -t fraudlens-react .
```

Exemple local sous Linux, si FastAPI écoute sur le port 8080 de l’hôte :

```bash
docker run --rm --add-host=host.docker.internal:host-gateway \
  -p 5173:5173 \
  -e FRAUD_API_URL=http://host.docker.internal:8080 \
  -e FRAUD_AUTH_MODE=none fraudlens-react
```

Dans ce cas, démarrer FastAPI avec `--host 0.0.0.0`. Pour GCP, déployer la même image avec `FRAUD_AUTH_MODE=gcp`, `FRAUD_API_URL` en HTTPS et le compte de service configuré.

L’image définit `PORT=5173` par défaut. Pour utiliser le port interne 8080, fournir `-e PORT=8080` et publier `-p 5173:8080`. Sur Cloud Run, le serveur respecte la variable `PORT` injectée par la plateforme.

Le fichier `.env` local n’est pas chargé par le démarrage Docker (`node server.mjs`). Utiliser `-e NOM=valeur` pour chaque variable ou `--env-file /chemin/vers/config.env` pour transmettre un fichier adapté au conteneur.

## Lecture des exports existants

- `GET /demo-data` lit `01_data/04_test_data/observations_2025_04_21.parquet`, trie les dix paiements chronologiquement et renvoie leurs identifiants, dates UTC et 71 features. Les labels et les autres données personnelles ne sont pas transmis.
- `GET /business-metrics` lit directement `{Full|Sample}_metriques_simulation_metier.csv` et `{Full|Sample}_parametres_export_metier.json` dans `07_artifacts`, selon `11_configs/paths.yaml`. Les taux existants sont utilisés tels quels ; seule leur moyenne arithmétique d’affichage est calculée, chaque mois ayant le même poids.

Il n’y a plus de JSON de données à régénérer dans React. Après les étapes 4 et 5 d’export Python, **recharger la page suffit**, sans reconstruction du frontend. Les fichiers sont relus à chaque requête. Après réentraînement ou nouvel export SHAP, redémarrer FastAPI pour charger le nouveau pipeline et l’explainer. S’ils manquent ou sont incomplets, un message et un bouton de réessai remplacent les données ; aucun ancien résultat n’est utilisé comme secours.

En Docker, ces exports sont copiés dans l’image **API**, avec les modèles. Après modification des fichiers locaux, reconstruire l’image API ou monter les dossiers d’exports en volumes. L’image React reste indépendante des données.

Pour exporter les dix observations en Parquet depuis la racine :

```bash
uv run python 04_scripts/step_5_export_test_data.py
```

Les métriques sont exportées séparément avec `uv run python 04_scripts/step_4_export_metrics.py`. La sélection des dix observations reste définie dans `03_src/FraudDetection/DataModelling/Export/TestData.py`. Ce lot sert à illustrer les explications, pas à mesurer les performances.

## Comportements importants

Les identifiants restent côté interface. Les scores sont associés à l’ordre initial avant le tri du tableau. Masquer une colonne ne modifie jamais les features envoyées. Les changements d’onglet conservent les scores et l’explication. Une nouvelle sélection annule la demande précédente ; aucune réponse ancienne ne remplace celle de la ligne active.

Les explications SHAP réussies sont conservées en mémoire dans le navigateur, par transaction. Sélectionner à nouveau un paiement déjà expliqué réutilise son résultat sans nouvel appel API. Le cache est vidé après un nouveau calcul réussi des probabilités ou un rechargement de la page. Les erreurs et les requêtes annulées ne sont pas mises en cache.

« Agrandir le graphique » ouvre le waterfall dans une fenêtre dédiée, refermable par le bouton de fermeture ou Échap, sans nouvel appel SHAP. Après le calcul des probabilités, « Exporter en CSV » télécharge les dix paiements dans l’ordre du tableau, avec leurs identifiants, les colonnes affichées et leurs probabilités en %. Les valeurs exportées sont brutes (parts historiques entre 0 et 1, dates en UTC), avec les noms techniques des colonnes. Le fichier utilise UTF-8 avec BOM, le séparateur `;` et la virgule décimale pour les tableurs français ; les valeurs manquantes restent vides.

Le waterfall Plotly utilise les dix principales contributions signées et leur reste sommé dans « Autres variables ». L’échelle renvoyée par l’API est `percentage_points` : référence + contributions = probabilité finale en %. Le frontend vérifie cette égalité et la cohérence avec le score du tableau. Les contributions concernent les variables **après preprocessing**, avec les indicatrices des catégories séparées.

## Vérification

```bash
npm test
npm run build
npx playwright install --with-deps chromium
npm run test:e2e
```

Les exemples JSON de `tests/fixtures` servent uniquement aux tests navigateur et ne sont jamais importés par l’application. La lecture et l’actualisation des exports se vérifient depuis la racine avec `uv run python -m unittest discover -s 10_app_react/tests -p 'test_*.py'`. Pour tester les prédictions réelles, il faut aussi disposer du pipeline et de l’explainer exportés.
