import express from "express";
import { GoogleAuth } from "google-auth-library";
import { fileURLToPath, pathToFileURL } from "node:url";
import path from "node:path";

const appDir = path.dirname(fileURLToPath(import.meta.url));

// Le token et les identifiants GCP restent dans ce serveur, commun aux deux variantes.
/**
 * Crée le relais Express vers FastAPI et valide la configuration.
 * Lit FRAUD_API_URL, FRAUD_AUTH_MODE et FRAUD_API_AUDIENCE par défaut ;
 * les options permettent notamment d'injecter un client Google pour les tests.
 * Le mode gcp exige HTTPS. Les identifiants et tokens restent côté serveur.
 * @returns Application Express avec les seules routes API autorisées.
 */
export function createApp({
  apiUrl = process.env.FRAUD_API_URL || "http://127.0.0.1:8080",
  authMode = process.env.FRAUD_AUTH_MODE || "none",
  audience = process.env.FRAUD_API_AUDIENCE || apiUrl,
  auth = new GoogleAuth(),
} = {}) {
  const upstream = new URL(apiUrl);
  if (
    !["http:", "https:"].includes(upstream.protocol) ||
    upstream.username ||
    upstream.password ||
    upstream.search ||
    upstream.hash
  ) {
    throw new Error(
      "FRAUD_API_URL doit être une URL HTTP(S) sans identifiants ni paramètres.",
    );
  }
  if (!["none", "gcp"].includes(authMode))
    throw new Error("FRAUD_AUTH_MODE doit valoir none ou gcp.");
  if (authMode === "gcp" && upstream.protocol !== "https:")
    throw new Error("La variante GCP nécessite une API HTTPS.");
  const app = express();
  app.disable("x-powered-by");
  app.use(express.json({ limit: "128kb" }));
  let tokenClient;

  /**
   * Transmet une requête à FastAPI, avec un ID token serveur en mode gcp.
   * Ne transmet pas l'en-tête Authorization du navigateur. L'appel à FastAPI
   * expire après 60 secondes ; ses erreurs sont renvoyées en JSON sans cache.
   */
  async function relay(req, res) {
    const headers = {
      Accept: "application/json",
      "Content-Type": "application/json",
    };
    if (authMode === "gcp") {
      try {
        tokenClient ??= auth.getIdTokenClient(audience);
        const client = await tokenClient;
        const authenticatedHeaders = await client.getRequestHeaders();
        headers.Authorization = authenticatedHeaders.get("authorization");
      } catch {
        tokenClient = undefined;
        return res.status(503).json({
          detail:
            "Authentification GCP indisponible. Vérifiez les identifiants et les autorisations du service frontend.",
        });
      }
    }
    try {
      const url = `${apiUrl.replace(/\/$/, "")}${req.path.replace(/^\/api/, "")}`;
      const response = await fetch(url, {
        method: req.method,
        headers,
        body: req.method === "POST" ? JSON.stringify(req.body) : undefined,
        signal: AbortSignal.timeout(60_000),
        redirect: "error",
      });
      const payload = await response.json().catch(() => null);
      if (response.status === 401 || response.status === 403) {
        return res.status(response.status).json({
          detail:
            "Accès à l’API refusé. Vérifiez son mode d’authentification et les droits Cloud Run Invoker.",
        });
      }
      res.set("Cache-Control", "no-store");
      if (payload === null) {
        return res.status(response.ok ? 502 : response.status).json({
          detail: response.ok
            ? "Le service de prédiction a renvoyé une réponse au format inattendu. Veuillez réessayer."
            : "Le service de prédiction a rencontré une erreur interne. Veuillez réessayer.",
        });
      }
      return res.status(response.status).json(payload);
    } catch (error) {
      const timeout = error.name === "TimeoutError";
      return res.status(timeout ? 504 : 502).json({
        detail: timeout
          ? "Le calcul prend trop de temps. Veuillez réessayer."
          : "L’API est momentanément inaccessible. Veuillez réessayer.",
      });
    }
  }

  app.get("/api/health", relay);
  app.get("/api/demo-data", relay);
  app.get("/api/business-metrics", relay);
  app.post("/api/batch-predict", relay);
  app.post("/api/shap-explanation", relay);
  app.use("/api", (_req, res) =>
    res.status(404).json({ detail: "Route API inconnue." }),
  );
  app.use((error, _req, res, next) => {
    if (error instanceof SyntaxError || error.type === "entity.too.large") {
      return res.status(400).json({
        detail:
          "Le contenu de la requête JSON est invalide ou trop volumineux.",
      });
    }
    next(error);
  });
  return app;
}

/**
 * Démarre le relais et sert React via Vite en développement ou dist en production.
 * Écoute sur PORT si défini, sinon 5173 en développement ou 8080 en production.
 * @returns Serveur HTTP en écoute sur toutes les interfaces.
 */
export async function startServer() {
  const app = createApp();
  const development = process.env.NODE_ENV !== "production";
  if (development) {
    const { createServer } = await import("vite");
    const vite = await createServer({
      root: appDir,
      server: { middlewareMode: true },
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.join(appDir, "dist")));
    app.get("/", (_req, res) =>
      res.sendFile(path.join(appDir, "dist/index.html")),
    );
  }
  const port = Number(process.env.PORT || (development ? 5173 : 8080));
  return app.listen(port, "0.0.0.0", () => {
    console.log(
      `FraudLens : http://localhost:${port} — authentification ${process.env.FRAUD_AUTH_MODE || "none"}`,
    );
  });
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href)
  await startServer();
