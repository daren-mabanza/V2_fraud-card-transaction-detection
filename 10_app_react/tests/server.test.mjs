import { test } from "node:test";
import assert from "node:assert/strict";
import { once } from "node:events";
import { createServer } from "node:http";
import { createApp } from "../server.mjs";

async function listen(app, t) {
  const server = createServer(app).listen(0, "127.0.0.1");
  await once(server, "listening");
  t.after(() => {
    server.closeAllConnections();
    server.close();
  });
  return `http://127.0.0.1:${server.address().port}`;
}

test("sans token : seules les routes prévues sont relayées, données inchangées", async (t) => {
  const calls = [];
  const upstream = await listen((req, res) => {
    let body = "";
    req.on("data", (chunk) => {
      body += chunk;
    });
    req.on("end", () => {
      calls.push({
        path: req.url,
        authorization: req.headers.authorization,
        body,
      });
      res.setHeader("Content-Type", "application/json");
      res.end(
        JSON.stringify({
          model_nb_probas: 1,
          model_probas: [{ id_observation: 1, model_proba: 21 }],
        }),
      );
    });
  }, t);
  const url = await listen(
    createApp({ apiUrl: upstream, authMode: "none" }),
    t,
  );
  const body = { data: [{ montant_eur: 5, pays_ip: null }] };
  const response = await fetch(`${url}/api/batch-predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: "Bearer navigateur-ignore",
    },
    body: JSON.stringify(body),
  });
  assert.equal(response.status, 200);
  assert.equal((await response.json()).model_probas[0].model_proba, 21);
  assert.deepEqual(JSON.parse(calls[0].body), body);
  assert.equal(calls[0].path, "/batch-predict");
  assert.equal(calls[0].authorization, undefined);
  assert.equal((await fetch(`${url}/api/autre`)).status, 404);
  assert.equal(
    (
      await fetch(`${url}/api/batch-predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{",
      })
    ).status,
    400,
  );
  assert.equal(calls.length, 1);
});

test("GCP : ID token obtenu côté serveur, audience conservée et renouvellement délégué", async (t) => {
  const originalFetch = globalThis.fetch;
  const headersReceived = [];
  let clients = 0,
    tokenRequests = 0;
  const auth = {
    async getIdTokenClient(audience) {
      clients++;
      assert.equal(audience, "https://service.a.run.app");
      return {
        async getRequestHeaders() {
          tokenRequests++;
          return new Headers({
            authorization: `Bearer token-${tokenRequests}`,
          });
        },
      };
    },
  };
  globalThis.fetch = async (url, options) => {
    if (String(url).startsWith("https://service.a.run.app")) {
      headersReceived.push(options.headers.Authorization);
      return Response.json({ status: "ok" });
    }
    return originalFetch(url, options);
  };
  t.after(() => {
    globalThis.fetch = originalFetch;
  });
  const url = await listen(
    createApp({
      apiUrl: "https://service.a.run.app",
      authMode: "gcp",
      audience: "https://service.a.run.app",
      auth,
    }),
    t,
  );
  for (let i = 0; i < 2; i++) {
    const response = await fetch(`${url}/api/health`);
    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), { status: "ok" });
  }
  assert.deepEqual(headersReceived, ["Bearer token-1", "Bearer token-2"]);
  assert.equal(clients, 1);
});

test("GCP indisponible : erreur compréhensible, pas de secret renvoyé", async (t) => {
  const auth = {
    async getIdTokenClient() {
      throw new Error("secret interne");
    },
  };
  const url = await listen(
    createApp({ apiUrl: "https://service.a.run.app", authMode: "gcp", auth }),
    t,
  );
  const response = await fetch(`${url}/api/health`);
  assert.equal(response.status, 503);
  const body = await response.text();
  assert.ok(body.includes("Authentification GCP indisponible"));
  assert.ok(!body.includes("secret interne"));
});

test("configuration invalide refusée au démarrage", () => {
  assert.throws(
    () => createApp({ apiUrl: "http://local", authMode: "gcp" }),
    /HTTPS/,
  );
  assert.throws(() => createApp({ apiUrl: "file:///etc/passwd" }), /HTTP/);
});

test("erreur interne FastAPI non JSON : réponse JSON explicite conservant le statut", async (t) => {
  const upstream = await listen((_req, res) => {
    res.writeHead(500, { "Content-Type": "text/plain" });
    res.end("Internal Server Error");
  }, t);
  const url = await listen(
    createApp({ apiUrl: upstream, authMode: "none" }),
    t,
  );
  const response = await fetch(`${url}/api/shap-explanation`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ data: [{}] }),
  });
  assert.equal(response.status, 500);
  assert.match((await response.json()).detail, /erreur interne/);
});
