import { test, expect, type Page } from "@playwright/test";
import dataset from "./fixtures/transactions.json" with { type: "json" };
import metrics from "./fixtures/metrics.json" with { type: "json" };

test.beforeEach(async ({ page }, info) => {
  if (info.title.startsWith("FastAPI réel")) return;
  await page.route("**/api/demo-data", route => route.fulfill({ json: dataset }));
  await page.route("**/api/business-metrics", route => route.fulfill({ json: {
    ...metrics,
    periods: metrics.periods.map(item => item.key === "average" ? { ...item, label: "Moyenne de tous les mois" } : item),
  } }));
});

const rows = dataset.observations;
const scoreFor = (index: number) => 2 + index * 8;
const requestLog: unknown[] = [];
const explanation = (prediction: number) => ({
  model_version: "test",
  output_scale: "percentage_points",
  explanations: [
    {
      id_observation: 1,
      base_value: 1,
      prediction,
      features: [
        { name: "nb_carte_1h_passe", value: 3, shap_value: prediction - 2 },
        ...Array.from({ length: 9 }, (_, i) => ({
          name: `part_client_heure_${String(i).padStart(2, "0")}_passe`,
          value: 0.1,
          shap_value: i % 2 ? -0.1 : 0.1,
        })),
        { name: "Autres variables", value: null, shap_value: 0.9 },
      ],
    },
  ],
});

async function mockPredictions(page: Page) {
  await page.route("**/api/batch-predict", async (route) => {
    requestLog.push(route.request().postDataJSON());
    await route.fulfill({
      json: {
        model_version: "test",
        model_nb_probas: 10,
        // Renvoi désordonné volontaire : les identifiants doivent être respectés.
        model_probas: rows
          .map((_, i) => ({ id_observation: i + 1, model_proba: scoreFor(i) }))
          .reverse(),
      },
    });
  });
}
async function openDemo(page: Page) {
  await page.goto("/");
  await page.getByRole("tab", { name: "Démonstration" }).click();
}

test("parcours, colonnes, tri, waterfall et conservation entre onglets", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  let shapCalls = 0;
  await mockPredictions(page);
  await page.route("**/api/shap-explanation", async (route) => {
    shapCalls++;
    const body = route.request().postDataJSON();
    expect(body.data).toEqual([rows[9].features]);
    await route.fulfill({ json: explanation(scoreFor(9)) });
  });
  await openDemo(page);
  await expect(page.locator("tbody tr")).toHaveCount(10);
  await expect(
    page.getByRole("button", { name: "Probabilité de fraude", exact: true }),
  ).toHaveCount(0);
  await page
    .getByRole("button", {
      name: "Expliquer la transaction TX-001",
      exact: true,
    })
    .click();
  await expect(
    page.getByText("Veuillez d’abord calculer les probabilités."),
  ).toBeVisible();
  expect(shapCalls).toBe(0);
  await page.getByRole("button", { name: /Colonnes/ }).click();
  await page.getByRole("checkbox", { name: "Canal", exact: true }).uncheck();
  await expect(
    page.getByRole("columnheader", { name: "Canal", exact: true }),
  ).toHaveCount(0);
  await page
    .getByRole("button", { name: "Fermer le menu des colonnes" })
    .click();
  await page.getByRole("button", { name: "Calculer les probabilités" }).click();
  await expect(
    page.getByText("Probabilités calculées", { exact: false }),
  ).toBeVisible();
  expect(requestLog.at(-1)).toEqual({ data: rows.map((row) => row.features) });
  await page
    .getByRole("button", { name: "Probabilité de fraude", exact: true })
    .click();
  await expect(page.locator("tbody tr").first()).toHaveAttribute(
    "data-transaction-id",
    rows[9].id,
  );
  await page
    .getByRole("button", {
      name: "Expliquer la transaction TX-010",
      exact: true,
    })
    .click();
  await expect(
    page.getByRole("heading", { name: "Explication de TX-010", exact: true }),
  ).toBeVisible();
  await expect(page.locator(".waterfall .main-svg").first()).toBeVisible({
    timeout: 25_000,
  });
  const plot = await page.locator(".waterfall").evaluate((element: any) => ({
    measures: element.data[0].measure,
    values: element.data[0].x,
    names: element.data[0].y,
  }));
  expect(plot.measures).toEqual([
    "absolute",
    ...Array(11).fill("relative"),
    "total",
  ]);
  expect(plot.names.at(-2)).toBe("Autres variables");
  expect(
    plot.values.slice(0, -1).reduce((a: number, b: number) => a + b, 0),
  ).toBeCloseTo(scoreFor(9), 5);
  await page.getByRole("tab", { name: "Présentation du projet" }).click();
  await page.getByRole("tab", { name: "Démonstration" }).click();
  await expect(
    page.getByRole("heading", { name: "Explication de TX-010", exact: true }),
  ).toBeVisible();
  await expect(page.locator("tbody tr").first()).toHaveAttribute(
    "data-transaction-id",
    rows[9].id,
  );
  await page.screenshot({
    path: "test-results/demonstration.png",
    fullPage: true,
  });
  expect(errors).toEqual([]);
});

test("seule la dernière transaction sélectionnée conserve son explication", async ({
  page,
}) => {
  await mockPredictions(page);
  await page.route("**/api/shap-explanation", async (route) => {
    const row = route.request().postDataJSON().data[0];
    const index = rows.findIndex(
      (r) => r.features.montant_eur === row.montant_eur,
    );
    await new Promise((resolve) =>
      setTimeout(resolve, index === 0 ? 1200 : 40),
    );
    await route.fulfill({ json: explanation(scoreFor(index)) }).catch(() => {});
  });
  await openDemo(page);
  await page.getByRole("button", { name: "Calculer les probabilités" }).click();
  await expect(
    page.getByText("Probabilités calculées", { exact: false }),
  ).toBeVisible();
  await page
    .getByRole("button", {
      name: "Expliquer la transaction TX-001",
      exact: true,
    })
    .click();
  await page
    .getByRole("button", {
      name: "Expliquer la transaction TX-002",
      exact: true,
    })
    .click();
  await expect(page.locator(".waterfall .main-svg").first()).toBeVisible();
  await page.waitForTimeout(1400);
  await expect(
    page.getByRole("heading", { name: "Explication de TX-002", exact: true }),
  ).toBeVisible();
  await expect(page.locator(".selected-score strong")).toHaveText("10 %");
});

test("erreur de calcul : message, réessai et conservation des résultats valides", async ({
  page,
}) => {
  let calls = 0;
  await page.route("**/api/batch-predict", async (route) => {
    calls++;
    await route.fulfill(
      calls === 1
        ? {
            json: {
              model_nb_probas: 10,
              model_probas: rows.map((_, i) => ({
                id_observation: i + 1,
                model_proba: scoreFor(i),
              })),
            },
          }
        : { status: 502, json: { detail: "API indisponible pour ce test." } },
    );
  });
  await openDemo(page);
  await page.getByRole("button", { name: "Calculer les probabilités" }).click();
  await expect(
    page.getByText("Probabilités calculées", { exact: false }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Calculer les probabilités" }).click();
  await expect(page.getByText("API indisponible pour ce test.")).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Réessayer", exact: true }),
  ).toBeVisible();
  await expect(page.locator(".probability-value")).toHaveCount(10);
});

test("petit écran : tableau défilable et colonne Actions fixe", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openDemo(page);
  const action = page.getByRole("button", {
    name: "Expliquer la transaction TX-001",
    exact: true,
  });
  await expect(action).toBeInViewport();
  const box = await action.boundingBox();
  await page.locator(".table-scroll").evaluate((element) => {
    element.scrollLeft = element.scrollWidth;
  });
  await expect(action).toBeInViewport();
  const movedBox = await action.boundingBox();
  expect(Math.abs(box!.x - movedBox!.x)).toBeLessThan(2);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({ path: "test-results/mobile.png", fullPage: true });
});

test("FastAPI réel : chaque transaction est explicable seule, dont TX-007", async ({
  page,
}) => {
  test.skip(
    !process.env.FRAUD_LIVE_API,
    "Nécessite FastAPI avec les artefacts du projet.",
  );
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await openDemo(page);
  const predictionResponse = page.waitForResponse((response) =>
    response.url().endsWith("/api/batch-predict"),
  );
  await page.getByRole("button", { name: "Calculer les probabilités" }).click();
  const response = await predictionResponse;
  expect(response.ok()).toBeTruthy();
  const predictions = await response.json();
  expect(predictions.model_probas).toHaveLength(10);
  // Un lot peut masquer un problème de type sur une ligne isolée avec des null.
  for (const [index, row] of rows.entries()) {
    const single = await page.request.post("/api/shap-explanation", {
      data: { data: [row.features] },
    });
    expect(single.ok(), `SHAP TX-${index + 1}`).toBeTruthy();
    const item = (await single.json()).explanations[0];
    const predicted = predictions.model_probas.find(
      (value: any) => value.id_observation === index + 1,
    ).model_proba;
    expect(item.prediction).toBeCloseTo(predicted, 5);
    expect(
      item.base_value +
        item.features.reduce(
          (sum: number, feature: any) => sum + feature.shap_value,
          0,
        ),
    ).toBeCloseTo(predicted, 3);
  }
  const id = "007";
  const expectedProbability = predictions.model_probas.find(
    (value: any) => value.id_observation === 7,
  ).model_proba;
  const shapResponse = page.waitForResponse((response) =>
    response.url().endsWith("/api/shap-explanation"),
  );
  await page
    .getByRole("button", {
      name: `Expliquer la transaction TX-${id}`,
      exact: true,
    })
    .click();
  expect((await shapResponse).ok()).toBeTruthy();
  await expect(page.locator(".waterfall .main-svg").first()).toBeVisible({
    timeout: 25_000,
  });
  const plot = await page.locator(".waterfall").evaluate((element: any) => ({
    values: element.data[0].x,
    measures: element.data[0].measure,
  }));
  expect(plot.measures).toHaveLength(13);
  expect(
    plot.values.slice(0, -1).reduce((a: number, b: number) => a + b, 0),
  ).toBeCloseTo(expectedProbability, 3);
  await page.screenshot({
    path: "test-results/fastapi-reel.png",
    fullPage: true,
  });
  expect(errors).toEqual([]);
});

test("lien GitHub, six mois hors entraînement et moyenne mensuelle", async ({
  page,
}) => {
  await page.goto("/");
  const github = page.getByRole("link", {
    name: "Voir le projet sur GitHub (nouvel onglet)",
  });
  await expect(github).toHaveAttribute(
    "href",
    "https://github.com/daren-mabanza/V2_fraud-card-transaction-detection",
  );
  await expect(github).toHaveAttribute("target", "_blank");
  const period = page.getByRole("combobox", { name: "Période" });
  await expect(period.locator("option")).toHaveText([
    "Janvier 2025",
    "Février 2025",
    "Mars 2025",
    "Avril 2025",
    "Mai 2025",
    "Juin 2025",
    "Moyenne de tous les mois",
  ]);
  const displayed = page
    .locator(".metric-card")
    .first()
    .locator(".metric-score");
  for (const selected of metrics.periods) {
    await period.selectOption(selected.key);
    const expected = metrics.results.find(
      (row) =>
        row.period === selected.key &&
        row.capacity === 10 &&
        row.method === "ML",
    )!;
    const formatted = new Intl.NumberFormat("fr-FR", {
      maximumFractionDigits: 2,
    }).format(expected.fraud_on_time);
    await expect(displayed).toContainText(`${formatted} %`);
  }
  await expect(
    page.getByText(metrics.average_method, { exact: false }),
  ).toBeVisible();
});

test("exports manquants : erreur visible puis chargement après réessai", async ({ page }) => {
  let calls = 0;
  await page.route("**/api/demo-data", async route => {
    calls++;
    await route.fulfill(calls === 1
      ? { status: 503, json: { detail: "Les transactions ne sont pas encore exportées." } }
      : { json: dataset });
  });
  await openDemo(page);
  await expect(page.getByText("Les transactions ne sont pas encore exportées.")).toBeVisible();
  await expect(page.locator("tbody tr")).toHaveCount(0);
  await page.getByRole("button", { name: "Réessayer", exact: true }).click();
  await expect(page.locator("tbody tr")).toHaveCount(10);
});
