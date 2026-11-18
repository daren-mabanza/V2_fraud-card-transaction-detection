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

test("export CSV : colonnes affichées, tri et probabilités conservés", async ({ page }) => {
  await mockPredictions(page);
  const customData = structuredClone(dataset);
  customData.observations[9].features.categorie_marchand = 'Commerce; "test"\nAutre';
  await page.route("**/api/demo-data", route => route.fulfill({ json: customData }));
  await openDemo(page);
  const exportButton = page.getByRole("button", { name: "Exporter en CSV" });
  await expect(exportButton).toBeDisabled();
  const toolbar = page.locator(".table-controls");
  await expect(toolbar.getByRole("button")).toHaveText([
    /Colonnes/, "", "Calculer les probabilités", "Exporter en CSV",
  ]);
  await page.getByRole("button", { name: /Colonnes/ }).click();
  await page.getByRole("checkbox", { name: "Canal", exact: true }).uncheck();
  await page.getByPlaceholder("Rechercher une variable…").fill("pays_ip");
  await page.getByRole("checkbox").check();
  await page.getByRole("button", { name: "Fermer le menu des colonnes" }).click();
  await page.getByRole("button", { name: "Calculer les probabilités" }).click();
  await expect(exportButton).toBeEnabled();
  await page.getByRole("button", { name: "Probabilité de fraude", exact: true }).click();
  const downloadPromise = page.waitForEvent("download");
  await exportButton.click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe("fraudlens-transactions-2025-04-21.csv");
  const stream = await download.createReadStream();
  const chunks: Buffer[] = [];
  for await (const chunk of stream!) chunks.push(Buffer.from(chunk));
  const csv = Buffer.concat(chunks).toString("utf8");
  expect(csv.startsWith("\uFEFF")).toBe(true);
  const lines = csv.slice(1).trimEnd().split("\r\n");
  expect(lines).toHaveLength(11);
  expect(lines[0]).toBe('"transaction";"id_paiement";"montant_eur";"mode_saisie";"categorie_marchand";"date_utc";"pays_ip";"probabilite_fraude_pct"');
  expect(lines[1]).toContain(`"TX-010";"${rows[9].id}"`);
  expect(lines[1]).toContain('"Commerce; ""test""\nAutre"');
  expect(lines[1]).toContain(`"${String(rows[9].features.montant_eur).replace(".", ",")}"`);
  expect(lines[1]).toMatch(/;"74"$/);
  expect(lines[10]).toContain(`"TX-001";"${rows[0].id}"`);
  expect(lines[10]).toMatch(/;"";"2"$/);
});

test("graphique SHAP agrandi : même résultat, fermeture et retour au bouton", async ({ page }) => {
  let calls = 0;
  await mockPredictions(page);
  await page.route("**/api/shap-explanation", route => {
    calls++;
    return route.fulfill({ json: explanation(scoreFor(0)) });
  });
  await openDemo(page);
  await page.getByRole("button", { name: "Calculer les probabilités" }).click();
  await expect(page.locator(".probability-value")).toHaveCount(10);
  await page.getByRole("button", { name: "Expliquer la transaction TX-001", exact: true }).click();
  await expect(page.locator(".waterfall .main-svg").first()).toBeVisible();
  const enlarge = page.getByRole("button", { name: "Agrandir le graphique" });
  await enlarge.click();
  const dialog = page.getByRole("dialog", { name: "Graphique SHAP agrandi" });
  await expect(dialog).toBeVisible();
  await expect(dialog.locator(".waterfall .main-svg").first()).toBeVisible();
  const plot = await dialog.locator(".waterfall").evaluate((element: any) => element.data[0].x);
  expect(plot.slice(0, -1).reduce((sum: number, value: number) => sum + value, 0)).toBeCloseTo(scoreFor(0), 5);
  await page.keyboard.press("Escape");
  await expect(dialog).not.toBeVisible();
  await expect(enlarge).toBeFocused();
  await page.setViewportSize({ width: 390, height: 844 });
  await enlarge.click();
  await expect(dialog).toBeVisible();
  await expect(dialog.locator(".waterfall .main-svg").first()).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  expect(await dialog.locator(".waterfall-wrap").evaluate(element => element.scrollWidth <= element.clientWidth)).toBe(true);
  await page.screenshot({ path: "test-results/shap-enlarged-mobile.png", fullPage: true });
  await dialog.getByRole("button", { name: "Fermer le graphique agrandi" }).click();
  await expect(dialog).not.toBeVisible();
  await expect(enlarge).toBeFocused();
  expect(await page.evaluate(() => document.body.style.overflow)).toBe("");
  expect(calls).toBe(1);
});

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

test("explications en cache : aucun nouvel appel avant un recalcul des probabilités", async ({ page }) => {
  const calls: number[] = [];
  await mockPredictions(page);
  await page.route("**/api/shap-explanation", async route => {
    const features = route.request().postDataJSON().data[0];
    const index = rows.findIndex(row => row.features.montant_eur === features.montant_eur);
    calls.push(index);
    await route.fulfill({ json: explanation(scoreFor(index)) });
  });
  await openDemo(page);
  await page.getByRole("button", { name: "Calculer les probabilités" }).click();
  await expect(page.locator(".probability-value")).toHaveCount(10);

  const explainFirst = page.getByRole("button", { name: "Expliquer la transaction TX-001", exact: true });
  const explainSecond = page.getByRole("button", { name: "Expliquer la transaction TX-002", exact: true });
  await explainFirst.click();
  await expect(page.locator(".waterfall .main-svg").first()).toBeVisible();
  await explainFirst.click();
  await expect(page.getByRole("region", { name: "Explication SHAP" })).toHaveAttribute("aria-busy", "false");
  expect(calls).toEqual([0]);

  await explainSecond.click();
  await expect(page.locator(".waterfall .main-svg").first()).toBeVisible();
  await page.getByRole("tab", { name: "Présentation du projet" }).click();
  await page.getByRole("tab", { name: "Démonstration" }).click();
  await explainFirst.click();
  await expect(page.locator(".selected-score strong")).toHaveText("2 %");
  await expect(page.locator(".waterfall .main-svg").first()).toBeVisible();
  expect(calls).toEqual([0, 1]);

  await page.getByRole("button", { name: "Calculer les probabilités" }).click();
  await expect(page.getByRole("heading", { name: "Un score, des explications." })).toBeVisible();
  await explainFirst.click();
  await expect(page.locator(".waterfall .main-svg").first()).toBeVisible();
  expect(calls).toEqual([0, 1, 0]);
});

test("cache SHAP : les erreurs et appels annulés restent réessayables", async ({ page }) => {
  const calls = [0, 0];
  let releasePending!: () => void;
  const pending = new Promise<void>(resolve => { releasePending = resolve; });
  let finishPending!: () => void;
  const finished = new Promise<void>(resolve => { finishPending = resolve; });
  await mockPredictions(page);
  await page.route("**/api/shap-explanation", async route => {
    const features = route.request().postDataJSON().data[0];
    const index = rows.findIndex(row => row.features.montant_eur === features.montant_eur);
    calls[index]++;
    if (index === 0 && calls[index] === 1) {
      await route.fulfill({ status: 503, json: { detail: "Explication temporairement indisponible." } });
      return;
    }
    if (index === 1 && calls[index] === 1) {
      await pending;
      await route.fulfill({ json: explanation(scoreFor(index)) }).catch(() => {});
      finishPending();
      return;
    }
    await route.fulfill({ json: explanation(scoreFor(index)) });
  });
  await openDemo(page);
  await page.getByRole("button", { name: "Calculer les probabilités" }).click();
  await expect(page.locator(".probability-value")).toHaveCount(10);
  const explainFirst = page.getByRole("button", { name: "Expliquer la transaction TX-001", exact: true });
  const explainSecond = page.getByRole("button", { name: "Expliquer la transaction TX-002", exact: true });
  await explainFirst.click();
  await expect(page.getByText("Explication temporairement indisponible.")).toBeVisible();
  await page.getByRole("button", { name: "Réessayer", exact: true }).click();
  await expect(page.locator(".waterfall .main-svg").first()).toBeVisible();

  await explainSecond.click();
  await expect(page.getByText("Calcul des contributions SHAP…")).toBeVisible();
  await expect.poll(() => calls[1]).toBe(1);
  await explainFirst.click();
  await expect(page.locator(".waterfall .main-svg").first()).toBeVisible();
  releasePending();
  await finished;
  await expect(page.locator(".selected-score strong")).toHaveText("2 %");
  await expect(page.getByRole("region", { name: "Explication SHAP" })).toHaveAttribute("aria-busy", "false");
  expect(calls).toEqual([2, 1]);

  await explainSecond.click();
  await expect(page.locator(".waterfall .main-svg").first()).toBeVisible();
  await expect(page.locator(".selected-score strong")).toHaveText("10 %");
  expect(calls).toEqual([2, 2]);
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

test("aide sur les colonnes : affichage, clavier et sélection de variables", async ({ page }) => {
  await openDemo(page);
  const help = page.getByRole("button", { name: "Aide sur les colonnes du tableau" });
  await help.focus();
  await page.keyboard.press("Enter");
  await expect(help).toHaveAttribute("aria-expanded", "true");
  await expect(page.locator("#column-help")).toContainText("afficher d’autres variables");
  await page.keyboard.press("Escape");
  await expect(help).toHaveAttribute("aria-expanded", "false");
  await help.click();
  await page.getByRole("button", { name: /Colonnes/ }).click();
  await expect(page.locator("#column-help")).toHaveCount(0);
  await page.getByPlaceholder("Rechercher une variable…").fill("pays_ip");
  await page.getByRole("checkbox").check();
  await page.getByRole("button", { name: "Fermer le menu des colonnes" }).click();
  await expect(page.locator('thead th[title="pays_ip"]')).toBeVisible();
});

for (const width of [390, 1440]) {
test(`tableau à ${width}px : colonnes Actions et Probabilité fixes`, async ({
  page,
}) => {
  await page.setViewportSize({ width, height: 844 });
  await mockPredictions(page);
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
  await expect(page.locator("thead .probability-cell")).toHaveCount(0);
  await page.getByRole("button", { name: "Calculer les probabilités" }).click();
  await expect(page.locator("tbody .probability-cell")).toHaveCount(10);
  const pinned = page.locator("thead .probability-cell, thead .action-cell, tbody tr:first-child .probability-cell, tbody tr:first-child .action-cell");
  const positions = await pinned.evaluateAll(cells => cells.map(cell => cell.getBoundingClientRect().x));
  for (const fraction of [0, 0.5, 1]) {
    await page.locator(".table-scroll").evaluate((element, fraction) => {
      element.scrollLeft = fraction * (element.scrollWidth - element.clientWidth);
    }, fraction);
    for (const [index, cell] of (await pinned.all()).entries()) {
      await expect(cell).toBeInViewport();
      const position = await cell.boundingBox();
      expect(Math.abs(position!.x - positions[index])).toBeLessThan(2);
    }
    const probability = await page.locator("tbody .probability-cell").first().boundingBox();
    const actionCell = await page.locator("tbody .action-cell").first().boundingBox();
    expect(probability!.x + probability!.width).toBeLessThanOrEqual(actionCell!.x + 1);
  }
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({ path: `test-results/pinned-columns-${width}.png`, fullPage: true });
});
}

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
  const linkedin = page.getByRole("link", { name: "Voir le profil LinkedIn de Daren Mabanza (nouvel onglet)" });
  await expect(linkedin).toHaveAttribute("href", "https://www.linkedin.com/in/daren-mabanza-53b9321b2/");
  await expect(linkedin).toHaveAttribute("target", "_blank");
  await expect(linkedin.locator("svg")).toBeVisible();
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
