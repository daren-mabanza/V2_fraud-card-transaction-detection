import type {
  Explanation,
  Observation,
  PredictionResponse,
  ShapResponse,
} from "./types";

/**
 * Charge un export JSON via le relais Node, sans cache HTTP.
 * L'appel peut être annulé et expire après 65 secondes ; les erreurs HTTP
 * ou les réponses vides sont signalées à l'appelant.
 */
export async function loadExport<T>(route: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`/api/${route}`, {
    cache: "no-store",
    signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(65_000)]) : AbortSignal.timeout(65_000),
  });
  const result = await response.json().catch(() => null);
  if (!response.ok || !result) {
    throw new Error(typeof result?.detail === "string" ? result.detail : "Les données du projet sont indisponibles. Vérifiez les exports et la connexion à l’API.");
  }
  return result as T;
}

/**
 * Envoie uniquement les features des observations dans le champ data.
 * Applique le délai de 65 secondes et traduit les erreurs réseau ou API
 * en messages affichables ; une annulation demandée par l'appelant est propagée.
 */
async function post<T>(
  route: string,
  observations: Observation[],
  signal?: AbortSignal,
): Promise<T> {
  try {
    const response = await fetch(`/api/${route}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data: observations.map((row) => row.features) }),
      signal: signal
        ? AbortSignal.any([signal, AbortSignal.timeout(65_000)])
        : AbortSignal.timeout(65_000),
    });
    const result = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(
        typeof result?.detail === "string"
          ? result.detail
          : response.status === 422
            ? "Les observations ne correspondent pas au format attendu par le modèle."
            : "Le calcul n’a pas abouti. Veuillez réessayer.",
      );
    }
    if (!result)
      throw new Error("La réponse du service est vide. Veuillez réessayer.");
    return result as T;
  } catch (error) {
    if (signal?.aborted) throw error;
    if (error instanceof TypeError)
      throw new Error(
        "Connexion au service impossible. Vérifiez que l’API est démarrée.",
      );
    if (error instanceof DOMException && error.name === "TimeoutError")
      throw new Error("Le calcul prend trop de temps. Veuillez réessayer.");
    throw error;
  }
}

/**
 * Calcule les probabilités du lot et les associe aux identifiants de paiement.
 * Vérifie le nombre de résultats, les identifiants positionnels et les valeurs
 * finies entre 0 et 100, même si l'API renvoie les résultats dans un autre ordre.
 * @returns Probabilités en pourcentage, indexées par identifiant de paiement.
 */
export async function predictBatch(
  observations: Observation[],
): Promise<Record<string, number>> {
  const result = await post<PredictionResponse>("batch-predict", observations);
  if (
    result.model_nb_probas !== observations.length ||
    result.model_probas?.length !== observations.length
  ) {
    throw new Error(
      "Réponse incomplète : le nombre de probabilités ne correspond pas au lot envoyé.",
    );
  }
  const predictions: Record<string, number> = {};
  const seen = new Set<number>();
  for (const item of result.model_probas) {
    const row = observations[item.id_observation - 1];
    if (
      !row ||
      seen.has(item.id_observation) ||
      !Number.isFinite(item.model_proba) ||
      item.model_proba < 0 ||
      item.model_proba > 100
    ) {
      throw new Error(
        "Les probabilités reçues ne peuvent pas être associées aux transactions.",
      );
    }
    seen.add(item.id_observation);
    predictions[row.id] = item.model_proba;
  }
  return predictions;
}

/**
 * Charge et vérifie l'explication SHAP d'un paiement.
 * Contrôle les dix contributions principales, « Autres variables » et l'égalité
 * référence + contributions = probabilité, avec une tolérance de 0,001 point.
 * @param expectedProbability Score du tableau en pourcentage, entre 0 et 100.
 * @param signal Permet d'annuler une sélection devenue obsolète.
 */
export async function explainObservation(
  row: Observation,
  expectedProbability: number,
  signal: AbortSignal,
): Promise<Explanation> {
  const result = await post<ShapResponse>("shap-explanation", [row], signal);
  const explanation = result.explanations?.[0];
  if (
    result.output_scale !== "percentage_points" ||
    result.explanations?.length !== 1 ||
    !explanation ||
    explanation.features?.length !== 11 ||
    explanation.features[10].name !== "Autres variables" ||
    !Number.isFinite(explanation.base_value) ||
    !Number.isFinite(explanation.prediction) ||
    explanation.features.some((f) => !Number.isFinite(f.shap_value))
  ) {
    throw new Error("Le format de l’explication SHAP est inattendu.");
  }
  const reconstructed =
    explanation.base_value +
    explanation.features.reduce((sum, f) => sum + f.shap_value, 0);
  if (
    Math.abs(reconstructed - explanation.prediction) > 0.001 ||
    Math.abs(explanation.prediction - expectedProbability) > 0.001
  ) {
    throw new Error(
      "L’explication ne correspond pas à la probabilité affichée. Recalculez les probabilités.",
    );
  }
  return explanation;
}
