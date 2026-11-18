const labels: Record<string, string> = {
  montant_eur: "Montant",
  categorie_marchand: "Catégorie marchand",
  pays_marchand: "Pays marchand",
  pays_ip: "Pays IP",
  canal: "Canal",
  mode_saisie: "Mode de saisie",
  heure: "Heure locale",
  distance_domicile_marchand_km: "Distance domicile–marchand",
  delai_precedent_carte_h: "Délai depuis le paiement précédent",
  nb_carte_1h_passe: "Paiements sur la dernière heure",
  nb_carte_jour_passe: "Paiements depuis minuit",
  nb_appareils_carte_passe: "Appareils déjà utilisés",
  anciennete_carte_j: "Ancienneté de la carte",
  ecart_mediane_enseigne_eur: "Écart à la médiane de l’enseigne",
  ecart_mediane_carte_eur: "Écart à la médiane de la carte",
  score_anomalie: "Score d’anomalie",
  "Autres variables": "Autres variables",
};
const modalities: Record<string, string> = {
  alimentation: "alimentation",
  boulangerie: "boulangerie",
  carburant: "carburant",
  habillement: "habillement",
  hebergement: "hébergement",
  loisirs: "loisirs",
  pharmacie: "pharmacie",
  restauration: "restauration",
  services_numeriques: "services numériques",
  transport: "transport",
  electronique: "électronique",
  equipement_de_la_maison: "équipement de la maison",
  application: "application",
  magasin: "magasin",
  site_web: "site web",
  a_distance: "à distance",
  carte_enregistree: "carte enregistrée",
  portefeuille_mobile: "portefeuille mobile",
  puce: "puce",
  saisie_en_ligne: "saisie en ligne",
  sans_contact: "sans contact",
};
export const number = (value: number, digits = 2) =>
  new Intl.NumberFormat("fr-FR", { maximumFractionDigits: digits }).format(
    value,
  );
export const percentage = (value: number) => `${number(value, 2)} %`;
export const euros = (value: number) =>
  new Intl.NumberFormat("fr-FR", { style: "currency", currency: "EUR" }).format(
    value,
  );
export const delta = (value: number) =>
  `${value > 0 ? "+" : value < 0 ? "−" : ""}${Math.abs(value) > 0 && Math.abs(value) < 0.0001 ? "< 0,0001" : number(Math.abs(value), 4)}`;
export const utcDate = (value: string) =>
  new Intl.DateTimeFormat("fr-FR", {
    timeZone: "UTC",
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date(value));

export function featureLabel(name: string): string {
  if (labels[name]) return labels[name];
  const hour = name.match(/^part_client_heure_(\d+)_passe$/);
  if (hour) return `Habitude horaire · ${hour[1]} h`;
  const encodedHour = name.match(/^heure_(\d+)(?:\.0)?$/);
  if (encodedHour) return `Heure locale · ${encodedHour[1].padStart(2, "0")} h`;
  const families = [
    ["part_client_categorie_marchand_", "Habitude marchand"],
    ["part_client_pays_marchand_", "Habitude pays"],
    ["part_client_canal_", "Habitude canal"],
    ["part_client_mode_saisie_", "Habitude de saisie"],
  ];
  for (const [prefix, label] of families) {
    if (name.startsWith(prefix)) {
      const key = name.slice(prefix.length).replace(/_passe$/, "");
      return `${label} · ${modalities[key] || key.toUpperCase()}`;
    }
  }
  for (const key of [
    "categorie_marchand",
    "pays_marchand",
    "pays_ip",
    "mode_saisie",
    "canal",
  ]) {
    if (name.startsWith(`${key}_`)) {
      const modality = name.slice(key.length + 1);
      return `${labels[key]} · ${/^(nan|none|null)$/i.test(modality) ? "Non renseigné" : modality}`;
    }
  }
  return name.replaceAll("_", " ");
}

export function featureValue(name: string, value: unknown): string {
  if (value === null || value === undefined) return "Non renseigné";
  if (typeof value === "boolean") return value ? "Oui" : "Non";
  if (typeof value !== "number") return String(value);
  if (name.startsWith("part_client_")) return percentage(value * 100);
  if (name.endsWith("_eur")) return euros(value);
  if (name.endsWith("_km")) return `${number(value)} km`;
  if (name.endsWith("_h")) return `${number(value)} h`;
  if (name.endsWith("_j")) return `${number(value)} j`;
  return number(value, 4);
}
