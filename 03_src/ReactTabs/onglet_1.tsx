import { useState } from "react";
import {
  ArrowRight,
  BarChart3,
  Database,
  Clock3,
  Layers3,
  TrendingUp,
  Info,
  CreditCard,
  ListOrdered,
  ScanLine,
} from "lucide-react";
import type { BusinessMetrics } from "./types";
import { number, percentage } from "./format";

const criteria = [
  {
    key: "fraud_on_time",
    title: "Fraudes contrôlées à temps",
    description: "Transactions frauduleuses revues avant leur confirmation.",
  },
  {
    key: "relevant_cases",
    title: "Dossiers pertinents",
    description: "Revues contenant au moins un paiement frauduleux.",
  },
  {
    key: "recoverable_amount",
    title: "Montant potentiellement récupérable",
    description:
      "Part du montant fraudé contrôlée à temps, dans la simulation.",
  },
] as const;

/**
 * Affiche la présentation et compare ML et règles à partir des métriques reçues.
 * Le mois et la capacité filtrent les résultats existants ; aucune simulation
 * n'est relancée. openDemo permet d'ouvrir l'onglet de démonstration.
 */
export default function Presentation({ openDemo, metrics }: { openDemo: () => void; metrics: BusinessMetrics }) {
  const [period, setPeriod] = useState("average");
  const [capacity, setCapacity] = useState(10);
  const results = metrics.results.filter(
    (row) => row.period === period && row.capacity === capacity,
  );
  const ml = results.find((row) => row.method === "ML")!;
  const baseline = results.find((row) => row.method === "Regle")!;

  return (
    <div className="presentation">
      <section className="hero">
        <div className="hero-copy">
          <span className="eyebrow">
            <span className="eyebrow-dot" /> DATA SCIENCE · AIDE À LA DÉCISION
          </span>
          <h1>
            Repérer la fraude.
            <br />
            <span>Comprendre le score.</span>
          </h1>
          <p>
            Chaque paiement raconte une histoire. Ce projet transforme les
            données transactionnelles en un score de fraude explicable, pour
            aider les analystes à prioriser leurs investigations.
          </p>
          <button
            className="button button-primary button-large"
            onClick={openDemo}
          >
            Tester la démonstration <ArrowRight size={18} />
          </button>
          <span className="hero-note">
            10 transactions · une journée · des explications concrètes
          </span>
        </div>
        <div className="workflow-card">
          <div className="workflow-heading">
            <span className="icon-tile">
              <ListOrdered size={21} />
            </span>
            <div>
              <span className="overline">LE DÉFI MÉTIER</span>
              <h2>Quelles cartes examiner en priorité ?</h2>
            </div>
          </div>
          <p>
            Les analystes peuvent traiter{" "}
            <strong>8 ou 10 dossiers par jour ouvré</strong>. Le modèle les aide
            à sélectionner les cartes à investiguer en premier.
          </p>
          <div className="workflow-step">
            <span>01</span>
            <div>
              <strong>Évaluer chaque paiement</strong>
              <p>Estimer sa probabilité de fraude avec XGBoost.</p>
            </div>
          </div>
          <div className="workflow-step">
            <span>02</span>
            <div>
              <strong>Classer les cartes à examiner</strong>
              <p>
                Additionner les probabilités de leurs paiements en attente. Les
                cartes au score cumulé le plus élevé passent en premier.
              </p>
            </div>
          </div>
          <div className="workflow-step">
            <span>03</span>
            <div>
              <strong>Comprendre les scores</strong>
              <p>
                Afficher les facteurs qui augmentent ou diminuent la probabilité
                de fraude d’un paiement.
              </p>
            </div>
          </div>
          <div className="workflow-foot">
            <CreditCard size={16} />
            <span>
              Un dossier = une carte et ses paiements à examiner.
            </span>
          </div>
        </div>
      </section>

      <section className="facts-grid" aria-label="Le projet en chiffres">
        <div className="fact-card">
          <span className="icon-tile">
            <Database size={21} />
          </span>
          <div>
            <strong>1,5 million</strong>
            <span>de paiements synthétiques</span>
          </div>
        </div>
        <div className="fact-card">
          <span className="icon-tile">
            <Layers3 size={21} />
          </span>
          <div>
            <strong>71 variables</strong>
            <span>métier et habitudes historiques</span>
          </div>
        </div>
        <div className="fact-card">
          <span className="icon-tile">
            <Clock3 size={21} />
          </span>
          <div>
            <strong>Historique passé</strong>
            <span>des features sans information future</span>
          </div>
        </div>
      </section>

      <section className="results-section">
        <div className="section-heading">
          <div>
            <span className="eyebrow">ÉVALUATION OPÉRATIONNELLE</span>
            <h2>Le modèle face aux règles métier</h2>
            <p>Une comparaison à capacité de traitement identique.</p>
          </div>
          <span className="quiet-badge">
            <BarChart3 size={14} /> Simulation métier
          </span>
        </div>
        <div className="results-toolbar">
          <label className="select-label">
            Période
            <select value={period} onChange={(e) => setPeriod(e.target.value)}>
              {metrics.periods.map((item) => (
                <option key={item.key} value={item.key}>
                  {item.label}
                </option>
              ))}
            </select>
          </label>
          <div className="capacity-control">
            <span>Capacité quotidienne</span>
            <div className="segmented" aria-label="Capacité quotidienne">
              {[8, 10].map((value) => (
                <button
                  key={value}
                  aria-pressed={capacity === value}
                  onClick={() => setCapacity(value)}
                >
                  {value} dossiers
                </button>
              ))}
            </div>
          </div>
          <div className="results-legend">
            <span>
              <i className="legend-ml" /> XGBoost
            </span>
            <span>
              <i className="legend-rule" /> Règles métier
            </span>
          </div>
        </div>
        <details className="methodology">
          <summary>Comprendre les données et la méthode</summary>
          <div className="methodology-content">
            <div>
              <h3>Des données françaises synthétiques</h3>
              <p>
                1,5 million de paiements fictifs. Le modèle combine contexte
                d’achat, vélocité de la carte et parts des habitudes du client.
                Ces historiques utilisent uniquement les transactions
                antérieures.
              </p>
            </div>
            <div>
              <h3>Une évaluation hors entraînement</h3>
              <p>
                Entraînement : du 31 décembre 2023 au 1er janvier 2025 exclu.
                L’évaluation métier porte sur chaque mois de janvier à juin
                2025. Les deux méthodes disposent des mêmes transactions et de
                la même capacité : 8 ou 10 dossiers par jour ouvré, selon le
                scénario choisi. Un dossier correspond à une carte. La moyenne
                donne le même poids à chacun des six mois.
              </p>
            </div>
            <div>
              <h3>Règles métier : un score à points</h3>
              <p>
                La méthode de référence attribue des points à chaque paiement :
              </p>
              <ul>
                <li>+1 si le montant est d’au moins 100 € ;</li>
                <li>+1 supplémentaire s’il est d’au moins 250 € ;</li>
                <li>+1 si le paiement est effectué hors magasin ;</li>
                <li>+1 si le marchand est situé hors de France ;</li>
                <li>
                  +1 si l’horaire est strictement entre minuit et 6 h, heure de
                  métropole.
                </li>
              </ul>
              <p>
                Ces points sont cumulables : un paiement de 300 € vaut déjà 2
                points pour son montant. Les cartes sont ensuite classées par
                somme des points de leurs paiements en attente, de la plus
                élevée à la plus faible.
              </p>
            </div>
            <div>
              <h3>XGBoost : une priorité fondée sur le modèle</h3>
              <p>
                XGBoost estime une probabilité de fraude pour chaque paiement.
                Les cartes sont classées par somme des probabilités de leurs
                paiements en attente. Le modèle remplace donc le score à points
                pour décider quels dossiers examiner en premier.
              </p>
              <p>
                Pour les deux méthodes, une égalité est départagée par le nombre
                de paiements, puis le montant total, dans l’ordre décroissant.
                Toutes les transactions sont éligibles, sans seuil de sélection.
                Les dossiers non traités restent en attente pour les jours
                ouvrés suivants du même mois. Une carte peut faire l’objet de
                plusieurs revues si de nouveaux paiements arrivent.
              </p>
            </div>
            <div className="methodology-metrics">
              <h3>À quoi correspondent les trois métriques ?</h3>
              <p>
                Chaque métrique est calculée séparément pour les deux méthodes,
                sur le mois et la capacité sélectionnés :{" "}
                <strong>bleu pour XGBoost, gris pour les règles métier</strong>.
                Le gain affiché en points correspond à la différence entre leurs
                deux taux.
              </p>
              <p>
                <strong>Fraudes contrôlées à temps :</strong> nombre de
                paiements frauduleux examinés avant leur date de confirmation,
                divisé par le nombre total de paiements frauduleux du mois.
              </p>
              <p>
                <strong>Dossiers pertinents :</strong> nombre de revues
                contenant au moins un paiement frauduleux, divisé par le nombre
                total de revues effectuées. Cette métrique compte aussi les
                fraudes examinées trop tard. Une carte examinée plusieurs fois
                compte plusieurs revues.
              </p>
              <p>
                <strong>Montant potentiellement récupérable :</strong> montant
                des fraudes examinées à temps, divisé par le montant total
                fraudé du mois. La simulation suppose que 100 % des montants
                contrôlés à temps sont récupérables ; il ne s’agit pas de sommes
                réellement récupérées.
              </p>
              <p>
                Ces résultats portent sur l’ensemble des transactions du mois,
                pas sur les dix exemples de la démonstration. Ils proviennent
                des simulations enregistrées et ne sont pas recalculés lorsque
                vous demandez un score ou une explication SHAP.
              </p>
            </div>
          </div>
        </details>
        <div className="metrics-grid">
          {criteria.map((criterion) => {
            const gain = ml[criterion.key] - baseline[criterion.key];
            return (
              <article className="metric-card" key={criterion.key}>
                <h3>{criterion.title}</h3>
                <div className="metric-score">
                  {percentage(ml[criterion.key])}
                  <span className="gain">
                    <TrendingUp size={13} /> +{number(gain, 1)} pts
                  </span>
                </div>
                <div className="comparison-row">
                  <span>XGBoost</span>
                  <strong>{percentage(ml[criterion.key])}</strong>
                </div>
                <div className="comparison-track">
                  <div style={{ width: `${ml[criterion.key]}%` }} />
                </div>
                <div className="comparison-row muted">
                  <span>Règles métier</span>
                  <strong>{percentage(baseline[criterion.key])}</strong>
                </div>
                <div className="comparison-track rule">
                  <div style={{ width: `${baseline[criterion.key]}%` }} />
                </div>
                <p>{criterion.description}</p>
              </article>
            );
          })}
        </div>
        <div className="source-note">
          <Info size={15} />
          <p>
            Résultats issus des exports métier du{" "}
            {new Intl.DateTimeFormat("fr-FR", {
              dateStyle: "long",
              timeZone: "UTC",
            }).format(new Date(metrics.exported_at))}
            , sur {number(ml.cases, 0)} revues par méthode. Le montant
            récupérable est une estimation simulée, pas une récupération
            constatée.
            {period === "average" && <> {metrics.average_method}</>}
          </p>
        </div>
      </section>
      <section className="demo-invitation">
        <span className="icon-tile">
          <ScanLine size={24} />
        </span>
        <div>
          <h2>Et si vous regardiez de plus près ?</h2>
          <p>
            Explorez dix paiements et découvrez ce qui fait évoluer leur score.
          </p>
        </div>
        <button className="button button-primary" onClick={openDemo}>
          Ouvrir la démonstration <ArrowRight size={17} />
        </button>
      </section>
    </div>
  );
}
