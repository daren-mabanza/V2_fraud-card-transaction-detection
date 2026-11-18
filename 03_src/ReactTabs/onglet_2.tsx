import { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertCircle,
  ArrowDown,
  ArrowDownUp,
  ArrowUp,
  CalendarDays,
  Check,
  CheckCircle2,
  Columns3,
  Download,
  Eye,
  Info,
  LoaderCircle,
  ScanLine,
  Search,
  X,
} from "lucide-react";
import { explainObservation, predictBatch } from "./api";
import {
  euros,
  featureLabel,
  featureValue,
  percentage,
  utcDate,
} from "./format";
import type { DemoData, Explanation, Observation } from "./types";
import Waterfall from "./components/Waterfall";

const defaultColumns = [
  "montant_eur",
  "canal",
  "mode_saisie",
  "categorie_marchand",
  "date_utc",
];
/** Renvoie le libellé d'une colonne, avec un intitulé dédié à la date UTC. */
const columnLabel = (key: string) =>
  key === "date_utc" ? "Horodatage UTC" : featureLabel(key);

/**
 * Affiche les paiements et gère leurs scores, explications et export CSV.
 * Les explications réussies restent en mémoire par paiement jusqu'au prochain
 * calcul réussi des scores ou au démontage du composant.
 */
export default function Demonstration({ dataset }: { dataset: DemoData }) {
  const observations = dataset.observations;
  const demoDay = new Intl.DateTimeFormat("fr-FR", { dateStyle: "long", timeZone: "UTC" }).format(new Date(dataset.day));
  const featureNames = Object.keys(observations[0].features);
  const optionalColumns = [...defaultColumns, ...featureNames.filter(name => !defaultColumns.includes(name))];
  const [scores, setScores] = useState<Record<string, number>>({});
  const [calculating, setCalculating] = useState(false);
  const calculatingRef = useRef(false);
  const [scoreError, setScoreError] = useState("");
  const [notice, setNotice] = useState("");
  const [visible, setVisible] = useState(new Set(defaultColumns));
  const [columnMenu, setColumnMenu] = useState(false);
  const [columnHelp, setColumnHelp] = useState(false);
  const [search, setSearch] = useState("");
  const columnMenuRef = useRef<HTMLDivElement>(null);
  const [sort, setSort] = useState<"initial" | "asc" | "desc">("initial");
  const [selected, setSelected] = useState<Observation | null>(null);
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [explaining, setExplaining] = useState(false);
  const [explanationError, setExplanationError] = useState("");
  const explanationRequest = useRef<AbortController | null>(null);
  const explanationCache = useRef(new Map<string, Explanation>());
  const explanationPanel = useRef<HTMLElement>(null);
  const hasScores = Object.keys(scores).length === observations.length;
  const columns = optionalColumns.filter((name) => visible.has(name));
  const rows = useMemo(
    () =>
      sort === "initial"
        ? observations
        : [...observations].sort(
            (a, b) =>
              (scores[a.id] - scores[b.id]) * (sort === "desc" ? -1 : 1),
          ),
    [sort, scores],
  );

  useEffect(() => {
    /** Ferme le sélecteur de colonnes et son aide lors d'un clic extérieur. */
    function closeMenu(event: MouseEvent) {
      if (!columnMenuRef.current?.contains(event.target as Node)) {
        setColumnMenu(false);
        setColumnHelp(false);
      }
    }
    /** Ferme le sélecteur de colonnes et son aide lorsque la touche Échap est pressée. */
    function escape(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setColumnMenu(false);
        setColumnHelp(false);
      }
    }
    document.addEventListener("mousedown", closeMenu);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("mousedown", closeMenu);
      document.removeEventListener("keydown", escape);
      explanationRequest.current?.abort();
    };
  }, []);

  /**
   * Calcule les probabilités en empêchant les lancements simultanés.
   * Un succès annule l'explication en cours et vide le cache SHAP ; un échec
   * conserve les scores et explications précédemment obtenus.
   */
  async function calculate() {
    if (calculatingRef.current) return;
    calculatingRef.current = true;
    setCalculating(true);
    setScoreError("");
    setNotice("");
    try {
      const nextScores = await predictBatch(observations);
      explanationRequest.current?.abort();
      explanationCache.current.clear();
      setSelected(null);
      setExplanation(null);
      setExplanationError("");
      setExplaining(false);
      setScores(nextScores);
    } catch (error) {
      setScoreError(
        error instanceof Error ? error.message : "Le calcul n’a pas abouti.",
      );
    } finally {
      setCalculating(false);
      calculatingRef.current = false;
    }
  }

  /**
   * Affiche l'explication en cache ou la demande pour le paiement sélectionné.
   * Exige des scores disponibles et annule la requête précédente. Seules les
   * réponses réussies et non annulées alimentent le cache et le panneau courant.
   */
  async function explain(row: Observation) {
    if (!hasScores) {
      setNotice("Veuillez d’abord calculer les probabilités.");
      return;
    }
    setNotice("");
    explanationRequest.current?.abort();
    setSelected(row);
    setExplanationError("");
    const cached = explanationCache.current.get(row.id);
    if (cached) {
      setExplanation(cached);
      setExplaining(false);
      return;
    }
    const request = new AbortController();
    explanationRequest.current = request;
    setExplanation(null);
    setExplaining(true);
    try {
      const result = await explainObservation(
        row,
        scores[row.id],
        request.signal,
      );
      if (!request.signal.aborted) {
        explanationCache.current.set(row.id, result);
        setExplanation(result);
      }
    } catch (error) {
      if (!request.signal.aborted)
        setExplanationError(
          error instanceof Error
            ? error.message
            : "L’explication n’a pas abouti.",
        );
    } finally {
      if (!request.signal.aborted) setExplaining(false);
    }
  }

  /** Ajoute ou retire une colonne de l'affichage, sans modifier les features envoyées. */
  function toggleColumn(name: string) {
    setVisible((previous) => {
      const next = new Set(previous);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }

  /**
   * Télécharge les paiements dans l'ordre du tableau après calcul des scores.
   * Inclut les identifiants, les colonnes affichées et les probabilités en %.
   * Le CSV utilise UTF-8 avec BOM, le point-virgule et la virgule décimale ;
   * les données exportées restent brutes, sans les unités d'affichage.
   */
  function exportCsv() {
    if (!hasScores || calculating) return;
    /**
     * Échappe une cellule CSV : guillemets doublés, null vide et virgule décimale.
     * Préfixe les chaînes pouvant être interprétées comme une formule par un tableur.
     */
    const cell = (value: unknown) => {
      let text = value == null ? "" : typeof value === "number"
        ? String(value).replace(".", ",") : String(value);
      if (typeof value === "string" && /^[=+\-@\t\r]/.test(text)) text = `'${text}`;
      return `"${text.replaceAll('"', '""')}"`;
    };
    const records = [
      ["transaction", "id_paiement", ...columns, "probabilite_fraude_pct"],
      ...rows.map(row => [
        `TX-${String(observations.indexOf(row) + 1).padStart(3, "0")}`,
        row.id,
        ...columns.map(name => name === "date_utc" ? row.date_utc : row.features[name]),
        scores[row.id],
      ]),
    ];
    const csv = "\uFEFF" + records.map(record => record.map(cell).join(";")).join("\r\n") + "\r\n";
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `fraudlens-transactions-${dataset.day}.csv`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return (
    <div className="demonstration">
      <div className="page-heading">
        <div>
          <span className="eyebrow">DÉMONSTRATION INTERACTIVE</span>
          <h1>Dix paiements à la loupe</h1>
          <p>
            Calculez les probabilités de fraude, puis explorez ce qui influence
            chaque score.
          </p>
        </div>
        <span className="date-badge">
          <CalendarDays size={16} /> {demoDay}
        </span>
      </div>
      <div className="demo-context">
        <Info size={16} />
        <span>
          10 paiements présélectionnés hors entraînement · même journée en
          France métropolitaine · horodatages affichés en UTC
        </span>
      </div>

      <section
        className="transactions-card"
        aria-label="Transactions de démonstration"
      >
        <div className="table-toolbar">
          <div className="table-title">
            <span className="icon-tile">
              <ScanLine size={20} />
            </span>
            <div>
              <h2>Transactions</h2>
              <span>10 observations · 71 variables par paiement</span>
            </div>
          </div>
          <div className="table-controls">
            <div className="column-control" ref={columnMenuRef}>
              <button
                className={`button button-secondary ${columnMenu ? "is-open" : ""}`}
                aria-expanded={columnMenu}
                aria-controls="column-menu"
                onClick={() => {
                  setColumnMenu(!columnMenu);
                  setColumnHelp(false);
                }}
              >
                <Columns3 size={16} /> Colonnes{" "}
                <span className="count-badge">{visible.size + 1}</span>
              </button>
              <button
                className="icon-button"
                aria-label="Aide sur les colonnes du tableau"
                aria-expanded={columnHelp}
                aria-controls="column-help"
                onClick={() => {
                  setColumnHelp(!columnHelp);
                  setColumnMenu(false);
                }}
              >
                <Info size={16} aria-hidden="true" />
              </button>
              {columnHelp && (
                <p className="column-help-text" id="column-help">
                  Cliquez sur « Colonnes » pour afficher d’autres variables du
                  paiement. Actions reste visible pendant le défilement ; après
                  le calcul, la probabilité de fraude reste visible aussi.
                </p>
              )}
              {columnMenu && (
                <div className="column-menu" id="column-menu">
                  <div className="column-menu-heading">
                    <strong>Colonnes du tableau</strong>
                    <button
                      className="icon-button"
                      aria-label="Fermer le menu des colonnes"
                      onClick={() => setColumnMenu(false)}
                    >
                      <X size={15} />
                    </button>
                  </div>
                  <label className="column-search">
                    <Search size={15} />
                    <input
                      autoFocus
                      placeholder="Rechercher une variable…"
                      value={search}
                      onChange={(e) => setSearch(e.target.value)}
                    />
                  </label>
                  <div className="column-options">
                    {optionalColumns
                      .filter((name) =>
                        `${name} ${columnLabel(name)}`
                          .toLocaleLowerCase("fr")
                          .includes(search.toLocaleLowerCase("fr")),
                      )
                      .map((name) => (
                        <label key={name} title={name}>
                          <input
                            type="checkbox"
                            checked={visible.has(name)}
                            onChange={() => toggleColumn(name)}
                          />
                          <span>{columnLabel(name)}</span>
                        </label>
                      ))}
                  </div>
                  <div className="column-menu-foot">
                    L’affichage n’affecte pas les variables du modèle.
                  </div>
                </div>
              )}
            </div>
            <button
              className="button button-primary"
              onClick={() => void calculate()}
              disabled={calculating}
            >
              {calculating ? (
                <LoaderCircle size={17} className="spin" />
              ) : (
                <ScanLine size={17} />
              )}
              {calculating ? "Calcul en cours…" : "Calculer les probabilités"}
            </button>
            <button
              className="button button-secondary"
              onClick={exportCsv}
              disabled={!hasScores || calculating}
              title={!hasScores ? "Calculez d’abord les probabilités pour exporter les résultats." : "Exporter les colonnes affichées et les probabilités."}
            >
              <Download size={16} /> Exporter en CSV
            </button>
          </div>
        </div>
        <div className="table-status" aria-live="polite">
          {calculating ? (
            <span>
              <LoaderCircle size={14} className="spin" /> Analyse des dix
              paiements…
            </span>
          ) : hasScores ? (
            <span className="status-success">
              <CheckCircle2 size={14} /> Probabilités calculées · cliquez sur «
              Expliquer » pour explorer un paiement.
            </span>
          ) : (
            <span>Les probabilités apparaîtront après le calcul.</span>
          )}
        </div>
        {notice && (
          <div className="alert alert-info" role="alert">
            <Info size={17} />
            <span>{notice}</span>
            <button
              className="icon-button"
              aria-label="Fermer le message"
              onClick={() => setNotice("")}
            >
              <X size={15} />
            </button>
          </div>
        )}
        {scoreError && (
          <div className="alert alert-error" role="alert">
            <AlertCircle size={18} />
            <span>{scoreError}</span>
            <button onClick={() => void calculate()} disabled={calculating}>
              Réessayer
            </button>
          </div>
        )}
        <div
          className="table-scroll"
          tabIndex={0}
          role="region"
          aria-label="Tableau des dix paiements, défilement horizontal disponible"
        >
          <table className="transactions-table">
            <thead>
              <tr>
                <th scope="col">Transaction</th>
                {columns.map((name) => (
                  <th scope="col" key={name} title={name}>
                    {columnLabel(name)}
                  </th>
                ))}
                {hasScores && (
                  <th
                    scope="col"
                    className="probability-cell"
                    aria-sort={
                      sort === "initial"
                        ? "none"
                        : sort === "desc"
                          ? "descending"
                          : "ascending"
                    }
                  >
                    <button
                      className="sort-button"
                      onClick={() => setSort(sort === "desc" ? "asc" : "desc")}
                    >
                      Probabilité de fraude{" "}
                      {sort === "initial" ? (
                        <ArrowDownUp size={13} />
                      ) : sort === "desc" ? (
                        <ArrowDown size={13} />
                      ) : (
                        <ArrowUp size={13} />
                      )}
                    </button>
                  </th>
                )}
                <th scope="col" className="action-cell">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr
                  key={row.id}
                  data-transaction-id={row.id}
                  className={selected?.id === row.id ? "selected" : ""}
                >
                  <th scope="row">
                    <div className="transaction-id">
                      TX-
                      {String(observations.indexOf(row) + 1).padStart(3, "0")}
                    </div>
                    <code title={row.id}>
                      {row.id.slice(0, 8)}…{row.id.slice(-4)}
                    </code>
                  </th>
                  {columns.map((name) => (
                    <td
                      key={name}
                      className={name === "montant_eur" ? "amount-cell" : ""}
                    >
                      {name === "date_utc" ? (
                        <span className="timestamp">
                          {utcDate(row.date_utc)}
                        </span>
                      ) : name === "canal" ? (
                        <span className="channel-tag">
                          {String(row.features[name])}
                        </span>
                      ) : (
                        featureValue(name, row.features[name])
                      )}
                    </td>
                  ))}
                  {hasScores && (
                    <td className="probability-value probability-cell">
                      <strong>{percentage(scores[row.id])}</strong>
                      <span className="probability-track" aria-hidden="true">
                        <span style={{ width: `${scores[row.id]}%` }} />
                      </span>
                    </td>
                  )}
                  <td className="action-cell">
                    <button
                      className={`explain-button ${selected?.id === row.id ? "active" : ""}`}
                      onClick={() => void explain(row)}
                      aria-label={`Expliquer la transaction TX-${String(observations.indexOf(row) + 1).padStart(3, "0")}`}
                      aria-pressed={selected?.id === row.id}
                    >
                      {selected?.id === row.id && explaining ? (
                        <LoaderCircle size={14} className="spin" />
                      ) : (
                        <Eye size={14} />
                      )}{" "}
                      Expliquer
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="table-foot">
          <span>
            <Check size={13} /> Données de démonstration synthétiques
          </span>
          <span>10 transactions</span>
        </div>
      </section>

      <section
        className={`explanation-panel ${selected ? "has-selection" : ""}`}
        ref={explanationPanel}
        aria-label="Explication SHAP"
        aria-busy={explaining}
      >
        {!selected ? (
          <div className="explanation-empty">
            <span className="empty-icon">
              <Eye size={25} />
            </span>
            <div>
              <h2>Un score, des explications.</h2>
              <p>
                Après le calcul, sélectionnez « Expliquer » sur une transaction
                pour découvrir ses contributions SHAP.
              </p>
            </div>
            <span className="quiet-badge">Explication individuelle</span>
          </div>
        ) : (
          <>
            <div className="explanation-heading">
              <div>
                <span className="eyebrow">COMPRENDRE CETTE TRANSACTION</span>
                <h2>
                  Explication de TX-
                  {String(observations.indexOf(selected) + 1).padStart(3, "0")}
                </h2>
                <code>{selected.id}</code>
              </div>
              <div className="selected-score">
                <span>Probabilité de fraude</span>
                <strong>{percentage(scores[selected.id])}</strong>
              </div>
            </div>
            {explaining ? (
              <div className="explanation-loading" role="status">
                <LoaderCircle size={29} className="spin" />
                <strong>Calcul des contributions SHAP…</strong>
                <span>Le modèle examine les facteurs de ce paiement.</span>
              </div>
            ) : explanationError ? (
              <div className="alert alert-error" role="alert">
                <AlertCircle size={18} />
                <span>{explanationError}</span>
                <button onClick={() => void explain(selected)}>
                  Réessayer
                </button>
              </div>
            ) : (
              explanation && (
                <div className="explanation-content">
                  <div className="chart-column">
                    <div className="chart-heading">
                      <h3>Contributions à la probabilité de fraude</h3>
                      <span>10 principales + autres variables</span>
                    </div>
                    <Waterfall explanation={explanation} />
                    <div className="chart-legend">
                      <span>
                        <i className="legend-increase" /> Augmente le score
                      </span>
                      <span>
                        <i className="legend-decrease" /> Diminue le score
                      </span>
                      <span>
                        <i className="legend-total" /> Référence et résultat
                      </span>
                    </div>
                  </div>
                  <aside className="shap-help">
                    <span className="icon-tile">
                      <Info size={20} />
                    </span>
                    <h3>Comment lire ce graphique ?</h3>
                    <p>
                      Le graphique part de la probabilité de référence du
                      modèle.
                    </p>
                    <p>
                      Chaque barre{" "}
                      <strong className="text-increase">rouge</strong> augmente
                      le score de fraude ; chaque barre{" "}
                      <strong className="text-decrease">bleue</strong> le
                      diminue. Sa longueur indique l’importance de sa
                      contribution pour ce paiement.
                    </p>
                    <p>
                      Les contributions sont exprimées en{" "}
                      <strong>points de pourcentage</strong>. Leur somme,
                      ajoutée à la référence, reconstitue la probabilité finale.
                    </p>
                    <p>
                      « Autres variables » rassemble les contributions
                      restantes, avec leur signe. Survolez une barre pour voir
                      la valeur de la variable transformée.
                    </p>
                    <div className="shap-equation">
                      <span>Référence + contributions</span>
                      <strong>= probabilité finale</strong>
                    </div>
                    <p className="help-footnote">
                      Une explication du modèle décrit son calcul ; elle
                      n’établit pas la cause d’une fraude.
                    </p>
                  </aside>
                </div>
              )
            )}
          </>
        )}
      </section>
    </div>
  );
}
