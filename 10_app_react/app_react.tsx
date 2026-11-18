import { StrictMode, useCallback, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  BookOpen, ScanLine, ShieldCheck, ArrowUpRight, LoaderCircle,
} from "lucide-react";
import Presentation from "../03_src/ReactTabs/onglet_1";
import Demonstration from "../03_src/ReactTabs/onglet_2";
import { loadExport } from "../03_src/ReactTabs/api";
import type { BusinessMetrics, DemoData } from "../03_src/ReactTabs/types";
import "./style.css";

function useExport<T>(route: string) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState("");
  const load = useCallback(
    async (signal?: AbortSignal) => {
      setError("");
      try {
        setData(await loadExport<T>(route, signal));
      } catch (reason) {
        if (!signal?.aborted) {
          setError(
            reason instanceof Error && !(reason instanceof TypeError)
              ? reason.message
              : "Connexion à l’API impossible. Vérifiez que le service est démarré.",
          );
        }
      }
    },
    [route],
  );
  useEffect(() => {
    const request = new AbortController();
    void load(request.signal);
    return () => request.abort();
  }, [load]);
  return { data, error, retry: () => void load() };
}

function ExportStatus({ error, retry }: { error: string; retry: () => void }) {
  return (
    <div className="explanation-panel">
      <div className="explanation-empty">
        {error ? (
          <>
            <p role="alert">{error}</p>
            <button className="button button-secondary" onClick={retry}>Réessayer</button>
          </>
        ) : (
          <p role="status">
            <LoaderCircle size={17} className="spin" /> Chargement des données du projet…
          </p>
        )}
      </div>
    </div>
  );
}

function App() {
  const metrics = useExport<BusinessMetrics>("business-metrics");
  const dataset = useExport<DemoData>("demo-data");
  const [tab, setTab] = useState<"presentation" | "demonstration">(
    "presentation",
  );
  function openDemo() {
    setTab("demonstration");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
  return (
    <>
      <header className="app-header">
        <div className="header-inner">
          <a
            className="brand"
            href="#"
            onClick={(event) => {
              event.preventDefault();
              setTab("presentation");
            }}
            aria-label="FraudLens, présentation du projet"
          >
            <span className="brand-icon">
              <ShieldCheck size={29} strokeWidth={1.8} />
            </span>
            <span>
              <strong>FraudLens</strong>
              <small>Détection de fraude bancaire</small>
            </span>
          </a>
          <div className="header-meta">
            <span className="quiet-badge">
              <span className="small-dot" /> Données synthétiques
            </span>
            <a
              className="button button-secondary github-link"
              href="https://github.com/daren-mabanza/V2_fraud-card-transaction-detection"
              target="_blank"
              rel="noopener noreferrer"
              aria-label="Voir le projet sur GitHub (nouvel onglet)"
            >
              <svg
                width="17"
                height="17"
                viewBox="0 0 24 24"
                fill="currentColor"
                aria-hidden="true"
              >
                <path d="M12 .297a12 12 0 0 0-3.793 23.385c.6.111.82-.261.82-.577v-2.234c-3.338.726-4.043-1.416-4.043-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.09-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.071 1.835 2.809 1.305 3.495.998.108-.776.419-1.305.762-1.605-2.665-.303-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.536-1.524.117-3.176 0 0 1.008-.322 3.301 1.23a11.51 11.51 0 0 1 6.006 0c2.291-1.552 3.297-1.23 3.297-1.23.655 1.652.243 2.873.119 3.176.77.84 1.235 1.91 1.235 3.221 0 4.609-2.807 5.625-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.216.694.825.576A12.001 12.001 0 0 0 12 .297Z" />
              </svg>
              <span>GitHub</span>
            </a>
            <span className="model-tag">
              XGBoost <b>V2</b>
            </span>
          </div>
        </div>
      </header>
      <div className="navigation-shell">
        <nav
          className="tab-navigation"
          role="tablist"
          aria-label="Navigation principale"
          onKeyDown={(event) => {
            if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
            event.preventDefault();
            const next =
              tab === "presentation" ? "demonstration" : "presentation";
            setTab(next);
            document.getElementById(`tab-${next}`)?.focus();
          }}
        >
          <button
            id="tab-presentation"
            role="tab"
            aria-selected={tab === "presentation"}
            aria-controls="panel-presentation"
            tabIndex={tab === "presentation" ? 0 : -1}
            onClick={() => setTab("presentation")}
          >
            <BookOpen size={17} /> Présentation du projet
          </button>
          <button
            id="tab-demonstration"
            role="tab"
            aria-selected={tab === "demonstration"}
            aria-controls="panel-demonstration"
            tabIndex={tab === "demonstration" ? 0 : -1}
            onClick={() => setTab("demonstration")}
          >
            <ScanLine size={17} /> Démonstration{" "}
            <ArrowUpRight size={13} className="tab-arrow" />
          </button>
        </nav>
      </div>
      <main className="app-main">
        <section
          id="panel-presentation"
          role="tabpanel"
          aria-labelledby="tab-presentation"
          hidden={tab !== "presentation"}
        >
          {metrics.data ? <Presentation openDemo={openDemo} metrics={metrics.data} /> : <ExportStatus error={metrics.error} retry={metrics.retry} />}
        </section>
        <section
          id="panel-demonstration"
          role="tabpanel"
          aria-labelledby="tab-demonstration"
          hidden={tab !== "demonstration"}
        >
          {dataset.data ? <Demonstration dataset={dataset.data} /> : <ExportStatus error={dataset.error} retry={dataset.retry} />}
        </section>
      </main>
      <footer className="app-footer">
        <span>
          <ShieldCheck size={15} /> FraudLens{" "}
          <span className="footer-divider">/</span> Projet de data science
        </span>
        <span>Données fictives · Aucune opération bancaire exécutée</span>
      </footer>
    </>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
