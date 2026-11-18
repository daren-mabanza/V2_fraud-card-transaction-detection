import { useEffect, useRef, useState } from "react";
import { Maximize2, X } from "lucide-react";
import type { Data, Layout } from "plotly.js";
import type { Explanation } from "../types";
import { delta, featureLabel, featureValue, percentage } from "../format";

export default function Waterfall({
  explanation,
  enlarged = false,
}: {
  explanation: Explanation;
  enlarged?: boolean;
}) {
  const container = useRef<HTMLDivElement>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState(false);
  const dialogRef = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    if (!expanded) return;
    const dialog = dialogRef.current!;
    dialog.showModal();
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      dialog.close();
      document.body.style.overflow = previousOverflow;
    };
  }, [expanded]);

  useEffect(() => {
    const element = container.current!;
    let disposed = false;
    let observer: ResizeObserver | undefined;
    let plotly: (typeof import("plotly.js-dist-min"))["default"] | undefined;
    setLoading(true);
    setError("");
    void import("plotly.js-dist-min")
      .then(async (module) => {
        if (disposed) return;
        plotly = module.default;
        const compact = enlarged && window.innerWidth < 600;
        const features = explanation.features;
        const labels = [
          "Référence du modèle",
          ...features.map((f) => featureLabel(f.name)),
          "Probabilité finale",
        ];
        const trace = {
          type: "waterfall",
          orientation: "h",
          y: compact
            ? labels.map(label => label.replace(/(.{1,21})(?:\s+|$)/g, "$1<br>").replace(/<br>$/, ""))
            : labels,
          x: [explanation.base_value, ...features.map((f) => f.shap_value), 0],
          measure: ["absolute", ...features.map(() => "relative"), "total"],
          text: [
            percentage(explanation.base_value),
            ...features.map((f) => delta(f.shap_value)),
            percentage(explanation.prediction),
          ],
          textposition: "outside",
          textfont: { size: compact ? 9 : 11, color: "#334155" },
          cliponaxis: false,
          customdata: [
            `Point de départ : ${percentage(explanation.base_value)}`,
            ...features.map(
              (f) =>
                `${f.name}<br>Valeur : ${f.name === "Autres variables" ? "Somme des contributions restantes" : featureValue(f.name, f.value)}<br>Contribution : ${delta(f.shap_value)} point(s)`,
            ),
            `Probabilité : ${percentage(explanation.prediction)}`,
          ],
          hovertemplate: "%{customdata}<extra></extra>",
          increasing: { marker: { color: "#EF4444" } },
          decreasing: { marker: { color: "#3B82F6" } },
          totals: { marker: { color: "#0F274D" } },
          connector: { line: { color: "#CBD5E1", width: 1, dash: "dot" } },
        } as Data;
        const layout: Partial<Layout> = {
          autosize: true,
          height: enlarged ? Math.max(540, Math.min(800, window.innerHeight - 180)) : 505,
          margin: { l: compact ? 125 : 225, r: compact ? 45 : 90, t: 20, b: 65 },
          paper_bgcolor: "transparent",
          plot_bgcolor: "transparent",
          showlegend: false,
          font: {
            family: "Inter, system-ui, sans-serif",
            size: compact ? 10 : 11,
            color: "#475569",
          },
          xaxis: {
            title: { text: "Probabilité de fraude (%)", font: { size: 11 } },
            gridcolor: "#EDF1F7",
            zerolinecolor: "#CBD5E1",
            ticksuffix: " %",
            automargin: true,
          },
          yaxis: { autorange: "reversed", fixedrange: true, automargin: true },
        };
        await plotly.newPlot(element, [trace], layout, {
          responsive: true,
          displayModeBar: false,
          locale: "fr",
        });
        if (disposed) {
          plotly.purge(element);
          return;
        }
        observer = new ResizeObserver(() => {
          if (element.clientWidth) void plotly?.Plots.resize(element);
        });
        observer.observe(element);
        setLoading(false);
      })
      .catch(() => {
        if (!disposed) {
          setError(
            "Le graphique n’a pas pu être affiché. Sélectionnez à nouveau la transaction pour réessayer.",
          );
          setLoading(false);
        }
      });
    return () => {
      disposed = true;
      observer?.disconnect();
      plotly?.purge(element);
    };
  }, [explanation, enlarged]);

  return (
    <div className={`waterfall-wrap${enlarged ? " waterfall-enlarged" : ""}`}>
      {!enlarged && (
        <>
          <div className="chart-tools">
            <button className="button button-secondary" onClick={() => setExpanded(true)}>
              <Maximize2 size={15} /> Agrandir le graphique
            </button>
          </div>
          <dialog
            ref={dialogRef}
            className="shap-dialog"
            aria-label="Graphique SHAP agrandi"
            onCancel={event => { event.preventDefault(); setExpanded(false); }}
            onClose={() => setExpanded(false)}
          >
            <div className="shap-dialog-heading">
              <div>
                <h2>Contributions à la probabilité de fraude</h2>
                <p>10 principales contributions + autres variables · Probabilité : {percentage(explanation.prediction)}</p>
              </div>
              <button className="icon-button" aria-label="Fermer le graphique agrandi" onClick={() => setExpanded(false)}>
                <X size={22} />
              </button>
            </div>
            {expanded && <Waterfall explanation={explanation} enlarged />}
          </dialog>
        </>
      )}
      {loading && (
        <p className="chart-notice" role="status">
          Construction du graphique…
        </p>
      )}
      {error && (
        <p className="alert alert-error" role="alert">
          {error}
        </p>
      )}
      <div
        ref={container}
        className="waterfall"
        role="img"
        aria-label="Waterfall SHAP : référence, dix contributions principales, autres variables et probabilité finale"
      />
    </div>
  );
}
