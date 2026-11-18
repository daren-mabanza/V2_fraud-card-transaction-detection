import { useEffect, useRef, useState } from "react";
import type { Data, Layout } from "plotly.js";
import type { Explanation } from "../types";
import { delta, featureLabel, featureValue, percentage } from "../format";

export default function Waterfall({
  explanation,
}: {
  explanation: Explanation;
}) {
  const container = useRef<HTMLDivElement>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

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
        const features = explanation.features;
        const labels = [
          "Référence du modèle",
          ...features.map((f) => featureLabel(f.name)),
          "Probabilité finale",
        ];
        const trace = {
          type: "waterfall",
          orientation: "h",
          y: labels,
          x: [explanation.base_value, ...features.map((f) => f.shap_value), 0],
          measure: ["absolute", ...features.map(() => "relative"), "total"],
          text: [
            percentage(explanation.base_value),
            ...features.map((f) => delta(f.shap_value)),
            percentage(explanation.prediction),
          ],
          textposition: "outside",
          textfont: { size: 11, color: "#334155" },
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
          height: 505,
          margin: { l: 225, r: 90, t: 20, b: 65 },
          paper_bgcolor: "transparent",
          plot_bgcolor: "transparent",
          showlegend: false,
          font: {
            family: "Inter, system-ui, sans-serif",
            size: 11,
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
  }, [explanation]);

  return (
    <div className="waterfall-wrap">
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
