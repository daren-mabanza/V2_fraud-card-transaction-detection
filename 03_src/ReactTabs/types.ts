export type Features = Record<string, number | string | boolean | null>;
export interface Observation {
  id: string;
  date_utc: string;
  features: Features;
}
export interface DemoData {
  day: string;
  timezone: string;
  source: string;
  observations: Observation[];
}
export interface BusinessMetrics {
  source: string;
  exported_at: string;
  average_method: string;
  periods: { key: string; label: string }[];
  results: {
    period: string;
    capacity: number;
    method: string;
    fraud_on_time: number;
    relevant_cases: number;
    recoverable_amount: number;
    cases: number;
  }[];
}
export interface ShapFeature {
  name: string;
  value: number | null;
  shap_value: number;
}
export interface Explanation {
  id_observation: number;
  base_value: number;
  prediction: number;
  features: ShapFeature[];
}
export interface ShapResponse {
  model_version: string;
  output_scale: "percentage_points";
  explanations: Explanation[];
}
export interface PredictionResponse {
  model_version: string;
  model_nb_probas: number;
  model_probas: { id_observation: number; model_proba: number }[];
}
