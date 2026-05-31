export interface Product {
  asin: string;
  title: string;
  description?: string;
  price: number | null;
  imUrl: string | null;
  target_class: string;
  in_graph: boolean;
  graph_idx: number;
}

export interface CatalogResponse {
  items: Product[];
  total: number;
  page: number;
  size: number;
}

export interface NCPrediction {
  predicted_class: string;
  confidence: number;
  top_3: { class: string; probability: number }[];
}

export type NCResults = Record<string, NCPrediction>;

export interface LPSuggestion {
  asin: string;
  score: number;
  predicted_link: boolean;
  target_class: string;
}

export type LPResults = Record<string, LPSuggestion[]>;

export interface FullPrediction {
  nc_predictions: NCResults;
  lp_predictions: LPResults;
  warning?: string;
}

export interface CytoscapeNode {
  data: {
    id: string;
    label: string;
    class: string;
    is_center?: boolean;
    is_new?: boolean;
    is_predicted_neighbor?: boolean;
    is_fp_neighbor?: boolean;
    importance?: number;
  };
}

export interface CytoscapeEdge {
  data: {
    source: string;
    target: string;
    type: "real" | "predicted" | "tp" | "fp" | "fn" | "explained";
    model?: string;
    score?: number;
    importance?: number;
  };
}

export interface GraphData {
  nodes: CytoscapeNode[];
  edges: CytoscapeEdge[];
}

export interface ClassDistribution {
  class_name: string;
  count: number;
}

export interface DegreeDistribution {
  bins: number[];
  counts: number[];
}

export interface FeatureImportance {
  word: string;
  importance: number;
}

export interface TSNEData {
  x: number[];
  y: number[];
  labels: number[];
  asins: string[];
  class_names: string[];
}

export interface GraphStats {
  num_nodes: number;
  num_edges: number;
  num_classes: number;
  avg_degree: number;
  density: number;
}

export interface HealthResponse {
  status: string;
  models_loaded: boolean;
  nc_models: string[];
  lp_models: string[];
}
