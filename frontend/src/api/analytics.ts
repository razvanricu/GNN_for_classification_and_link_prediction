import api from "./client";
import type {
  ClassDistribution,
  DegreeDistribution,
  TSNEData,
  GraphStats,
  HealthResponse,
} from "../types";

export async function fetchHealth(): Promise<HealthResponse> {
  const { data } = await api.get("/health");
  return data;
}

export async function fetchGraphStats(): Promise<GraphStats> {
  const { data } = await api.get("/analytics/graph-stats");
  return data;
}

export async function fetchClassDistribution(): Promise<ClassDistribution[]> {
  const { data } = await api.get("/analytics/class-distribution");
  return data;
}

export async function fetchDegreeDistribution(): Promise<DegreeDistribution> {
  const { data } = await api.get("/analytics/degree-distribution");
  return data;
}

export async function fetchTSNE(sample = 2000): Promise<TSNEData> {
  const { data } = await api.get("/analytics/tsne", { params: { sample } });
  return data;
}
