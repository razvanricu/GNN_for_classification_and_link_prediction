import api from "./client";
import type { NCResults, LPResults, FullPrediction } from "../types";

export async function predictNC(title: string, description: string): Promise<NCResults> {
  const { data } = await api.post("/predict/nc", { title, description });
  return data;
}

export async function predictLP(
  title: string,
  description: string,
  topK = 10
): Promise<LPResults> {
  const { data } = await api.post("/predict/lp", { title, description, top_k: topK });
  return data;
}

export async function predictFull(
  title: string,
  description: string,
  topK = 10
): Promise<FullPrediction> {
  const { data } = await api.post("/predict/full", { title, description, top_k: topK });
  return data;
}
