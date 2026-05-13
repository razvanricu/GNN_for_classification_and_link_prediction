import api from "./client";
import type { GraphData } from "../types";

export async function fetchNeighborhood(
  asin: string,
  hops = 2,
  maxNodes = 80
): Promise<GraphData> {
  const { data } = await api.get(`/graph/neighborhood/${asin}`, {
    params: { hops, max_nodes: maxNodes },
  });
  return data;
}

export async function fetchWithPredictedLinks(
  asin: string,
  model = "GraphSAGE",
  topK = 5,
  hops = 2,
  maxNodes = 80
): Promise<GraphData> {
  const { data } = await api.get(`/graph/with-predicted-links/${asin}`, {
    params: { model, top_k: topK, hops, max_nodes: maxNodes },
  });
  return data;
}

export async function fetchWithTpFpFn(
  asin: string,
  model = "GraphSAGE",
  hops = 1,
  maxNodes = 30
): Promise<GraphData> {
  const { data } = await api.get(`/graph/with-tp-fp-fn/${asin}`, {
    params: { model, hops, max_nodes: maxNodes },
  });
  return data;
}

export async function fetchInductivePreview(
  title: string,
  description: string,
  model = "GraphSAGE",
  topK = 5
): Promise<GraphData> {
  const { data } = await api.post(`/graph/inductive-preview`, {
    title,
    description,
    top_k: topK,
  }, { params: { model } });
  return data;
}
