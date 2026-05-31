import api from "./client";
import type { GraphData } from "../types";

export interface NCExplainResponse {
  predicted_class: number;
  predicted_label: string;
  confidence: number;
  graph: GraphData;
}

export interface LPExplainResponse {
  lp_score: number;
  linked: boolean;
  graph: GraphData;
}

export async function fetchNCExplanation(
  asin: string,
  model: string,
  numSteps = 200
): Promise<NCExplainResponse> {
  const { data } = await api.post("/explainer/nc", {
    asin,
    model,
    num_steps: numSteps,
  });
  return data;
}

export async function fetchLPExplanation(
  srcAsin: string,
  dstAsin: string,
  model: string,
  numSteps = 200
): Promise<LPExplainResponse> {
  const { data } = await api.post("/explainer/lp", {
    src_asin: srcAsin,
    dst_asin: dstAsin,
    model,
    num_steps: numSteps,
  });
  return data;
}

export async function fetchInductiveNCExplanation(
  title: string,
  description: string,
  model: string,
  numSteps = 200
): Promise<NCExplainResponse> {
  const { data } = await api.post("/explainer/nc-inductive", {
    title,
    description,
    model,
    num_steps: numSteps,
  });
  return data;
}
