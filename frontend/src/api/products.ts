import api from "./client";
import type { Product, NCResults, LPResults } from "../types";

export async function fetchProduct(asin: string): Promise<Product> {
  const { data } = await api.get(`/products/${asin}`);
  return data;
}

export async function fetchNodePredictions(asin: string): Promise<NCResults> {
  const { data } = await api.get(`/products/${asin}/predictions`);
  return data;
}

export async function fetchLinkSuggestions(asin: string, topK = 10): Promise<LPResults> {
  const { data } = await api.get(`/products/${asin}/link-suggestions`, {
    params: { top_k: topK },
  });
  return data;
}
