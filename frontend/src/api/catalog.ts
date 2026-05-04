import api from "./client";
import type { CatalogResponse } from "../types";

export interface CatalogParams {
  q?: string;
  category?: string;
  min_price?: number;
  max_price?: number;
  in_graph?: boolean;
  page?: number;
  size?: number;
}

export async function fetchCatalog(params: CatalogParams): Promise<CatalogResponse> {
  const { data } = await api.get("/catalog", { params });
  return data;
}

export async function fetchCategories(): Promise<string[]> {
  const { data } = await api.get("/catalog/categories");
  return data;
}

export async function fetchCatalogStats() {
  const { data } = await api.get("/catalog/stats");
  return data;
}
