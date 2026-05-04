import api from "./client";

export interface VocabWord {
  word: string;
  idf: number;
  index: number;
}

export interface VocabularyResponse {
  total: number;
  words: VocabWord[];
}

export async function fetchVocabulary(): Promise<VocabularyResponse> {
  const { data } = await api.get("/analytics/vocabulary");
  return data;
}
