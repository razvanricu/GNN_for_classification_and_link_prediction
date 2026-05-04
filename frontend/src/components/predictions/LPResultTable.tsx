import { useState } from "react";
import type { LPResults } from "../../types";

const MODEL_COLORS: Record<string, string> = {
  GCN: "#3b82f6",
  GAT: "#10b981",
  GraphSAGE: "#f59e0b",
  MLP: "#8b5cf6",
};

interface Props {
  results: LPResults;
  onAsinClick?: (asin: string) => void;
}

export default function LPResultTable({ results, onAsinClick }: Props) {
  const models = Object.keys(results);
  const [selectedModel, setSelectedModel] = useState(models[0] || "GraphSAGE");
  const suggestions = results[selectedModel] || [];

  return (
    <div>
      {/* Model tabs */}
      <div className="flex gap-1 mb-3">
        {models.map((m) => (
          <button
            key={m}
            onClick={() => setSelectedModel(m)}
            className={`px-3 py-1 rounded text-xs font-medium transition ${
              m === selectedModel
                ? "text-white"
                : "bg-gray-100 text-gray-600 hover:bg-gray-200"
            }`}
            style={m === selectedModel ? { backgroundColor: MODEL_COLORS[m] } : undefined}
          >
            {m}
          </button>
        ))}
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b">
              <th className="text-left py-2 px-2 font-medium text-gray-500">#</th>
              <th className="text-left py-2 px-2 font-medium text-gray-500">ASIN</th>
              <th className="text-left py-2 px-2 font-medium text-gray-500">Class</th>
              <th className="text-right py-2 px-2 font-medium text-gray-500">Score</th>
              <th className="text-center py-2 px-2 font-medium text-gray-500">Link?</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-50">
            {suggestions.map((s, i) => (
              <tr key={s.asin} className="hover:bg-gray-50">
                <td className="py-1.5 px-2 text-gray-400">{i + 1}</td>
                <td
                  className="py-1.5 px-2 font-mono text-indigo-600 cursor-pointer hover:underline"
                  onClick={() => onAsinClick?.(s.asin)}
                >
                  {s.asin}
                </td>
                <td className="py-1.5 px-2 text-gray-600">{s.target_class}</td>
                <td className="py-1.5 px-2 text-right font-medium">
                  {(s.score * 100).toFixed(1)}%
                </td>
                <td className="py-1.5 px-2 text-center">
                  {s.predicted_link ? (
                    <span className="text-green-600 font-bold">Yes</span>
                  ) : (
                    <span className="text-gray-400">No</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
