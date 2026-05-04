import type { NCResults } from "../../types";

const MODEL_COLORS: Record<string, string> = {
  GCN: "#3b82f6",
  GAT: "#10b981",
  GraphSAGE: "#f59e0b",
  MLP: "#8b5cf6",
};

export default function NCResultCard({ results }: { results: NCResults }) {
  return (
    <div className="space-y-3">
      {Object.entries(results).map(([model, pred]) => (
        <div key={model} className="border border-gray-200 rounded-lg p-3">
          <div className="flex items-center justify-between mb-2">
            <span
              className="text-xs font-bold px-2 py-0.5 rounded text-white"
              style={{ backgroundColor: MODEL_COLORS[model] || "#6b7280" }}
            >
              {model}
            </span>
            <span className="text-sm font-semibold text-gray-900">
              {pred.predicted_class}
            </span>
          </div>
          {/* Confidence bar */}
          <div className="w-full bg-gray-100 rounded-full h-2 mb-2">
            <div
              className="h-2 rounded-full"
              style={{
                width: `${pred.confidence * 100}%`,
                backgroundColor: MODEL_COLORS[model] || "#6b7280",
              }}
            />
          </div>
          <p className="text-xs text-gray-500 mb-1">
            Confidence: {(pred.confidence * 100).toFixed(1)}%
          </p>
          {/* Top 3 */}
          <div className="space-y-1">
            {pred.top_3.map((t, i) => (
              <div key={i} className="flex justify-between text-xs text-gray-600">
                <span>{t.class}</span>
                <span>{(t.probability * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
