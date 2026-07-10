import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";
import { predictFull } from "../api/predict";
import { fetchInductivePreview } from "../api/graph";
import { fetchInductiveNCExplanation } from "../api/explainer";
import type { NCExplainResponse } from "../api/explainer";
import NCResultCard from "../components/predictions/NCResultCard";
import LPResultTable from "../components/predictions/LPResultTable";
import CytoscapeGraph from "../components/graph/CytoscapeGraph";
import type { FullPrediction, GraphData } from "../types";

export default function InductiveDemo() {
  const navigate = useNavigate();
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [graphModel, setGraphModel] = useState("GraphSAGE");
  const [result, setResult] = useState<FullPrediction | null>(null);
  const [graphData, setGraphData] = useState<GraphData | null>(null);

  const [explainModel, setExplainModel] = useState("GraphSAGE");
  const [explainData, setExplainData] = useState<NCExplainResponse | null>(null);
  const [explainLoading, setExplainLoading] = useState(false);

  const predict = useMutation({
    mutationFn: async () => {
      const [pred, graph] = await Promise.all([
        predictFull(title, description, 10),
        fetchInductivePreview(title, description, graphModel, 5),
      ]);
      return { pred, graph };
    },
    onSuccess: ({ pred, graph }) => {
      setResult(pred);
      setGraphData(graph);
      setExplainData(null);
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (title.trim() && description.trim()) {
      predict.mutate();
    }
  };

  const handleExplain = async () => {
    if (!title.trim() || !description.trim()) return;
    setExplainLoading(true);
    try {
      const data = await fetchInductiveNCExplanation(title, description, explainModel);
      setExplainData(data);
    } catch (e) {
      console.error("Explanation failed:", e);
    } finally {
      setExplainLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Inductive Inference Demo</h1>
        <p className="text-sm text-gray-500 mt-1">
          Enter a new product (not in the graph) to see how models classify it and predict links.
        </p>
      </div>

      {/* Input form */}
      <form onSubmit={handleSubmit} className="bg-white rounded-lg shadow p-4 space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Product Title</label>
          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder='e.g. "Canon EOS R5 Mirrorless Camera"'
            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder='e.g. "45MP full-frame mirrorless camera with 8K video recording"'
            rows={3}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
          />
        </div>
        <div className="flex items-center gap-3">
          <button
            type="submit"
            disabled={predict.isPending || !title.trim() || !description.trim()}
            className="px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm font-medium hover:bg-indigo-700 disabled:opacity-50"
          >
            {predict.isPending ? "Predicting..." : "Run Inference"}
          </button>
          <select
            value={graphModel}
            onChange={(e) => setGraphModel(e.target.value)}
            className="text-sm border rounded px-2 py-1"
          >
            {["GraphSAGE", "MLP", "GCN", "GAT"].map((m) => (
              <option key={m} value={m}>{m} (graph)</option>
            ))}
          </select>
        </div>
      </form>

      {/* Warning */}
      {result?.warning && (
        <div className="bg-amber-50 border border-amber-300 rounded-lg p-3 text-sm text-amber-800">
          {result.warning}
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg shadow p-4">
            <h2 className="text-lg font-semibold mb-3">Node Classification</h2>
            <NCResultCard results={result.nc_predictions} />
          </div>
          <div className="bg-white rounded-lg shadow p-4">
            <h2 className="text-lg font-semibold mb-3">Link Predictions</h2>
            <LPResultTable
              results={result.lp_predictions}
              onAsinClick={(a) => navigate(`/product/${a}`)}
            />
          </div>
        </div>
        </div>
      )}

      {/* Graph */}
      {graphData && graphData.edges.length > 0 ? (
        <div className="bg-white rounded-lg shadow p-4">
          <h2 className="text-lg font-semibold mb-3">
            Inductive Graph Preview ({graphModel})
          </h2>
          <CytoscapeGraph
            graph={graphData}
            height={400}
            onNodeClick={(a) => navigate(`/product/${a}`)}
          />
          <p className="text-xs text-gray-500 mt-2">
            Star = new product. Dashed edges = predicted links above threshold. Click a node to view it.
          </p>
        </div>
      ) : graphData && graphData.edges.length === 0 ? (
        <div className="bg-white rounded-lg shadow p-4">
          <h2 className="text-lg font-semibold mb-3">
            Inductive Graph Preview ({graphModel})
          </h2>
          <p className="text-sm text-gray-500 py-8 text-center">
            No links predicted above the model threshold. The model is not confident enough
            to suggest any connections for this product.
          </p>
        </div>
      ) : null}

      {/* GNNExplainer for Inductive */}
      {result && (
        <div className="bg-white rounded-lg shadow p-4">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-lg font-semibold">GNNExplainer - Inductive Edge Importance</h2>
            <div className="flex items-center gap-2">
              <select
                value={explainModel}
                onChange={(e) => { setExplainModel(e.target.value); setExplainData(null); }}
                className="text-sm border rounded px-2 py-1"
              >
                {["GraphSAGE", "GCN", "GAT"].map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
              <button
                onClick={handleExplain}
                disabled={explainLoading}
                className="px-3 py-1 text-sm bg-purple-600 text-white rounded hover:bg-purple-700 disabled:opacity-50"
              >
                {explainLoading ? "Computing..." : "Explain Prediction"}
              </button>
            </div>
          </div>

          {explainLoading && (
            <div className="py-8 text-center text-sm text-gray-500">
              Optimizing edge mask...
            </div>
          )}

          {explainData && !explainLoading && (
            <>
              <div className="mb-3 flex gap-4 text-sm">
                <span>
                  Predicted: <strong className="text-purple-700">{explainData.predicted_label}</strong>
                </span>
                <span>
                  Confidence: <strong>{(explainData.confidence * 100).toFixed(1)}%</strong>
                </span>
              </div>
              <CytoscapeGraph
                graph={explainData.graph}
                height={400}
                onNodeClick={(a) => { if (a !== "__new__") navigate(`/product/${a}`); }}
              />
              <div className="flex gap-4 mt-2 text-xs text-gray-500">
                <span className="flex items-center gap-1">
                  <span className="inline-block w-4 h-1 bg-purple-600 rounded opacity-100" /> High importance
                </span>
                <span className="flex items-center gap-1">
                  <span className="inline-block w-4 h-0.5 bg-purple-600 rounded opacity-30" /> Low importance
                </span>
              </div>
              <p className="text-xs text-gray-400 mt-2">
                Shows which pseudo-neighbor edges contribute most to the GNN's classification of the new product.
              </p>
            </>
          )}

          {!explainData && !explainLoading && (
            <p className="text-sm text-gray-400 py-4 text-center">
              Click "Explain Prediction" to see which edges in the pseudo-neighborhood contribute most to classification.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
