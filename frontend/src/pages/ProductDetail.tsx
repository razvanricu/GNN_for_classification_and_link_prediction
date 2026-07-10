import { useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { fetchProduct, fetchNodePredictions, fetchLinkSuggestions } from "../api/products";
import { fetchWithTpFpFn } from "../api/graph";
import { fetchNCExplanation } from "../api/explainer";
import NCResultCard from "../components/predictions/NCResultCard";
import LPResultTable from "../components/predictions/LPResultTable";
import CytoscapeGraph from "../components/graph/CytoscapeGraph";

export default function ProductDetail() {
  const { asin } = useParams<{ asin: string }>();
  const navigate = useNavigate();
  const [graphModel, setGraphModel] = useState("GraphSAGE");
  const [explainModel, setExplainModel] = useState("GraphSAGE");
  const [explainEnabled, setExplainEnabled] = useState(false);

  const { data: product } = useQuery({
    queryKey: ["product", asin],
    queryFn: () => fetchProduct(asin!),
    enabled: !!asin,
  });

  const { data: ncResults } = useQuery({
    queryKey: ["nc", asin],
    queryFn: () => fetchNodePredictions(asin!),
    enabled: !!asin,
  });

  const { data: lpResults } = useQuery({
    queryKey: ["lp", asin],
    queryFn: () => fetchLinkSuggestions(asin!, 10),
    enabled: !!asin,
  });

  const { data: graphData } = useQuery({
    queryKey: ["graph-tpfpfn", asin, graphModel],
    queryFn: () => fetchWithTpFpFn(asin!, graphModel, 1, 30),
    enabled: !!asin,
  });

  const { data: explainData, isFetching: explainLoading } = useQuery({
    queryKey: ["explain-nc", asin, explainModel],
    queryFn: () => fetchNCExplanation(asin!, explainModel),
    enabled: !!asin && explainEnabled,
  });

  if (!asin) return null;

  return (
    <div className="space-y-6">
      {/* Product header */}
      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex items-start gap-4">
          {product?.imUrl && (
            <img
              src={product.imUrl}
              alt=""
              className="w-20 h-20 object-contain rounded border"
              onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
            />
          )}
          <div className="flex-1 min-w-0">
            <h1 className="text-lg font-bold text-gray-900 leading-tight">
              {product?.title || asin}
            </h1>
            {product?.description && (
              <p className="text-sm text-gray-600 mt-1 line-clamp-3">{product.description}</p>
            )}
            <div className="flex gap-3 mt-2 text-sm text-gray-600">
              <span className="px-2 py-0.5 bg-indigo-100 text-indigo-800 rounded text-xs font-medium">
                {product?.target_class}
              </span>
              {product?.price != null && <span>${product.price.toFixed(2)}</span>}
              <span className="font-mono text-xs text-gray-400">{asin}</span>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* NC Predictions */}
        <div className="bg-white rounded-lg shadow p-4">
          <h2 className="text-lg font-semibold mb-3">Node Classification</h2>
          {ncResults ? <NCResultCard results={ncResults} /> : <Loading />}
        </div>

        {/* LP Predictions */}
        <div className="bg-white rounded-lg shadow p-4">
          <h2 className="text-lg font-semibold mb-3">Link Predictions</h2>
          {lpResults ? (
            <LPResultTable results={lpResults} onAsinClick={(a) => navigate(`/product/${a}`)} />
          ) : (
            <Loading />
          )}
        </div>
      </div>

      {/* Graph */}
      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-semibold">Graph Neighborhood - Model Evaluation</h2>
          <select
            value={graphModel}
            onChange={(e) => setGraphModel(e.target.value)}
            className="text-sm border rounded px-2 py-1"
          >
            {["GraphSAGE", "MLP", "GCN", "GAT"].map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </div>
        {graphData ? (
          <CytoscapeGraph
            graph={graphData}
            height={450}
            onNodeClick={(a) => navigate(`/product/${a}`)}
          />
        ) : (
          <Loading />
        )}
        <div className="flex gap-4 mt-2 text-xs text-gray-500">
          <span className="flex items-center gap-1">
            <span className="inline-block w-4 h-0.5" style={{ borderTop: "2px dashed #2ecc71" }} /> True Positive
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-4 h-0.5" style={{ borderTop: "2px dashed #e74c3c" }} /> False Positive
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-4 h-0.5 bg-orange-500" /> False Negative
          </span>
        </div>
      </div>

      {/* GNNExplainer */}
      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-semibold">GNNExplainer - Edge Importance</h2>
          <div className="flex items-center gap-2">
            <select
              value={explainModel}
              onChange={(e) => setExplainModel(e.target.value)}
              className="text-sm border rounded px-2 py-1"
            >
              {["GraphSAGE", "GCN", "GAT"].map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
            <button
              onClick={() => setExplainEnabled(true)}
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
              onNodeClick={(a) => navigate(`/product/${a}`)}
            />
            <div className="flex gap-4 mt-2 text-xs text-gray-500">
              <span className="flex items-center gap-1">
                <span className="inline-block w-4 h-1 bg-purple-600 rounded opacity-100" /> High importance
              </span>
              <span className="flex items-center gap-1">
                <span className="inline-block w-4 h-0.5 bg-purple-600 rounded opacity-30" /> Low importance
              </span>
            </div>
          </>
        )}

        {!explainEnabled && !explainData && (
          <p className="text-sm text-gray-400 py-4 text-center">
            Click "Explain Prediction" to see which edges contribute most to the GNN's classification.
          </p>
        )}
      </div>
    </div>
  );
}

function Loading() {
  return <div className="py-8 text-center text-sm text-gray-400">Loading...</div>;
}
