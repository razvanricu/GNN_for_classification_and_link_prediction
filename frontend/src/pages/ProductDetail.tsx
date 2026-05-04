import { useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { fetchProduct, fetchNodePredictions, fetchLinkSuggestions } from "../api/products";
import { fetchWithPredictedLinks } from "../api/graph";
import NCResultCard from "../components/predictions/NCResultCard";
import LPResultTable from "../components/predictions/LPResultTable";
import CytoscapeGraph from "../components/graph/CytoscapeGraph";

export default function ProductDetail() {
  const { asin } = useParams<{ asin: string }>();
  const navigate = useNavigate();
  const [graphModel, setGraphModel] = useState("GraphSAGE");

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
    queryKey: ["graph", asin, graphModel],
    queryFn: () => fetchWithPredictedLinks(asin!, graphModel, 5, 1, 30),
    enabled: !!asin,
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
          <h2 className="text-lg font-semibold">Graph Neighborhood + Predicted Links</h2>
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
            <span className="inline-block w-4 h-0.5 bg-gray-300" /> Real edge
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-4 h-0.5 bg-pink-500 border-dashed" style={{ borderTop: "2px dashed #ff006e" }} /> Predicted edge
          </span>
        </div>
      </div>
    </div>
  );
}

function Loading() {
  return <div className="py-8 text-center text-sm text-gray-400">Loading...</div>;
}
