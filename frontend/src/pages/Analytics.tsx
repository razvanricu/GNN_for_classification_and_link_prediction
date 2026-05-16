import { useQuery } from "@tanstack/react-query";
import {
  fetchClassDistribution,
  fetchDegreeDistribution,
  fetchTSNE,
} from "../api/analytics";
import ClassDistChart from "../components/charts/ClassDistChart";
import DegreeDistChart from "../components/charts/DegreeDistChart";
import TSNEPlot from "../components/charts/TSNEPlot";

export default function Analytics() {
  const { data: classDist } = useQuery({
    queryKey: ["class-distribution"],
    queryFn: fetchClassDistribution,
  });
  const { data: degreeDist } = useQuery({
    queryKey: ["degree-distribution"],
    queryFn: fetchDegreeDistribution,
  });
  const { data: tsne } = useQuery({
    queryKey: ["tsne"],
    queryFn: () => fetchTSNE(2000),
  });

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-900">Analytics</h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Class Distribution */}
        <div className="bg-white rounded-lg shadow p-4">
          <h2 className="text-lg font-semibold mb-3">Class Distribution</h2>
          {classDist ? <ClassDistChart data={classDist} /> : <Loading />}
        </div>

        {/* Degree Distribution */}
        <div className="bg-white rounded-lg shadow p-4">
          <h2 className="text-lg font-semibold mb-3">Degree Distribution</h2>
          {degreeDist ? <DegreeDistChart data={degreeDist} /> : <Loading />}
        </div>

        {/* t-SNE */}
        <div className="bg-white rounded-lg shadow p-4">
          <h2 className="text-lg font-semibold mb-3">t-SNE Embedding (2000 nodes)</h2>
          {tsne ? <TSNEPlot data={tsne} /> : <Loading />}
        </div>
      </div>

      {/* Model Evaluation Charts */}
      <h2 className="text-xl font-bold text-gray-900 mt-8">Model Evaluation</h2>

      <h3 className="text-lg font-semibold text-gray-700">Node Classification</h3>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow p-4">
          <h4 className="text-sm font-semibold mb-2 text-gray-600">Performance Comparison</h4>
          <img src="/nc_comparison_bar.png" alt="Node Classification - Performance Comparison" className="w-full rounded" />
        </div>
        <div className="bg-white rounded-lg shadow p-4">
          <h4 className="text-sm font-semibold mb-2 text-gray-600">Confusion Matrices</h4>
          <img src="/nc_confusion_matrices.png" alt="Node Classification - Confusion Matrices" className="w-full rounded" />
        </div>
      </div>

      <h3 className="text-lg font-semibold text-gray-700">Link Prediction</h3>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow p-4">
          <h4 className="text-sm font-semibold mb-2 text-gray-600">Performance Comparison</h4>
          <img src="/lp_comparison_bar.png" alt="Link Prediction - Performance Comparison" className="w-full rounded" />
        </div>
        <div className="bg-white rounded-lg shadow p-4">
          <h4 className="text-sm font-semibold mb-2 text-gray-600">ROC Curves</h4>
          <img src="/lp_roc_overlay.png" alt="Link Prediction - ROC Curves" className="w-full rounded" />
        </div>
        <div className="bg-white rounded-lg shadow p-4 lg:col-span-2">
          <h4 className="text-sm font-semibold mb-2 text-gray-600">Confusion Matrices</h4>
          <img src="/lp_confusion_matrices.png" alt="Link Prediction - Confusion Matrices" className="w-full rounded" />
        </div>
      </div>
    </div>
  );
}

function Loading() {
  return <div className="py-8 text-center text-sm text-gray-400">Loading...</div>;
}
