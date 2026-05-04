import { useQuery } from "@tanstack/react-query";
import {
  fetchClassDistribution,
  fetchDegreeDistribution,
  fetchFeatureImportance,
  fetchTSNE,
} from "../api/analytics";
import ClassDistChart from "../components/charts/ClassDistChart";
import DegreeDistChart from "../components/charts/DegreeDistChart";
import FeatureImportanceChart from "../components/charts/FeatureImportanceChart";
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
  const { data: featureImp } = useQuery({
    queryKey: ["feature-importance"],
    queryFn: fetchFeatureImportance,
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

        {/* Feature Importance */}
        <div className="bg-white rounded-lg shadow p-4">
          <h2 className="text-lg font-semibold mb-3">Top TF-IDF Features</h2>
          {featureImp ? <FeatureImportanceChart data={featureImp} /> : <Loading />}
        </div>

        {/* t-SNE */}
        <div className="bg-white rounded-lg shadow p-4">
          <h2 className="text-lg font-semibold mb-3">t-SNE Embedding (2000 nodes)</h2>
          {tsne ? <TSNEPlot data={tsne} /> : <Loading />}
        </div>
      </div>
    </div>
  );
}

function Loading() {
  return <div className="py-8 text-center text-sm text-gray-400">Loading...</div>;
}
