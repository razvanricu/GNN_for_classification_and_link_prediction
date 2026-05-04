import { useQuery } from "@tanstack/react-query";
import { fetchGraphStats, fetchClassDistribution } from "../api/analytics";
import { fetchHealth } from "../api/analytics";
import ClassDistChart from "../components/charts/ClassDistChart";

export default function Dashboard() {
  const { data: health } = useQuery({ queryKey: ["health"], queryFn: fetchHealth });
  const { data: stats } = useQuery({ queryKey: ["graph-stats"], queryFn: fetchGraphStats });
  const { data: classDist } = useQuery({
    queryKey: ["class-distribution"],
    queryFn: fetchClassDistribution,
  });

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>

      {/* Status */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatCard label="Status" value={health?.models_loaded ? "Ready" : "Loading..."} />
        <StatCard label="Nodes" value={stats?.num_nodes?.toLocaleString() ?? "..."} />
        <StatCard label="Edges" value={stats?.num_edges?.toLocaleString() ?? "..."} />
        <StatCard label="Classes" value={stats?.num_classes?.toString() ?? "..."} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <StatCard label="Avg Degree" value={stats?.avg_degree?.toString() ?? "..."} />
        <StatCard label="Density" value={stats?.density?.toFixed(6) ?? "..."} />
      </div>

      {/* Models */}
      {health && (
        <div className="bg-white rounded-lg shadow p-4">
          <h2 className="text-lg font-semibold mb-3">Loaded Models</h2>
          <div className="flex gap-6">
            <div>
              <p className="text-sm text-gray-500 mb-1">Node Classification</p>
              <div className="flex gap-2">
                {health.nc_models.map((m) => (
                  <span key={m} className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs font-medium">
                    {m}
                  </span>
                ))}
              </div>
            </div>
            <div>
              <p className="text-sm text-gray-500 mb-1">Link Prediction</p>
              <div className="flex gap-2">
                {health.lp_models.map((m) => (
                  <span key={m} className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs font-medium">
                    {m}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Class distribution */}
      {classDist && (
        <div className="bg-white rounded-lg shadow p-4">
          <h2 className="text-lg font-semibold mb-3">Class Distribution</h2>
          <ClassDistChart data={classDist} />
        </div>
      )}
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-white rounded-lg shadow p-4">
      <p className="text-sm text-gray-500">{label}</p>
      <p className="text-2xl font-bold text-gray-900">{value}</p>
    </div>
  );
}
