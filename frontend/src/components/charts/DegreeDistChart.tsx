import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import type { DegreeDistribution } from "../../types";

export default function DegreeDistChart({ data }: { data: DegreeDistribution }) {
  const chartData = data.counts.map((count, i) => ({
    bin: Math.round(data.bins[i]),
    count,
  }));

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={chartData}>
        <XAxis dataKey="bin" tick={{ fontSize: 10 }} label={{ value: "Degree", position: "bottom", fontSize: 12 }} />
        <YAxis tick={{ fontSize: 10 }} label={{ value: "Count", angle: -90, position: "insideLeft", fontSize: 12 }} />
        <Tooltip />
        <Bar dataKey="count" fill="#10b981" radius={[2, 2, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
