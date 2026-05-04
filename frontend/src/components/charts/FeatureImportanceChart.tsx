import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import type { FeatureImportance } from "../../types";

export default function FeatureImportanceChart({ data }: { data: FeatureImportance[] }) {
  const top15 = data.slice(0, 15).reverse();

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={top15} layout="vertical" margin={{ left: 60 }}>
        <XAxis type="number" tick={{ fontSize: 10 }} />
        <YAxis type="category" dataKey="word" width={60} tick={{ fontSize: 11 }} />
        <Tooltip formatter={(v) => Number(v).toFixed(4)} />
        <Bar dataKey="importance" fill="#f59e0b" radius={[0, 4, 4, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
