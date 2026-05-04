import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import type { ClassDistribution } from "../../types";

export default function ClassDistChart({ data }: { data: ClassDistribution[] }) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data} layout="vertical" margin={{ left: 120 }}>
        <XAxis type="number" />
        <YAxis type="category" dataKey="class_name" width={120} tick={{ fontSize: 11 }} />
        <Tooltip />
        <Bar dataKey="count" fill="#6366f1" radius={[0, 4, 4, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
