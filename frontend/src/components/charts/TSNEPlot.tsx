import { ScatterChart, Scatter, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import type { TSNEData } from "../../types";

const COLORS = [
  "#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12",
  "#1abc9c", "#e67e22", "#f1c40f", "#34495e", "#16a085",
  "#8e44ad", "#d35400", "#27ae60", "#2980b9", "#c0392b", "#7f8c8d",
];

export default function TSNEPlot({ data }: { data: TSNEData }) {
  // Group points by class
  const grouped: Record<number, { x: number; y: number }[]> = {};
  for (let i = 0; i < data.x.length; i++) {
    const label = data.labels[i];
    if (!grouped[label]) grouped[label] = [];
    grouped[label].push({ x: data.x[i], y: data.y[i] });
  }

  return (
    <div>
      <ResponsiveContainer width="100%" height={350}>
        <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
          <XAxis type="number" dataKey="x" hide />
          <YAxis type="number" dataKey="y" hide />
          <Tooltip />
          {Object.entries(grouped).map(([label, points]) => (
            <Scatter
              key={label}
              name={data.class_names[Number(label)] || `Class ${label}`}
              data={points}
              fill={COLORS[Number(label) % COLORS.length]}
              opacity={0.6}
              shape="circle"
            />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
      {/* Legend */}
      <div className="flex flex-wrap gap-x-3 gap-y-1 mt-2">
        {data.class_names.map((name, i) => (
          <span key={i} className="flex items-center gap-1 text-[10px] text-gray-600">
            <span
              className="inline-block w-2 h-2 rounded-full"
              style={{ backgroundColor: COLORS[i % COLORS.length] }}
            />
            {name}
          </span>
        ))}
      </div>
    </div>
  );
}
