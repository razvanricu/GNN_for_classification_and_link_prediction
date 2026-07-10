import { NavLink } from "react-router-dom";

const links = [
  { to: "/", label: "Dashboard" },
  { to: "/catalog", label: "Catalog" },
  { to: "/inductive", label: "Inductive Demo" },
  { to: "/analytics", label: "Analytics" },
];

export default function Sidebar() {
  return (
    <aside className="w-56 bg-gray-900 text-gray-200 flex flex-col min-h-screen">
      <div className="p-4 border-b border-gray-700">
        <h1 className="text-lg font-bold text-white">GNN Thesis</h1>
        <p className="text-xs text-gray-400 mt-1">Amazon Electronics Graph</p>
      </div>
      <nav className="flex-1 p-3 space-y-1">
        {links.map((l) => (
          <NavLink
            key={l.to}
            to={l.to}
            className={({ isActive }) =>
              `block px-3 py-2 rounded text-sm ${
                isActive
                  ? "bg-indigo-600 text-white"
                  : "hover:bg-gray-800 text-gray-300"
              }`
            }
          >
            {l.label}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
