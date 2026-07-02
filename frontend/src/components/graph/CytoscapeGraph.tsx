import { useEffect, useRef } from "react";
import cytoscape from "cytoscape";
import type { GraphData } from "../../types";

const CLASS_COLORS: Record<string, string> = {
  "Film Photography": "#e74c3c",
  "Digital Cameras": "#3498db",
  "Binoculars & Scopes": "#2ecc71",
  "Lenses": "#9b59b6",
  "Tripods & Monopods": "#f39c12",
  "Video Surveillance": "#1abc9c",
  "Lighting & Studio": "#e67e22",
  "Flashes": "#f1c40f",
  "Video Cables": "#34495e",
  "Audio Cables": "#16a085",
  "TV Accessories & Parts": "#8e44ad",
  "Headphone Accessories": "#d35400",
  "Connectors & Adapters": "#27ae60",
  "Media Storage & Organization": "#2980b9",
  "Distribution": "#c0392b",
  "Telephone Accessories": "#7f8c8d",
  "New Product": "#ff006e",
};

interface Props {
  graph: GraphData;
  height?: number;
  onNodeClick?: (asin: string) => void;
}

export default function CytoscapeGraph({ graph, height = 500, onNodeClick }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<cytoscape.Core | null>(null);

  useEffect(() => {
    if (!containerRef.current || !graph) return;

    const elements = [
      ...graph.nodes.map((n) => ({
        group: "nodes" as const,
        data: { ...n.data, color: CLASS_COLORS[n.data.class] || "#95a5a6" },
      })),
      ...graph.edges.map((e) => ({
        group: "edges" as const,
        data: e.data,
      })),
    ];

    if (cyRef.current) {
      cyRef.current.destroy();
    }

    const cy = cytoscape({
      container: containerRef.current,
      elements,
      style: [
        {
          selector: "node",
          style: {
            "background-color": "data(color)",
            label: "data(label)",
            "font-size": "8px",
            "text-valign": "bottom",
            "text-margin-y": 4,
            width: 20,
            height: 20,
          },
        },
        {
          selector: "node[?is_center]",
          style: {
            width: 35,
            height: 35,
            "border-width": 3,
            "border-color": "#000",
          },
        },
        {
          selector: "node[?is_new]",
          style: {
            shape: "star",
            width: 40,
            height: 40,
            "border-width": 2,
            "border-color": "#ff006e",
          },
        },
        {
          selector: "node[?is_predicted_neighbor]",
          style: {
            "border-width": 2,
            "border-color": "#ff006e",
            "border-style": "dashed",
          },
        },
        {
          selector: "node[?is_fp_neighbor]",
          style: {
            "border-width": 2,
            "border-color": "#e74c3c",
            "border-style": "dashed",
          },
        },
        {
          selector: "edge[type='real']",
          style: {
            "line-color": "#ccc",
            width: 1,
            "curve-style": "bezier",
          },
        },
        {
          selector: "edge[type='predicted']",
          style: {
            "line-color": "#ff006e",
            "line-style": "dashed",
            width: 2,
            "curve-style": "bezier",
          },
        },
        {
          selector: "edge[type='tp']",
          style: {
            "line-color": "#2ecc71",
            "line-style": "dashed",
            width: 2,
            "curve-style": "bezier",
          },
        },
        {
          selector: "edge[type='fp']",
          style: {
            "line-color": "#e74c3c",
            "line-style": "dashed",
            width: 2,
            "curve-style": "bezier",
          },
        },
        {
          selector: "edge[type='fn']",
          style: {
            "line-color": "#e67e22",
            width: 2,
            "curve-style": "bezier",
          },
        },
        {
          selector: "edge[type='explained']",
          style: {
            "line-color": "#8b5cf6",
            width: "mapData(importance, 0, 1, 1, 8)",
            opacity: "mapData(importance, 0, 1, 0.3, 1)" as unknown as number,
            "curve-style": "bezier",
          },
        },
        {
          selector: "node[importance]",
          style: {
            width: "mapData(importance, 0, 1, 15, 40)",
            height: "mapData(importance, 0, 1, 15, 40)",
            "border-width": 2,
            "border-color": "#8b5cf6",
          },
        },
      ],
      layout: {
        name: "concentric",
        concentric: (node: cytoscape.NodeSingular) =>
          node.data("is_center") || node.data("is_new") ? 10 : 1,
        levelWidth: () => 2,
        animate: false,
        minNodeSpacing: 30,
      },
      wheelSensitivity: 0.3,
    });

    if (onNodeClick) {
      cy.on("tap", "node", (evt) => {
        const id = evt.target.data("id");
        if (id && id !== "__new__") onNodeClick(id);
      });
    }

    cyRef.current = cy;

    return () => {
      cy.destroy();
    };
  }, [graph, onNodeClick]);

  return (
    <div
      ref={containerRef}
      style={{ height }}
      className="w-full border border-gray-200 rounded-lg bg-white"
    />
  );
}
