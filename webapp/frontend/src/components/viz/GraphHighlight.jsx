import { useMemo, useState } from "react";
import CytoscapeComponent from "react-cytoscapejs";

/**
 * Algorithm-aware graph view — colours nodes by community (clustering algos)
 * and sizes them by score (score-style algos). Highlighted nodes get a
 * glowing border. Hover tooltips show label + score + community.
 *
 * This is a separate component from GraphView.jsx (which is preserved
 * untouched for the Step 3 graph preview).
 *
 * Props
 * -----
 * graphData : output of src/visualization/graph_viz.make_highlight_data
 *   {nodes: [{id, label, score, highlight, community}],
 *    edges: [{source, target, weight}]}
 */

const COMMUNITY_PALETTE = [
  "#2f6f7e", "#d95f43", "#7c4d96", "#3a8c4d", "#c9a227",
  "#b04e88", "#3a6b8f", "#a05a2c", "#5a8da8", "#8a7f3f",
  "#6a4f9e", "#d36b6b",
];

const MIN_NODE_SIZE = 8;
const MAX_NODE_SIZE = 24;

const CYTOSCAPE_LAYOUT = {
  name: "cose",
  animate: false,
  fit: true,
  padding: 32,
};

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function lerpColor(rgbA, rgbB, t) {
  const r = Math.round(lerp(rgbA[0], rgbB[0], t));
  const g = Math.round(lerp(rgbA[1], rgbB[1], t));
  const b = Math.round(lerp(rgbA[2], rgbB[2], t));
  return `rgb(${r}, ${g}, ${b})`;
}

function scoreGradient(t) {
  // Light teal → deep teal (keep consistent with app palette)
  return lerpColor([152, 193, 217], [47, 111, 126], t);
}

function bfsDepthGradient(t) {
  // Source/near nodes darker; far nodes lighter.
  return lerpColor([47, 111, 126], [152, 193, 217], t);
}

function colorForCommunity(community) {
  if (community === null || community === undefined) return "#2f6f7e";
  return COMMUNITY_PALETTE[
    Math.abs(Number(community)) % COMMUNITY_PALETTE.length
  ];
}

function buildElements(graphData, algorithmName) {
  const nodes = graphData?.nodes || [];
  const edges = graphData?.edges || [];

  const rawValues = nodes
    .map((n) => Number(n.score))
    .filter((s) => Number.isFinite(s));
  const minVal = rawValues.length ? Math.min(...rawValues) : 0;
  const maxVal = rawValues.length ? Math.max(...rawValues) : 1;
  const spread = Math.max(maxVal - minVal, 1e-12);

  const isCluster = algorithmName === "louvain" || algorithmName === "mcl";
  const isBfs = algorithmName === "bfs";
  const isScoreAlgo = algorithmName === "pagerank" || algorithmName === "rwr" || algorithmName === "hits";

  const nodeElements = nodes.map((n) => {
    const v = Number(n.score);
    const normRaw = rawValues.length && Number.isFinite(v) ? (v - minVal) / spread : 0.5;
    const norm = Math.max(0, Math.min(1, normRaw));

    // For BFS, smaller distance should look more important.
    const sizeNorm = isBfs ? (1 - norm) : norm;
    const size = MIN_NODE_SIZE + (MAX_NODE_SIZE - MIN_NODE_SIZE) * sizeNorm;

    const isHighlight = Boolean(n.highlight);
    let color = "#2f6f7e";
    if (isCluster) {
      color = colorForCommunity(n.community);
    } else if (isBfs) {
      color = bfsDepthGradient(norm);
      if (isHighlight) color = "#d95f43"; // emphasize source
    } else if (isScoreAlgo) {
      color = scoreGradient(norm);
      if (isHighlight) color = "#d95f43"; // top nodes highlighted
    } else {
      color = "#2f6f7e";
    }

    return {
      data: {
        id:        n.id,
        label:     n.label || n.id,
        score:     Number.isFinite(v) ? v : 0,
        community: n.community ?? null,
        highlight: isHighlight,
        size,
        color,
      },
      classes: n.highlight ? "is-highlight" : "",
    };
  });

  const edgeElements = edges.map((e, i) => ({
    data: {
      id:     `${e.source}-${e.target}-${i}`,
      source: e.source,
      target: e.target,
      weight: Number(e.weight || 1),
    },
  }));

  return [...nodeElements, ...edgeElements];
}

const STYLESHEET = [
  {
    selector: "node",
    style: {
      label:               "data(label)",
      "font-size":         8,
      color:               "#1f3f55",
      "text-outline-width": 1,
      "text-outline-color": "#ffffff",
      "background-color":  "data(color)",
      width:               "data(size)",
      height:              "data(size)",
      "border-width":      1,
      "border-color":      "#eaf1f6",
    },
  },
  {
    selector: "edge",
    style: {
      width:        1,
      "line-color": "#8ba6b7",
      opacity:      0.65,
      "curve-style": "bezier",
    },
  },
  {
    selector: ".is-highlight",
    style: {
      "border-width":  4,
      "border-color":  "#fac748",
      "border-opacity": 1,
      "z-index":       999,
    },
  },
  {
    selector: ".is-hovered",
    style: {
      "border-width": 3,
      "border-color": "#d95f43",
      "z-index":      998,
    },
  },
];

function GraphHighlight({ graphData, algorithmName }) {
  const [tooltip, setTooltip] = useState(null);

  const elements = useMemo(
    () => buildElements(graphData, algorithmName),
    [graphData, algorithmName],
  );
  const nodeCount = (graphData?.nodes || []).length;
  const edgeCount = (graphData?.edges || []).length;
  const isCapped  = Boolean(graphData?.node_count_capped);

  if (graphData?.unsupported || nodeCount === 0) {
    return (
      <div className="viz-empty">
        {graphData?.reason || "Graph view is not available for this algorithm."}
      </div>
    );
  }

  function handleCy(cy) {
    if (cy.scratch("_highlightHandlersAttached")) return;
    cy.scratch("_highlightHandlersAttached", true);

    cy.on("mouseover", "node", (event) => {
      const n = event.target;
      n.addClass("is-hovered");
      const renderedPos = n.renderedPosition();
      setTooltip({
        x: renderedPos.x,
        y: renderedPos.y,
        label:     n.data("label"),
        score:     n.data("score"),
        community: n.data("community"),
        highlight: n.data("highlight"),
      });
    });
    cy.on("mouseout", "node", (event) => {
      event.target.removeClass("is-hovered");
      setTooltip(null);
    });
  }

  return (
    <div className="graph-visualization">
      <div className="graph-visualization-header">
        <div>
          <h3>Graph Highlight</h3>
          <p>
            Rendering {nodeCount} nodes and {edgeCount} edges
            {isCapped ? " (capped for performance)" : ""}.
          </p>
        </div>
        {isCapped ? (
          <span className="graph-preview-warning">
            Showing top {nodeCount} of {graphData?.total_nodes_in_graph || "?"} nodes
          </span>
        ) : null}
      </div>

      <div className="graph-canvas" style={{ position: "relative" }}>
        <CytoscapeComponent
          cy={handleCy}
          elements={elements}
          layout={CYTOSCAPE_LAYOUT}
          maxZoom={2.5}
          minZoom={0.2}
          pan={{ x: 0, y: 0 }}
          style={{ width: "100%", height: "100%" }}
          stylesheet={STYLESHEET}
          wheelSensitivity={0.18}
        />
        {tooltip ? (
          <div
            className="graph-tooltip"
            style={{
              left: `${tooltip.x + 12}px`,
              top:  `${tooltip.y + 12}px`,
            }}
          >
            <strong>{tooltip.label}</strong>
            {tooltip.score !== undefined ? (
              <span>
                {algorithmName === "bfs" ? "distance" : "score"}: {Number(tooltip.score).toFixed(6)}
              </span>
            ) : null}
            {tooltip.community !== null && tooltip.community !== undefined ? (
              <span>community: {tooltip.community}</span>
            ) : null}
            {tooltip.highlight ? <span>★ highlighted</span> : null}
          </div>
        ) : null}
      </div>
    </div>
  );
}

export default GraphHighlight;
