import CytoscapeComponent from "react-cytoscapejs";

const MAX_RENDER_NODES = 500;
const MAX_RENDER_EDGES = 1000;

const CYTOSCAPE_LAYOUT = {
  name: "cose",
  animate: false,
  fit: true,
  padding: 24,
};

const CYTOSCAPE_STYLESHEET = [
  {
    selector: "node",
    style: {
      label: "data(label)",
      "font-size": 8,
      color: "#1f3f55",
      "text-outline-width": 1,
      "text-outline-color": "#ffffff",
      "background-color": "#2f6f7e",
      width: 14,
      height: 14,
      "border-width": 1,
      "border-color": "#eaf1f6",
    },
  },
  {
    selector: "edge",
    style: {
      width: 1,
      "line-color": "#8ba6b7",
      opacity: 0.75,
      "curve-style": "bezier",
    },
  },
  {
    selector: ".is-active",
    style: {
      "background-color": "#d95f43",
      "border-width": 3,
      "border-color": "#762d1d",
      width: 18,
      height: 18,
      "z-index": 999,
    },
  },
];

function buildGraphPreview(graph) {
  const nodes = graph?.nodes || {};
  const edges = graph?.edges || [];
  const nodeEntries = Object.entries(nodes);
  const limitedNodeEntries = nodeEntries.slice(0, MAX_RENDER_NODES);
  const allowedNodeIds = new Set(limitedNodeEntries.map(([nodeId]) => nodeId));

  const limitedEdges = [];
  for (const edge of edges) {
    if (!allowedNodeIds.has(edge.source) || !allowedNodeIds.has(edge.target)) {
      continue;
    }

    limitedEdges.push(edge);
    if (limitedEdges.length >= MAX_RENDER_EDGES) {
      break;
    }
  }

  const elements = [
    ...limitedNodeEntries.map(([nodeId]) => ({
      data: {
        id: nodeId,
        label: nodeId,
      },
    })),
    ...limitedEdges.map((edge, index) => ({
      data: {
        id: `${edge.source}-${edge.target}-${index}`,
        source: edge.source,
        target: edge.target,
      },
    })),
  ];

  const isSubset =
    nodeEntries.length > limitedNodeEntries.length || edges.length > limitedEdges.length;

  return {
    elements,
    isSubset,
    previewNodeCount: limitedNodeEntries.length,
    previewEdgeCount: limitedEdges.length,
  };
}

function GraphView({ graph }) {
  const { elements, isSubset, previewNodeCount, previewEdgeCount } =
    buildGraphPreview(graph);

  function handleCy(cy) {
    if (cy.scratch("_graphHandlersAttached")) {
      return;
    }

    cy.scratch("_graphHandlersAttached", true);
    cy.on("tap", "node", (event) => {
      cy.nodes().removeClass("is-active");
      event.target.addClass("is-active");
      console.log("Graph node clicked:", event.target.id());
    });
    cy.on("tap", (event) => {
      if (event.target === cy) {
        cy.nodes().removeClass("is-active");
      }
    });
  }

  if (!previewNodeCount || !previewEdgeCount) {
    return (
      <div className="graph-visualization-empty">
        <p>No graph preview is available for visualization.</p>
      </div>
    );
  }

  return (
    <div className="graph-visualization">
      <div className="graph-visualization-header">
        <div>
          <h3>Graph Preview</h3>
          <p>
            Rendering {previewNodeCount} nodes and {previewEdgeCount} edges
            locally.
          </p>
        </div>
        {isSubset ? (
          <span className="graph-preview-warning">
            Showing preview of graph (subset)
          </span>
        ) : null}
      </div>

      <div className="graph-canvas">
        <CytoscapeComponent
          cy={handleCy}
          elements={elements}
          layout={CYTOSCAPE_LAYOUT}
          maxZoom={2.5}
          minZoom={0.2}
          pan={{ x: 0, y: 0 }}
          style={{ width: "100%", height: "100%" }}
          stylesheet={CYTOSCAPE_STYLESHEET}
          wheelSensitivity={0.18}
        />
      </div>
    </div>
  );
}

export default GraphView;
