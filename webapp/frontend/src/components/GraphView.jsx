import { useEffect, useMemo, useRef, useState } from "react";
import CytoscapeComponent from "react-cytoscapejs";

const MAX_RENDER_NODES = 500;
const MAX_RENDER_EDGES = 1000;
const FOCUS_ZOOM = 1.55;

const CYTOSCAPE_LAYOUT = {
  name: "cose",
  animate: false,
  fit: true,
  padding: 48,
  randomize: true,
  nodeRepulsion: 9000,
  idealEdgeLength: 120,
  edgeElasticity: 120,
  nestingFactor: 1.2,
  gravity: 0.2,
  numIter: 1200,
};

const CYTOSCAPE_STYLESHEET = [
  {
    selector: "node",
    style: {
      label: "data(defaultLabel)",
      "font-size": 8,
      color: "#18384c",
      "text-outline-width": 2,
      "text-outline-color": "#ffffff",
      "background-color": "data(color)",
      width: "data(size)",
      height: "data(size)",
      "border-width": 1.4,
      "border-color": "#eaf1f6",
      "text-wrap": "wrap",
      "text-max-width": 78,
      "overlay-opacity": 0,
      "transition-property": "opacity, border-width, border-color, width, height",
      "transition-duration": "120ms",
    },
  },
  {
    selector: "edge",
    style: {
      width: "data(width)",
      "line-color": "#8ba6b7",
      opacity: "data(opacity)",
      "curve-style": "bezier",
      "transition-property": "opacity, width, line-color",
      "transition-duration": "120ms",
    },
  },
  {
    selector: ".is-active",
    style: {
      "background-color": "#d95f43",
      "border-width": 4,
      "border-color": "#762d1d",
      width: "mapData(size, 10, 24, 18, 32)",
      height: "mapData(size, 10, 24, 18, 32)",
      "z-index": 999,
    },
  },
  {
    selector: ".is-neighbor",
    style: {
      "border-width": 2.5,
      "border-color": "#5c8da7",
      opacity: 1,
    },
  },
  {
    selector: ".is-faded",
    style: {
      opacity: 0.18,
    },
  },
  {
    selector: ".connected-edge",
    style: {
      opacity: 0.95,
      width: "mapData(width, 1, 5, 2, 6)",
      "line-color": "#d95f43",
      "z-index": 998,
    },
  },
  {
    selector: ".show-label",
    style: {
      label: "data(label)",
    },
  },
  {
    selector: ".is-search-match",
    style: {
      "border-width": 5,
      "border-color": "#2d8faf",
      "z-index": 1001,
    },
  },
];

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function shortenLabel(value) {
  const label = String(value || "");
  const withoutTaxon = label.replace(/^\d+\./, "");

  if (withoutTaxon.length <= 18) {
    return withoutTaxon;
  }

  return `${withoutTaxon.slice(0, 7)}...${withoutTaxon.slice(-7)}`;
}

function nodeLabelFromAttributes(nodeId, attributes = {}) {
  return (
    attributes.label ||
    attributes.name ||
    attributes.symbol ||
    attributes.gene ||
    attributes.protein ||
    nodeId
  );
}

function stableHue(seed) {
  let value = 0;
  const text = String(seed);
  for (let i = 0; i < text.length; i += 1) {
    value = (value * 31 + text.charCodeAt(i)) % 360;
  }
  return value;
}

function buildGraphPreview(graph) {
  const nodes = graph?.nodes || {};
  const edges = graph?.edges || [];
  const nodeEntries = Object.entries(nodes);
  const limitedNodeEntries = nodeEntries.slice(0, MAX_RENDER_NODES);
  const allowedNodeIds = new Set(limitedNodeEntries.map(([nodeId]) => String(nodeId)));
  const degreeMap = new Map(limitedNodeEntries.map(([nodeId]) => [String(nodeId), 0]));
  const neighborMap = new Map(limitedNodeEntries.map(([nodeId]) => [String(nodeId), new Set()]));

  const limitedEdges = [];
  for (const edge of edges) {
    const source = String(edge.source);
    const target = String(edge.target);
    if (!allowedNodeIds.has(source) || !allowedNodeIds.has(target)) {
      continue;
    }

    limitedEdges.push({
      ...edge,
      source,
      target,
    });
    degreeMap.set(source, (degreeMap.get(source) || 0) + 1);
    degreeMap.set(target, (degreeMap.get(target) || 0) + 1);
    neighborMap.get(source)?.add(target);
    neighborMap.get(target)?.add(source);

    if (limitedEdges.length >= MAX_RENDER_EDGES) {
      break;
    }
  }

  const maxDegree = Math.max(1, ...degreeMap.values());
  const nodeSummaries = limitedNodeEntries.map(([nodeId, attributes]) => {
    const id = String(nodeId);
    const label = String(nodeLabelFromAttributes(nodeId, attributes));
    const degree = degreeMap.get(id) || 0;
    const degreeNorm = degree / maxDegree;
    const hue = stableHue(id);

    return {
      id,
      label,
      shortLabel: shortenLabel(label),
      labelLower: label.toLowerCase(),
      idLower: id.toLowerCase(),
      degree,
      neighborCount: neighborMap.get(id)?.size || 0,
      size: 13 + degreeNorm * 11,
      color: `hsl(${190 + (hue % 24)} 48% ${34 + Math.round(degreeNorm * 10)}%)`,
    };
  });

  const nodeLookup = new Map(nodeSummaries.map((node) => [node.id, node]));
  const elements = [
    ...nodeSummaries.map((node) => ({
      data: {
        id: node.id,
        label: node.label,
        defaultLabel: node.shortLabel,
        degree: node.degree,
        neighborCount: node.neighborCount,
        size: node.size,
        color: node.color,
      },
    })),
    ...limitedEdges.map((edge, index) => {
      const rawWeight = Number(edge.attributes?.weight || edge.weight || 1);
      const width = 1 + clamp(Number.isFinite(rawWeight) ? rawWeight : 1, 0, 4);

      return {
        data: {
          id: `${edge.source}-${edge.target}-${index}`,
          source: edge.source,
          target: edge.target,
          width,
          opacity: 0.55,
        },
      };
    }),
  ];

  const totalNodes = Number(graph?.total_nodes || nodeEntries.length);
  const totalEdges = Number(graph?.total_edges || edges.length);
  const isSubset =
    Boolean(graph?.preview_capped) ||
    totalNodes > limitedNodeEntries.length ||
    totalEdges > limitedEdges.length ||
    nodeEntries.length > limitedNodeEntries.length ||
    edges.length > limitedEdges.length;

  return {
    elements,
    nodeSummaries,
    nodeLookup,
    isSubset,
    totalNodes,
    totalEdges,
    previewNodeCount: limitedNodeEntries.length,
    previewEdgeCount: limitedEdges.length,
  };
}

function GraphView({ graph }) {
  const graphModel = useMemo(() => buildGraphPreview(graph), [graph]);
  const cyRef = useRef(null);
  const flashTimeoutRef = useRef(null);
  const selectedIdRef = useRef(null);
  const hoveredIdRef = useRef(null);
  const [selectedNodeId, setSelectedNodeId] = useState(null);
  const [hoveredNodeId, setHoveredNodeId] = useState(null);
  const [tooltip, setTooltip] = useState(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchFeedback, setSearchFeedback] = useState("");

  const {
    elements,
    isSubset,
    totalNodes,
    totalEdges,
    previewNodeCount,
    previewEdgeCount,
  } = graphModel;

  useEffect(() => {
    selectedIdRef.current = selectedNodeId;
  }, [selectedNodeId]);

  useEffect(() => {
    hoveredIdRef.current = hoveredNodeId;
  }, [hoveredNodeId]);

  useEffect(() => {
    setSelectedNodeId(null);
    setHoveredNodeId(null);
    setTooltip(null);
    setSearchQuery("");
    setSearchFeedback("");
  }, [graph]);

  useEffect(() => () => {
    if (flashTimeoutRef.current) {
      clearTimeout(flashTimeoutRef.current);
    }
  }, []);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;

    cy.batch(() => {
      cy.elements().removeClass(
        "is-faded is-neighbor is-active connected-edge show-label",
      );

      const activeId = hoveredNodeId || selectedNodeId;
      if (!activeId) {
        return;
      }

      const activeNode = cy.getElementById(String(activeId));
      if (!activeNode || activeNode.length === 0) {
        return;
      }

      cy.elements().addClass("is-faded");
      activeNode.closedNeighborhood().removeClass("is-faded");
      activeNode.connectedEdges().addClass("connected-edge");
      activeNode.neighborhood("node").addClass("is-neighbor");
      activeNode.addClass("is-active show-label");
    });
  }, [hoveredNodeId, selectedNodeId, elements]);

  function buildTooltipPayload(node) {
    const data = node.data();
    const renderedPos = node.renderedPosition();
    return {
      id: String(data.id),
      label: String(data.label || data.id),
      degree: Number(data.degree || 0),
      neighborCount: Number(data.neighborCount || 0),
      x: renderedPos.x,
      y: renderedPos.y,
    };
  }

  function syncTooltipForNodeId(nodeId) {
    const cy = cyRef.current;
    if (!cy || !nodeId) return;
    const node = cy.getElementById(String(nodeId));
    if (node && node.length > 0) {
      setTooltip(buildTooltipPayload(node));
    }
  }

  function flashNode(node) {
    const cy = cyRef.current;
    if (!cy) return;
    cy.nodes().removeClass("is-search-match");
    node.addClass("is-search-match");
    if (flashTimeoutRef.current) {
      clearTimeout(flashTimeoutRef.current);
    }
    flashTimeoutRef.current = setTimeout(() => {
      cy.nodes().removeClass("is-search-match");
    }, 1600);
  }

  function focusNode(nodeId, options = {}) {
    const cy = cyRef.current;
    if (!cy || !nodeId) return false;
    const node = cy.getElementById(String(nodeId));
    if (!node || node.length === 0) return false;

    const nextZoom = clamp(Math.max(cy.zoom(), FOCUS_ZOOM), cy.minZoom(), cy.maxZoom());
    cy.center(node);
    cy.zoom({ level: nextZoom, position: node.position() });
    setSelectedNodeId(String(nodeId));
    setHoveredNodeId(null);
    setTooltip(buildTooltipPayload(node));

    if (options.flash !== false) {
      flashNode(node);
    }

    return true;
  }

  function handleSearchSubmit(event) {
    event.preventDefault();
    const query = searchQuery.trim().toLowerCase();
    if (!query) {
      setSearchFeedback("Enter a rendered node ID or label.");
      return;
    }

    const match =
      graphModel.nodeSummaries.find(
        (node) => node.idLower === query || node.labelLower === query,
      ) ||
      graphModel.nodeSummaries.find(
        (node) => node.idLower.includes(query) || node.labelLower.includes(query),
      );

    if (!match || !focusNode(match.id)) {
      setSearchFeedback(`No rendered node matches "${searchQuery}".`);
      return;
    }

    setSearchFeedback(`Focused ${match.shortLabel}.`);
  }

  function clearSelection() {
    setSelectedNodeId(null);
    setHoveredNodeId(null);
    setTooltip(null);
    setSearchFeedback("");
  }

  function fitGraph() {
    const cy = cyRef.current;
    if (!cy) return;
    cy.fit(undefined, 48);
    clearSelection();
  }

  function handleCy(cy) {
    cyRef.current = cy;
    if (cy.scratch("_graphHandlersAttached")) {
      return;
    }

    cy.scratch("_graphHandlersAttached", true);
    cy.on("mouseover", "node", (event) => {
      const node = event.target;
      setHoveredNodeId(String(node.id()));
      setTooltip(buildTooltipPayload(node));
    });
    cy.on("mouseout", "node", (event) => {
      const nodeId = String(event.target.id());
      setHoveredNodeId((current) => (current === nodeId ? null : current));
      const selectedId = selectedIdRef.current;
      if (selectedId) {
        syncTooltipForNodeId(selectedId);
      } else {
        setTooltip((current) => (current?.id === nodeId ? null : current));
      }
    });
    cy.on("tap", "node", (event) => {
      const node = event.target;
      setSelectedNodeId(String(node.id()));
      setHoveredNodeId(null);
      setTooltip(buildTooltipPayload(node));
      setSearchFeedback("");
    });
    cy.on("tap", (event) => {
      if (event.target === cy) {
        clearSelection();
      }
    });
    cy.on("pan zoom dragfree render", () => {
      const activeId = hoveredIdRef.current || selectedIdRef.current;
      if (activeId) {
        syncTooltipForNodeId(activeId);
      }
    });
  }

  const selectedNode = selectedNodeId
    ? graphModel.nodeLookup.get(selectedNodeId) || null
    : null;

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
            Rendering {previewNodeCount} of {totalNodes.toLocaleString()} nodes and{" "}
            {previewEdgeCount} of {totalEdges.toLocaleString()} edges locally.
          </p>
        </div>
        {isSubset ? (
          <span className="graph-preview-warning">
            Preview capped for performance
          </span>
        ) : null}
      </div>

      <div className="graph-preview-toolbar">
        <form className="graph-search-form" onSubmit={handleSearchSubmit}>
          <label className="graph-search-label" htmlFor="graph-preview-search">
            Find node
          </label>
          <input
            className="graph-search-input"
            id="graph-preview-search"
            list="graph-preview-node-options"
            onChange={(event) => {
              setSearchQuery(event.target.value);
              if (searchFeedback) setSearchFeedback("");
            }}
            placeholder="Search ID or label"
            type="search"
            value={searchQuery}
          />
          <datalist id="graph-preview-node-options">
            {graphModel.nodeSummaries.map((node) => (
              <option key={node.id} value={node.label}>
                {node.id}
              </option>
            ))}
          </datalist>
          <button className="secondary-button" type="submit">
            Focus
          </button>
        </form>

        <div className="graph-toolbar-actions">
          <button className="secondary-button" onClick={fitGraph} type="button">
            Fit
          </button>
          {selectedNode ? (
            <button className="secondary-button" onClick={clearSelection} type="button">
              Clear
            </button>
          ) : null}
        </div>
      </div>

      {searchFeedback ? (
        <div className={`graph-feedback${searchFeedback.startsWith("No ") ? " is-error" : ""}`}>
          {searchFeedback}
        </div>
      ) : null}

      <div className="graph-preview-grid">
        <div className="graph-preview-main">
          <div className="graph-canvas graph-canvas--preview">
            <CytoscapeComponent
              cy={handleCy}
              elements={elements}
              layout={CYTOSCAPE_LAYOUT}
              maxZoom={2.8}
              minZoom={0.18}
              pan={{ x: 0, y: 0 }}
              style={{ width: "100%", height: "100%" }}
              stylesheet={CYTOSCAPE_STYLESHEET}
              wheelSensitivity={0.18}
            />
            {tooltip ? (
              <div
                aria-live="polite"
                className="graph-tooltip"
                role="status"
                style={{
                  left: `${tooltip.x + 12}px`,
                  top: `${tooltip.y + 12}px`,
                }}
              >
                <strong>{tooltip.label}</strong>
                <span>ID: {tooltip.id}</span>
                <span>Degree: {tooltip.degree}</span>
                <span>Neighbors: {tooltip.neighborCount}</span>
              </div>
            ) : null}
          </div>
        </div>

        <aside className="graph-side-card graph-preview-detail">
          <h4>Node details</h4>
          {selectedNode ? (
            <dl className="graph-detail-list">
              <div>
                <dt>ID</dt>
                <dd>{selectedNode.id}</dd>
              </div>
              <div>
                <dt>Label</dt>
                <dd>{selectedNode.label}</dd>
              </div>
              <div>
                <dt>Degree</dt>
                <dd>{selectedNode.degree}</dd>
              </div>
              <div>
                <dt>Neighbors</dt>
                <dd>{selectedNode.neighborCount}</dd>
              </div>
            </dl>
          ) : (
            <p className="graph-side-empty">
              No node selected.
            </p>
          )}
        </aside>
      </div>
    </div>
  );
}

export default GraphView;
