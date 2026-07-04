import { useEffect, useId, useMemo, useRef, useState } from "react";
import cytoscape from "cytoscape";
import CytoscapeComponent from "react-cytoscapejs";

const MIN_NODE_SIZE = 10;
const MAX_NODE_SIZE = 26;
const SMALL_LAYOUT_LIMIT = 1000;
const MEDIUM_LAYOUT_LIMIT = 5000;
const LABEL_REDUCTION_LIMIT = 120;
const FOCUS_ZOOM = 1.6;

const COMMUNITY_COLOR_CACHE = new Map();

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

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
  return lerpColor([152, 193, 217], [47, 111, 126], t);
}

function bfsDepthGradient(t) {
  return lerpColor([47, 111, 126], [152, 193, 217], t);
}

function formatNumber(value, digits = 4) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "-";
  if (Number.isInteger(numeric)) return numeric.toLocaleString();
  return numeric.toFixed(digits);
}

function formatCompact(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "-";
  if (Math.abs(numeric) >= 1000) return numeric.toLocaleString();
  return numeric.toFixed(3);
}

function stableHue(seed) {
  let value = 0;
  const text = String(seed);
  for (let i = 0; i < text.length; i += 1) {
    value = (value * 31 + text.charCodeAt(i)) % 360;
  }
  return value;
}

function colorForCommunity(community) {
  const key = community ?? "default";
  if (COMMUNITY_COLOR_CACHE.has(key)) {
    return COMMUNITY_COLOR_CACHE.get(key);
  }
  const hue = stableHue(key);
  const saturation = 58 + (hue % 12);
  const lightness = 44 + (hue % 10);
  const color = `hsl(${hue} ${saturation}% ${lightness}%)`;
  COMMUNITY_COLOR_CACHE.set(key, color);
  return color;
}

function hasFcoseLayout() {
  try {
    return Boolean(cytoscape("layout", "fcose"));
  } catch {
    return false;
  }
}

const FCOSE_AVAILABLE = hasFcoseLayout();

function buildPresetPosition(index, total, community) {
  const angle = (((index * 137.508) + stableHue(community ?? index)) % 360) * (Math.PI / 180);
  const radius = 48 + Math.sqrt(index + 1) * 22 + ((stableHue(community ?? index) % 7) * 6);
  const scale = total > 0 ? Math.max(1, Math.log10(total + 1)) : 1;
  return {
    x: Math.cos(angle) * radius * scale,
    y: Math.sin(angle) * radius * scale,
  };
}

function pickLayout(nodeCount) {
  if (nodeCount > MEDIUM_LAYOUT_LIMIT) {
    return {
      name: "preset",
      fit: true,
      padding: 48,
      animate: false,
    };
  }
  if (nodeCount >= SMALL_LAYOUT_LIMIT) {
    if (FCOSE_AVAILABLE) {
      return {
        name: "fcose",
        animate: false,
        fit: true,
        padding: 40,
        quality: "draft",
        randomize: true,
      };
    }
    return {
      name: "concentric",
      animate: false,
      fit: true,
      padding: 40,
      minNodeSpacing: 10,
      avoidOverlap: true,
    };
  }
  return {
    name: "cose",
    animate: false,
    fit: true,
    padding: 32,
  };
}

function buildLegendItems(algorithmName, hasWeightedEdges) {
  if (algorithmName === "bfs") {
    return [
      {
        title: "Node size",
        description: "Larger nodes are closer to the BFS source.",
        type: "size",
      },
      {
        title: "Distance color",
        description: "Color shifts from near nodes to far nodes.",
        type: "gradient",
        startLabel: "Near",
        endLabel: "Far",
      },
      {
        title: "Highlight",
        description: "The source node is highlighted for quick orientation.",
        type: "highlight",
      },
    ];
  }

  if (algorithmName === "louvain" || algorithmName === "mcl") {
    return [
      {
        title: "Communities",
        description: "Node colors group rendered communities.",
        type: "communities",
      },
      {
        title: "Highlight",
        description: "Highlighted nodes mark the dominant rendered community.",
        type: "highlight",
      },
    ];
  }

  const items = [
    {
      title: "Node size",
      description: "Larger nodes carry stronger scores.",
      type: "size",
    },
    {
      title: "Score color",
      description: "Color moves from low score to high score.",
      type: "gradient",
      startLabel: "Low",
      endLabel: "High",
    },
    {
      title: "Highlight",
      description: "Highlighted nodes show the top-ranked results.",
      type: "highlight",
    },
  ];

  if (hasWeightedEdges) {
    items.push({
      title: "Edge weights",
      description: "Thicker, darker edges represent larger weights.",
      type: "edge-weight",
    });
  }

  return items;
}

function buildGuidance(algorithmName) {
  const common = [
    "Search by node ID or label to center and inspect a node.",
    "Drag to pan and use the mouse wheel to zoom.",
    "Click a node to keep its details panel open.",
    "Use arrow keys inside the graph to move through nodes, then press Escape to clear.",
  ];

  if (algorithmName === "bfs") {
    return [
      "BFS uses node distance as the visual score, so smaller distances look more prominent.",
      "Neighbor highlighting helps trace the local cascade around the hovered or selected node.",
      ...common,
    ];
  }

  if (algorithmName === "louvain" || algorithmName === "mcl") {
    return [
      "Community colors are stable within this session, which makes repeated inspection easier.",
      "Large rendered communities stay distinguishable even when the color count grows.",
      ...common,
    ];
  }

  return [
    "Node size and color both reflect score strength for fast ranking comparisons.",
    "Highlighted nodes stay labeled on dense graphs so the most important results remain readable.",
    ...common,
  ];
}

function buildGraphModel(graphData, algorithmName) {
  const nodes = Array.isArray(graphData?.nodes) ? graphData.nodes : [];
  const edges = Array.isArray(graphData?.edges) ? graphData.edges : [];
  const isCluster = algorithmName === "louvain" || algorithmName === "mcl";
  const isBfs = algorithmName === "bfs";
  const isScoreAlgo =
    algorithmName === "pagerank" || algorithmName === "rwr" || algorithmName === "hits";

  const finiteScores = nodes
    .map((node) => Number(node.score))
    .filter((score) => Number.isFinite(score));
  const minScore = finiteScores.length ? Math.min(...finiteScores) : 0;
  const maxScore = finiteScores.length ? Math.max(...finiteScores) : 1;
  const scoreSpread = Math.max(maxScore - minScore, 1e-12);
  const reduceLabels = nodes.length > LABEL_REDUCTION_LIMIT;

  const weightValues = edges
    .map((edge) => Number(edge.weight))
    .filter((weight) => Number.isFinite(weight));
  const minWeight = weightValues.length ? Math.min(...weightValues) : 1;
  const maxWeight = weightValues.length ? Math.max(...weightValues) : 1;
  const weightSpread = Math.max(maxWeight - minWeight, 1e-12);
  const hasWeightedEdges = weightValues.length > 0;
  const hasWeightVariation = hasWeightedEdges && Math.abs(maxWeight - minWeight) > 1e-9;

  const nodeIds = new Set(nodes.map((node) => String(node.id)));
  const degreeMap = new Map();
  const neighborMap = new Map();
  for (const node of nodes) {
    degreeMap.set(String(node.id), 0);
    neighborMap.set(String(node.id), new Set());
  }

  for (const edge of edges) {
    const source = String(edge.source);
    const target = String(edge.target);
    if (!nodeIds.has(source) || !nodeIds.has(target)) {
      continue;
    }
    degreeMap.set(source, (degreeMap.get(source) || 0) + 1);
    degreeMap.set(target, (degreeMap.get(target) || 0) + 1);
    neighborMap.get(source)?.add(target);
    neighborMap.get(target)?.add(source);
  }

  const nodeSummaries = nodes.map((node) => {
    const id = String(node.id);
    const label = String(node.label || node.id);
    const score = Number(node.score);
    const scoreNorm = finiteScores.length && Number.isFinite(score)
      ? clamp((score - minScore) / scoreSpread, 0, 1)
      : 0.5;
    const sizeNorm = isBfs ? 1 - scoreNorm : scoreNorm;
    const size = MIN_NODE_SIZE + (MAX_NODE_SIZE - MIN_NODE_SIZE) * sizeNorm;
    const isHighlight = Boolean(node.highlight);

    let color = "#2f6f7e";
    if (isCluster) {
      color = colorForCommunity(node.community);
    } else if (isBfs) {
      color = isHighlight ? "#d95f43" : bfsDepthGradient(scoreNorm);
    } else if (isScoreAlgo) {
      color = isHighlight ? "#d95f43" : scoreGradient(scoreNorm);
    }

    return {
      id,
      label,
      labelLower: label.toLowerCase(),
      idLower: id.toLowerCase(),
      score,
      community: node.community ?? null,
      highlight: isHighlight,
      degree: degreeMap.get(id) || 0,
      neighborCount: neighborMap.get(id)?.size || 0,
      size,
      color,
      defaultLabel: reduceLabels && !isHighlight ? "" : label,
    };
  });

  const nodeLookup = new Map(nodeSummaries.map((node) => [node.id, node]));
  const nodeElements = nodeSummaries.map((node, index) => {
    const element = {
      data: {
        id: node.id,
        label: node.label,
        defaultLabel: node.defaultLabel,
        score: Number.isFinite(node.score) ? node.score : 0,
        community: node.community,
        highlight: node.highlight,
        size: node.size,
        color: node.color,
      },
      classes: node.highlight ? "is-highlight" : "",
    };

    if (nodes.length > MEDIUM_LAYOUT_LIMIT) {
      element.position = buildPresetPosition(index, nodes.length, node.community);
    }

    return element;
  });

  const edgeElements = edges.map((edge, index) => {
    const rawWeight = Number(edge.weight);
    const weightNorm = hasWeightVariation && Number.isFinite(rawWeight)
      ? clamp((rawWeight - minWeight) / weightSpread, 0, 1)
      : 0.35;
    return {
      data: {
        id: `${edge.source}-${edge.target}-${index}`,
        source: String(edge.source),
        target: String(edge.target),
        weight: Number.isFinite(rawWeight) ? rawWeight : 1,
        width: hasWeightedEdges ? lerp(1.2, 5.2, weightNorm) : 1.3,
        opacity: hasWeightedEdges ? lerp(0.35, 0.85, weightNorm) : 0.6,
      },
    };
  });

  const communitySizes = new Map();
  for (const node of nodeSummaries) {
    if (node.community === null || node.community === undefined) {
      continue;
    }
    communitySizes.set(node.community, (communitySizes.get(node.community) || 0) + 1);
  }
  const sortedCommunities = [...communitySizes.entries()].sort((a, b) => b[1] - a[1]);

  const averageScore = finiteScores.length
    ? finiteScores.reduce((sum, value) => sum + value, 0) / finiteScores.length
    : null;
  const bfsSource = nodeSummaries.find((node) => node.highlight)?.label || "-";
  const avgDepth = isBfs && finiteScores.length
    ? finiteScores.reduce((sum, value) => sum + value, 0) / finiteScores.length
    : null;

  const stats = [
    { label: "Rendered nodes", value: nodes.length.toLocaleString() },
    { label: "Rendered edges", value: edges.length.toLocaleString() },
    { label: "Capping", value: graphData?.node_count_capped ? "Active" : "Off" },
  ];

  if (graphData?.node_count_capped) {
    stats.push({
      label: "Original nodes",
      value: Number(graphData?.total_nodes_in_graph || 0).toLocaleString(),
    });
  }

  if (isCluster) {
    stats.push(
      { label: "Communities", value: sortedCommunities.length.toLocaleString() },
      { label: "Largest community", value: sortedCommunities.length ? sortedCommunities[0][1].toLocaleString() : "-" },
    );
  } else if (isBfs) {
    stats.push(
      { label: "Source node", value: bfsSource },
      { label: "Max depth", value: finiteScores.length ? String(Math.max(...finiteScores)) : "-" },
      { label: "Average depth", value: avgDepth == null ? "-" : formatCompact(avgDepth) },
    );
  } else if (isScoreAlgo) {
    stats.push(
      { label: "Highest score", value: finiteScores.length ? formatCompact(maxScore) : "-" },
      { label: "Lowest score", value: finiteScores.length ? formatCompact(minScore) : "-" },
      { label: "Average score", value: averageScore == null ? "-" : formatCompact(averageScore) },
    );
  }

  if (hasWeightedEdges) {
    stats.push({
      label: "Edge weights",
      value: hasWeightVariation
        ? `${formatCompact(minWeight)} to ${formatCompact(maxWeight)}`
        : formatCompact(minWeight),
    });
  }

  return {
    elements: [...nodeElements, ...edgeElements],
    nodeSummaries,
    nodeLookup,
    edgeCount: edges.length,
    nodeCount: nodes.length,
    isCapped: Boolean(graphData?.node_count_capped),
    totalNodes: Number(graphData?.total_nodes_in_graph || nodes.length),
    reduceLabels,
    hasWeightedEdges,
    stats,
    legendItems: buildLegendItems(algorithmName, hasWeightedEdges),
    guidance: buildGuidance(algorithmName),
    layout: pickLayout(nodes.length),
  };
}

const STYLESHEET = [
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
      "border-width": 1.2,
      "border-color": "#ebf3f8",
      opacity: 1,
      "overlay-opacity": 0,
      "text-wrap": "wrap",
      "text-max-width": 72,
      "transition-property": "opacity, border-width, border-color, width, height",
      "transition-duration": "120ms",
    },
  },
  {
    selector: "edge",
    style: {
      width: "data(width)",
      opacity: "data(opacity)",
      "line-color": "#8ba6b7",
      "curve-style": "bezier",
      "target-arrow-shape": "none",
      "transition-property": "opacity, width, line-color",
      "transition-duration": "120ms",
    },
  },
  {
    selector: ".is-highlight",
    style: {
      "border-width": 4,
      "border-color": "#fac748",
      "z-index": 999,
    },
  },
  {
    selector: ".is-hovered",
    style: {
      "border-width": 4,
      "border-color": "#d95f43",
      "z-index": 1001,
    },
  },
  {
    selector: ".is-selected",
    style: {
      "border-width": 4,
      "border-color": "#1f3f55",
      "z-index": 1002,
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
    selector: ".connected-edge",
    style: {
      opacity: 0.95,
      width: "mapData(width, 1, 6, 2, 7)",
      "line-color": "#d95f43",
      "z-index": 998,
    },
  },
  {
    selector: ".is-faded",
    style: {
      opacity: 0.14,
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
      "z-index": 1003,
    },
  },
];

function LegendSwatch({ item }) {
  if (item.type === "gradient") {
    const background = item.startLabel === "Near"
      ? "linear-gradient(90deg, rgb(47, 111, 126), rgb(152, 193, 217))"
      : "linear-gradient(90deg, rgb(152, 193, 217), rgb(47, 111, 126))";
    return (
      <div className="graph-legend-visual">
        <span className="graph-gradient-bar" style={{ background }} />
        <span>{item.startLabel}</span>
        <span>{item.endLabel}</span>
      </div>
    );
  }

  if (item.type === "communities") {
    return (
      <div className="graph-legend-visual graph-community-sample" aria-hidden="true">
        <span style={{ background: colorForCommunity(0) }} />
        <span style={{ background: colorForCommunity(1) }} />
        <span style={{ background: colorForCommunity(2) }} />
      </div>
    );
  }

  if (item.type === "edge-weight") {
    return (
      <div className="graph-legend-visual graph-edge-weight-sample" aria-hidden="true">
        <span className="graph-edge-thin" />
        <span className="graph-edge-thick" />
      </div>
    );
  }

  if (item.type === "highlight") {
    return <span className="graph-highlight-chip" aria-hidden="true">Highlighted</span>;
  }

  return (
    <div className="graph-legend-visual graph-size-sample" aria-hidden="true">
      <span className="graph-size-small" />
      <span className="graph-size-large" />
    </div>
  );
}

function GraphHighlight({ graphData, algorithmName }) {
  const graphModel = useMemo(
    () => buildGraphModel(graphData, algorithmName),
    [graphData, algorithmName],
  );
  const searchListId = useId();
  const cyRef = useRef(null);
  const flashTimeoutRef = useRef(null);
  const graphModelRef = useRef(graphModel);
  const selectedIdRef = useRef(null);
  const hoveredIdRef = useRef(null);
  const [tooltip, setTooltip] = useState(null);
  const [selectedNodeId, setSelectedNodeId] = useState(null);
  const [hoveredNodeId, setHoveredNodeId] = useState(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchFeedback, setSearchFeedback] = useState("");
  const [statusMessage, setStatusMessage] = useState("");

  useEffect(() => {
    graphModelRef.current = graphModel;
  }, [graphModel]);

  useEffect(() => {
    selectedIdRef.current = selectedNodeId;
  }, [selectedNodeId]);

  useEffect(() => {
    hoveredIdRef.current = hoveredNodeId;
  }, [hoveredNodeId]);

  useEffect(() => {
    setTooltip(null);
    setSelectedNodeId(null);
    setHoveredNodeId(null);
    setSearchQuery("");
    setSearchFeedback("");
    setStatusMessage("");
  }, [graphData, algorithmName]);

  useEffect(() => () => {
    if (flashTimeoutRef.current) {
      clearTimeout(flashTimeoutRef.current);
    }
  }, []);

  const selectedNode = selectedNodeId ? graphModel.nodeLookup.get(selectedNodeId) || null : null;

  function buildTooltipPayload(node) {
    const data = node.data();
    const id = String(data.id);
    const details = graphModelRef.current.nodeLookup.get(id);
    const renderedPos = node.renderedPosition();
    return {
      id,
      x: renderedPos.x,
      y: renderedPos.y,
      label: data.label,
      score: data.score,
      community: data.community,
      highlight: Boolean(data.highlight),
      degree: details?.degree ?? 0,
      neighborCount: details?.neighborCount ?? 0,
    };
  }

  function syncTooltipForNodeId(nodeId) {
    const cy = cyRef.current;
    if (!cy || !nodeId) return;
    const node = cy.getElementById(String(nodeId));
    if (!node || node.length === 0) return;
    setTooltip(buildTooltipPayload(node));
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
    }, 1800);
  }

  function focusNodeById(nodeId, options = {}) {
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

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;

    cy.batch(() => {
      cy.elements().removeClass(
        "is-faded is-neighbor is-hovered is-selected connected-edge show-label",
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
      const activeNeighborhood = activeNode.closedNeighborhood();
      const activeEdges = activeNode.connectedEdges();
      const activeNeighbors = activeNode.neighborhood("node");

      activeNeighborhood.removeClass("is-faded");
      activeEdges.removeClass("is-faded").addClass("connected-edge");
      activeNode.addClass("show-label");
      activeNeighbors.addClass("is-neighbor");

      if (hoveredNodeId) {
        activeNode.addClass("is-hovered");
      }
      if (selectedNodeId) {
        const selected = cy.getElementById(String(selectedNodeId));
        if (selected && selected.length > 0) {
          selected.removeClass("is-faded").addClass("is-selected show-label");
        }
      }
    });
  }, [selectedNodeId, hoveredNodeId, graphModel.elements]);

  function handleSearchSubmit(event) {
    event.preventDefault();
    const query = searchQuery.trim().toLowerCase();
    if (!query) {
      setSearchFeedback("Enter a node ID or label to search.");
      return;
    }

    const exact = graphModel.nodeSummaries.find(
      (node) => node.idLower === query || node.labelLower === query,
    );
    const partial = graphModel.nodeSummaries.find(
      (node) => node.idLower.includes(query) || node.labelLower.includes(query),
    );
    const match = exact || partial;

    if (!match || !focusNodeById(match.id)) {
      setSearchFeedback(`No rendered node matches "${searchQuery}".`);
      setStatusMessage(`Search failed for ${searchQuery}.`);
      return;
    }

    setSearchFeedback(`Focused node ${match.label}.`);
    setStatusMessage(`Focused node ${match.label}.`);
  }

  function handleExport() {
    const cy = cyRef.current;
    if (!cy) return;

    const dataUrl = cy.png({
      bg: "#ffffff",
      full: false,
      scale: 2,
    });

    const image = new Image();
    image.onload = () => {
      const title = `Graph Highlight - ${algorithmName}`;
      const canvas = document.createElement("canvas");
      canvas.width = image.width;
      canvas.height = image.height + 72;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        return;
      }

      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#17384b";
      ctx.font = "bold 30px Arial";
      ctx.fillText(title, 28, 38);
      ctx.fillStyle = "#5f7384";
      ctx.font = "18px Arial";
      ctx.fillText("Current zoom and pan preserved", 28, 62);
      ctx.drawImage(image, 0, 72);

      const link = document.createElement("a");
      link.href = canvas.toDataURL("image/png");
      link.download = `graph-highlight-${algorithmName}.png`;
      link.click();
      setStatusMessage(`Exported graph snapshot for ${algorithmName}.`);
    };
    image.src = dataUrl;
  }

  function handleCanvasKeyDown(event) {
    const ids = graphModel.nodeSummaries.map((node) => node.id);
    if (ids.length === 0) return;

    const currentId = selectedNodeId || hoveredNodeId || ids[0];
    const currentIndex = Math.max(ids.indexOf(currentId), 0);
    let nextIndex = currentIndex;

    if (event.key === "ArrowRight" || event.key === "ArrowDown") {
      nextIndex = (currentIndex + 1) % ids.length;
    } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
      nextIndex = (currentIndex - 1 + ids.length) % ids.length;
    } else if (event.key === "Home") {
      nextIndex = 0;
    } else if (event.key === "End") {
      nextIndex = ids.length - 1;
    } else if (event.key === "Escape") {
      setSelectedNodeId(null);
      setHoveredNodeId(null);
      setTooltip(null);
      setStatusMessage("Cleared graph selection.");
      return;
    } else {
      return;
    }

    event.preventDefault();
    const nextId = ids[nextIndex];
    if (focusNodeById(nextId, { flash: false })) {
      const nextNode = graphModel.nodeLookup.get(nextId);
      setStatusMessage(`Focused node ${nextNode?.label || nextId}.`);
    }
  }

  function handleCy(cy) {
    cyRef.current = cy;
    if (cy.scratch("_highlightHandlersAttached")) return;
    cy.scratch("_highlightHandlersAttached", true);

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
      const nodeId = String(node.id());
      setSelectedNodeId(nodeId);
      setHoveredNodeId(null);
      setTooltip(buildTooltipPayload(node));
      setSearchFeedback("");
      setStatusMessage(`Selected node ${node.data("label")}.`);
    });

    cy.on("tap", (event) => {
      if (event.target !== cy) return;
      setSelectedNodeId(null);
      setHoveredNodeId(null);
      setTooltip(null);
      setStatusMessage("Cleared graph selection.");
    });

    const syncTooltip = () => {
      const activeId = hoveredIdRef.current || selectedIdRef.current;
      if (activeId) {
        syncTooltipForNodeId(activeId);
      }
    };

    cy.on("pan zoom dragfree render", syncTooltip);
  }

  if (graphData?.unsupported || graphModel.nodeCount === 0) {
    return (
      <div className="viz-empty">
        {graphData?.reason || "Graph view is not available for this algorithm."}
      </div>
    );
  }

  return (
    <div className="graph-visualization">
      <div className="graph-visualization-header">
        <div>
          <h3>Graph Highlight</h3>
          <p>
            Explore rendered nodes, scores, communities, and local neighborhoods for{" "}
            <strong>{algorithmName}</strong>.
          </p>
        </div>
        {graphModel.isCapped ? (
          <span className="graph-preview-warning">
            Showing {graphModel.nodeCount} of {graphModel.totalNodes} nodes
          </span>
        ) : null}
      </div>

      <div className="graph-stats-grid" role="list" aria-label="Graph statistics">
        {graphModel.stats.map((item) => (
          <div key={item.label} className="graph-stat-card" role="listitem">
            <span>{item.label}</span>
            <strong>{item.value}</strong>
          </div>
        ))}
      </div>

      <div className="graph-help-card" id="graph-help-text">
        <strong>Graph guide</strong>
        <ul className="graph-help-list">
          {graphModel.guidance.map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>
      </div>

      <div className="graph-toolbar">
        <form className="graph-search-form" onSubmit={handleSearchSubmit}>
          <label className="graph-search-label" htmlFor={`graph-search-${searchListId}`}>
            Find node
          </label>
          <input
            id={`graph-search-${searchListId}`}
            aria-describedby="graph-help-text"
            aria-label="Search nodes by ID or label"
            className="graph-search-input"
            list={`graph-options-${searchListId}`}
            onChange={(event) => {
              setSearchQuery(event.target.value);
              if (searchFeedback) setSearchFeedback("");
            }}
            placeholder="Search by node ID or label"
            type="search"
            value={searchQuery}
          />
          <datalist id={`graph-options-${searchListId}`}>
            {graphModel.nodeSummaries.map((node) => (
              <option key={node.id} value={node.label}>
                {node.id}
              </option>
            ))}
          </datalist>
          <button className="secondary-button" type="submit">
            Focus
          </button>
          <button
            className="secondary-button"
            onClick={() => {
              setSearchQuery("");
              setSearchFeedback("");
            }}
            type="button"
          >
            Clear
          </button>
        </form>

        <div className="graph-toolbar-actions">
          <button
            aria-label="Export graph snapshot as PNG"
            className="secondary-button"
            onClick={handleExport}
            type="button"
          >
            Export PNG
          </button>
          {selectedNode ? (
            <button
              aria-label="Clear selected node"
              className="secondary-button"
              onClick={() => {
                setSelectedNodeId(null);
                setTooltip(null);
                setStatusMessage("Cleared selected node.");
              }}
              type="button"
            >
              Clear Selection
            </button>
          ) : null}
        </div>
      </div>

      {searchFeedback ? (
        <div className={`graph-feedback${searchFeedback.startsWith("No ") ? " is-error" : ""}`}>
          {searchFeedback}
        </div>
      ) : null}

      <div className="graph-dashboard-grid">
        <div className="graph-main-column">
          <div
            aria-describedby="graph-help-text"
            aria-label="Interactive graph canvas"
            className="graph-canvas"
            onKeyDown={handleCanvasKeyDown}
            role="application"
            style={{ position: "relative" }}
            tabIndex={0}
          >
            <CytoscapeComponent
              cy={handleCy}
              elements={graphModel.elements}
              layout={graphModel.layout}
              maxZoom={2.5}
              minZoom={0.2}
              pan={{ x: 0, y: 0 }}
              style={{ width: "100%", height: "100%" }}
              stylesheet={STYLESHEET}
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
                <span>
                  {algorithmName === "bfs" ? "Distance" : "Score"}: {formatNumber(tooltip.score, 6)}
                </span>
                {tooltip.community !== null && tooltip.community !== undefined ? (
                  <span>Community: {tooltip.community}</span>
                ) : null}
                <span>Degree: {tooltip.degree}</span>
                <span>Neighbors: {tooltip.neighborCount}</span>
                {tooltip.highlight ? <span>Highlighted</span> : null}
              </div>
            ) : null}
          </div>
        </div>

        <aside className="graph-side-column">
          <section className="graph-side-card" aria-labelledby="graph-legend-title">
            <h4 id="graph-legend-title">Legend</h4>
            <div className="graph-legend-list">
              {graphModel.legendItems.map((item) => (
                <div key={item.title} className="graph-legend-item">
                  <LegendSwatch item={item} />
                  <div>
                    <strong>{item.title}</strong>
                    <p>{item.description}</p>
                  </div>
                </div>
              ))}
            </div>
          </section>

          <section className="graph-side-card" aria-labelledby="graph-node-details-title">
            <div className="graph-side-card-header">
              <h4 id="graph-node-details-title">Node details</h4>
              {selectedNode ? (
                <button
                  className="text-button"
                  onClick={() => {
                    setSelectedNodeId(null);
                    setTooltip(null);
                  }}
                  type="button"
                >
                  Clear
                </button>
              ) : null}
            </div>
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
                  <dt>{algorithmName === "bfs" ? "Distance" : "Score"}</dt>
                  <dd>{formatNumber(selectedNode.score, 6)}</dd>
                </div>
                <div>
                  <dt>Community</dt>
                  <dd>{selectedNode.community ?? "-"}</dd>
                </div>
                <div>
                  <dt>Highlighted</dt>
                  <dd>{selectedNode.highlight ? "Yes" : "No"}</dd>
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
                Click a node or use search to inspect its score, community, and local topology.
              </p>
            )}
          </section>
        </aside>
      </div>

      <div aria-live="polite" className="graph-status-message">
        {statusMessage}
      </div>
    </div>
  );
}

export default GraphHighlight;
